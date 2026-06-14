from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import tyro
import wandb
from rich.pretty import pretty_repr
from tqdm import tqdm

from scripts.ashared import create_logger, log_device_info, set_seed
from scripts.data.augmentation.create_noise_augmentation import compute_noise_std
from scripts.data.augmentation.tacil.create_data.config import Config
from scripts.data.augmentation.tacil.create_data.inverse_dynamics import (
    CONVERGENCE_TOLERANCE,
    solve_inverse_dynamics_state,
)
from scripts.data.augmentation.tacil.create_data.statistics_logging import (
    log_dataset_statistics,
)
from scripts.data.augmentation.tacil.dynamics.checkpoint import load_dynamics_model
from scripts.data.augmentation.tacil.dynamics_model import DynamicsModel
from scripts.shared.data import (
    EpisodeDataset,
    load_episode_dataset,
    upload_episode_dataset,
)
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass(frozen=True)
class AugmentationResult:
    @dataclass(frozen=True)
    class OutcomeCounts:
        not_converged: int
        rejected: int
        accepted: int

    episode_dataset: EpisodeDataset

    bounds: npt.NDArray[np.float32]  # per-step normalized bounds over converged steps
    bound_threshold: float
    counts: OutcomeCounts
    synthetic_part_length: int  # synthetic transitions per accepted episode


@dataclass(frozen=True)
class BackwardStep:
    noisy_actions: torch.Tensor  # (N, action_dimensionality)
    predecessor_states: torch.Tensor  # (N, state_dimensionality)
    converged: torch.Tensor  # (N,)
    error_bounds: torch.Tensor  # (N,)


@dataclass(frozen=True)
class BackwardTrajectories:
    anchor_alive: torch.Tensor  # (N,)

    predecessor_states: torch.Tensor  # (N, step_count, state_dimensionality)
    predecessor_actions: torch.Tensor  # (N, step_count, action_dimensionality)
    step_bounds: torch.Tensor  # (N, step_count)


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    set_seed(seed=config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project="abc",
        job_type="create-tacil-dataset",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        dynamics_model = load_dynamics_model(
            checkpoint_file_path=Path(
                run.use_artifact(artifact_or_name=config.model_artifact).download()
            )
            / "dynamics.pt",
            device=device,
        )
        dynamics_model.eval()

        episode_dataset = load_episode_dataset(
            episode_file_path=Path(
                run.use_artifact(artifact_or_name=config.dataset_artifact).download()
            )
        )

        anchor_indices = _get_anchor_indices(
            episode_end_indices=episode_dataset.episode_end_indices,
            expert_part_length=config.expert_part_length,
        )  # (anchor_count,)

        action_noise_std = compute_noise_std(
            samples=episode_dataset.actions, noise_scale=config.action_noise_scale
        )  # (action_dimensionality,)

        backward_trajectories = _roll_out_backward(
            dynamics_model=dynamics_model,
            state_batch=torch.as_tensor(episode_dataset.states[anchor_indices]).to(
                device=device
            ),
            action_batch=torch.as_tensor(episode_dataset.actions[anchor_indices]).to(
                device=device
            ),
            step_count=config.synthetic_part_length,
            action_noise_std=torch.as_tensor(action_noise_std).to(device=device),
            max_newton_iteration_count=config.maximum_newton_iterations,
        )

        augmentation_result = _create_augmented_episodes(
            backward_trajectories=backward_trajectories,
            episode_dataset=episode_dataset,
            anchor_indices=anchor_indices,
            expert_part_length=config.expert_part_length,
            label_error_bound_quantile=config.label_error_bound_quantile,
        )

        log_dataset_statistics(
            augmentation_result=augmentation_result,
            source_episode_count=len(episode_dataset.episode_end_indices),
            source_transition_count=len(episode_dataset.states),
            run=run,
        )

        upload_episode_dataset(
            episode_dataset=_combine_episode_datasets(
                expert_episode_dataset=episode_dataset,
                augmentation_episode_dataset=augmentation_result.episode_dataset,
            ),
            name="tacil-dataset",
            run=run,
        )


def _get_anchor_indices(
    *,
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
    expert_part_length: int,
) -> npt.NDArray[np.int64]:  # (anchor_count,)
    mask = np.ones(
        shape=episode_end_indices[-1], dtype=bool
    )  # (total_transition_count,)
    mask[episode_end_indices - expert_part_length] = False
    return np.flatnonzero(mask).astype(episode_end_indices.dtype, copy=False)


@torch.no_grad()
def _roll_out_backward(
    *,
    dynamics_model: DynamicsModel,
    state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    action_batch: torch.Tensor,  # (anchor_count, action_dimensionality)
    step_count: int,
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
    max_newton_iteration_count: int,
) -> BackwardTrajectories:
    device = state_batch.device
    anchor_count, state_dimensionality = state_batch.shape
    action_dimensionality = action_batch.shape[1]

    anchor_state_batch = state_batch.clone()  # (anchor_count, state_dimensionality)
    predecessor_state_batch = torch.empty(
        size=(anchor_count, step_count, state_dimensionality), device=device
    )  # (anchor_count, step_count, state_dimensionality)
    predecessor_action_batch = torch.empty(
        size=(anchor_count, step_count, action_dimensionality), device=device
    )  # (anchor_count, step_count, action_dimensionality)
    step_error_bound_batch = torch.empty(
        size=(anchor_count, step_count), device=device
    )  # (anchor_count, step_count)
    alive_mask_batch = torch.ones(
        anchor_count, dtype=torch.bool, device=device
    )  # (anchor_count,)

    for step_index in tqdm(range(step_count), desc="backward step", unit="step"):
        alive_indices = alive_mask_batch.nonzero(as_tuple=False).squeeze(
            1
        )  # (alive_count,)

        backward_step = _compute_backward_step(
            dynamics_model=dynamics_model,
            state_batch=anchor_state_batch[alive_indices],
            action_batch=action_batch[alive_indices],
            action_noise_std=action_noise_std,
            max_newton_iteration_count=max_newton_iteration_count,
        )

        predecessor_state_batch[alive_indices, step_index] = (
            backward_step.predecessor_states
        )
        predecessor_action_batch[alive_indices, step_index] = (
            backward_step.noisy_actions
        )
        step_error_bound_batch[alive_indices, step_index] = backward_step.error_bounds

        alive_mask_batch[alive_indices[~backward_step.converged]] = False
        anchor_state_batch[alive_indices[backward_step.converged]] = (
            backward_step.predecessor_states[backward_step.converged]
        )

    return BackwardTrajectories(
        predecessor_states=predecessor_state_batch,
        predecessor_actions=predecessor_action_batch,
        anchor_alive=alive_mask_batch,
        step_bounds=step_error_bound_batch,
    )


def _compute_backward_step(
    *,
    dynamics_model: DynamicsModel,
    state_batch: torch.Tensor,  # (N, state_dimensionality)
    action_batch: torch.Tensor,  # (N, action_dimensionality)
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
    max_newton_iteration_count: int,
) -> BackwardStep:
    noisy_action_batch = action_batch + torch.normal(
        mean=0.0, std=action_noise_std.expand_as(action_batch)
    )  # (N, action_dimensionality)

    predecessor_state_batch = solve_inverse_dynamics_state(
        dynamics_model=dynamics_model,
        initial_state_estimate_batch=state_batch,
        action_batch=noisy_action_batch,
        target_next_state_batch=state_batch,
        max_newton_iteration_count=max_newton_iteration_count,
    )  # (N, state_dimensionality)

    converged_batch = _check_inverse_dynamics_convergence(
        predecessor_state_batch=predecessor_state_batch,
        action_batch=noisy_action_batch,
        state_batch=state_batch,
        dynamics_model=dynamics_model,
    )  # (N,)

    error_bound_batch = _compute_label_error_bound(
        dynamics_model=dynamics_model,
        anchor_state_batch=state_batch,
        action_batch=noisy_action_batch,
        predecessor_state_batch=predecessor_state_batch,
    )  # (N,)

    return BackwardStep(
        noisy_actions=noisy_action_batch,
        predecessor_states=predecessor_state_batch,
        converged=converged_batch,
        error_bounds=error_bound_batch,
    )


def _check_inverse_dynamics_convergence(
    *,
    predecessor_state_batch: torch.Tensor,  # (N, state_dimensionality)
    action_batch: torch.Tensor,  # (N, action_dimensionality)
    state_batch: torch.Tensor,  # (N, state_dimensionality)
    dynamics_model: DynamicsModel,
) -> torch.Tensor:  # (N,)
    reconstruction_error_batch: torch.Tensor = (
        dynamics_model.normalize_state(
            state_batch=dynamics_model.predict_next_state(
                state_batch=predecessor_state_batch, action_batch=action_batch
            )
        )
        - dynamics_model.normalize_state(state_batch=state_batch)
    ).norm(dim=-1)  # (N,)

    return reconstruction_error_batch <= CONVERGENCE_TOLERANCE  # (N,)


@torch.no_grad()
def _compute_label_error_bound(
    *,
    dynamics_model: DynamicsModel,
    anchor_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    predecessor_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size,)
    local_lipschitz_constant_batch = dynamics_model.local_lipschitz_constants(
        state_batch=anchor_state_batch, action_batch=action_batch
    )  # (batch_size,)

    normalized_step_distance_batch: torch.Tensor = torch.linalg.norm(
        dynamics_model.normalize_state(state_batch=predecessor_state_batch)
        - dynamics_model.normalize_state(state_batch=anchor_state_batch),
        dim=-1,
    )  # (batch_size,)

    return (
        local_lipschitz_constant_batch * normalized_step_distance_batch
    )  # (batch_size,)


@torch.no_grad()
def _create_augmented_episodes(
    *,
    backward_trajectories: BackwardTrajectories,
    episode_dataset: EpisodeDataset,
    anchor_indices: npt.NDArray[np.int64],
    expert_part_length: int,
    label_error_bound_quantile: float,
) -> AugmentationResult:

    converged_step_bounds = backward_trajectories.step_bounds[
        backward_trajectories.anchor_alive
    ].flatten()  # (alive_count * step,)
    if converged_step_bounds.numel() == 0:
        raise RuntimeError("No backward steps converged. ")

    error_bound_threshold = float(
        torch.quantile(input=converged_step_bounds, q=label_error_bound_quantile).item()
    )

    episode_accepted = backward_trajectories.anchor_alive & (
        backward_trajectories.step_bounds <= error_bound_threshold
    ).all(dim=1)  # (anchor_count,)

    augmented_episode_dataset = _assemble_episodes(
        states=episode_dataset.states,
        actions=episode_dataset.actions,
        predecessor_states=backward_trajectories.predecessor_states.cpu().numpy(),
        predecessor_actions=backward_trajectories.predecessor_actions.cpu().numpy(),
        episode_accepted=episode_accepted.cpu().numpy(),
        anchor_positions=np.asarray(anchor_indices, dtype=np.int64),
        expert_part_length=expert_part_length,
    )

    anchor_count = backward_trajectories.predecessor_actions.shape[0]
    accepted_count = int(episode_accepted.sum().item())
    not_converged_count = int((~backward_trajectories.anchor_alive).sum().item())
    rejected_count = anchor_count - accepted_count - not_converged_count

    return AugmentationResult(
        episode_dataset=augmented_episode_dataset,
        bounds=converged_step_bounds.cpu().numpy(),
        bound_threshold=error_bound_threshold,
        counts=AugmentationResult.OutcomeCounts(
            not_converged=not_converged_count,
            rejected=rejected_count,
            accepted=accepted_count,
        ),
        synthetic_part_length=backward_trajectories.predecessor_states.shape[1],
    )


def _compute_expert_part_lengths(
    *,
    anchor_positions: npt.NDArray[np.int64],
    episode_ends: npt.NDArray[np.int64],
    target_expert_part_length: int,
) -> npt.NDArray[np.int64]:  # (anchor_count,)
    anchor_positions = np.asarray(anchor_positions, dtype=np.int64)
    episode_ends = np.asarray(episode_ends, dtype=np.int64)

    episode_index_per_anchor = np.searchsorted(
        episode_ends, anchor_positions, side="right"
    )  # (anchor_count,)
    episode_end_per_anchor = episode_ends[episode_index_per_anchor]  # (anchor_count,)
    available_per_anchor = episode_end_per_anchor - anchor_positions  # (anchor_count,)
    return np.minimum(target_expert_part_length, available_per_anchor).astype(np.int64)


def _assemble_episodes(
    *,
    states: npt.NDArray[np.float32],
    actions: npt.NDArray[np.float32],
    predecessor_states: npt.NDArray[np.float32],
    predecessor_actions: npt.NDArray[np.float32],
    episode_accepted: npt.NDArray[np.bool_],
    anchor_positions: npt.NDArray[np.int64],
    expert_part_length: int,
) -> EpisodeDataset:
    synthetic_part_length = predecessor_states.shape[1]

    episode_state_chunks: list[np.ndarray] = []
    episode_action_chunks: list[np.ndarray] = []
    episode_lengths: list[int] = []

    for anchor_index in np.nonzero(episode_accepted)[0]:
        anchor_position: int = anchor_positions[anchor_index]

        # Synthetic prefix in temporal order = predecessors reversed.
        synthetic_states = predecessor_states[anchor_index][
            ::-1
        ]  # (synthetic_part_length, state_dimensionality)
        synthetic_actions = predecessor_actions[anchor_index][
            ::-1
        ]  # (synthetic_part_length, action_dimensionality)

        expert_part_slice = slice(anchor_position, anchor_position + expert_part_length)

        episode_state_chunks.append(
            np.concatenate(
                [
                    synthetic_states,
                    states[
                        expert_part_slice
                    ],  # (expert_part_length, state_dimensionality)
                ],
                axis=0,
            )
        )
        episode_action_chunks.append(
            np.concatenate(
                [
                    synthetic_actions,
                    actions[
                        expert_part_slice
                    ],  # (expert_part_length, action_dimensionality)
                ],
                axis=0,
            )
        )
        episode_lengths.append(synthetic_part_length + expert_part_length)

    return EpisodeDataset(
        states=np.concatenate(
            episode_state_chunks, axis=0
        ),  # (total_timestep_count, state_dimensionality)
        actions=np.concatenate(
            episode_action_chunks, axis=0
        ),  # (total_timestep_count, action_dimensionality)
        episode_end_indices=np.cumsum(
            np.asarray(episode_lengths, dtype=np.int64)
        ),  # (episode_count,)
    )


def _combine_episode_datasets(
    *,
    expert_episode_dataset: EpisodeDataset,
    augmentation_episode_dataset: EpisodeDataset,
) -> EpisodeDataset:
    states = np.concatenate(
        [expert_episode_dataset.states, augmentation_episode_dataset.states], axis=0
    )  # (total_timestep_count, state_dimensionality)

    actions = np.concatenate(
        [expert_episode_dataset.actions, augmentation_episode_dataset.actions], axis=0
    )  # (total_timestep_count, action_dimensionality)

    original_timestep_count = expert_episode_dataset.states.shape[0]
    augmented_episode_ends_absolute = (
        original_timestep_count + augmentation_episode_dataset.episode_end_indices
    )  # (augmented_episode_count,)

    episode_end_indices = np.concatenate(
        [expert_episode_dataset.episode_end_indices, augmented_episode_ends_absolute],
        axis=0,
    )  # (episode_count,)

    return EpisodeDataset(
        states=states, actions=actions, episode_end_indices=episode_end_indices
    )


if __name__ == "__main__":
    main()
