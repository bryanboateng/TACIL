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
from scripts.data.augmentation.tacil.create_data.wandb_logging import (
    _log_augmentation_to_wandb,
    _log_summary_to_wandb,
)
from scripts.data.augmentation.tacil.dynamics.checkpoint import load_dynamics_model
from scripts.data.augmentation.tacil.dynamics_model import DynamicsModel
from scripts.shared.data import EpisodeDataset, load_episode_dataset
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass(frozen=True)
class Counts:
    not_converged: int
    rejected: int
    accepted: int


@dataclass(frozen=True)
class AugmentationResult:
    episode_dataset: EpisodeDataset

    bounds: npt.NDArray[np.float32]  # per-step normalized bounds over converged steps
    bound_threshold: float
    counts: Counts
    synthetic_part_length: int  # synthetic transitions per accepted episode


@dataclass(frozen=True)
class InversionOutcomes:
    anchor_alive: torch.Tensor  # (anchor,) bool — True iff all steps converged

    predecessor_states: torch.Tensor  # (anchor, step, state_dim)
    predecessor_actions: torch.Tensor  # (anchor, step, action_dim)
    step_bounds: (
        torch.Tensor
    )  # (anchor, step) float, normalized; valid only for alive anchors


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
            artifact=run.use_artifact(artifact_or_name=config.dataset_artifact),
            logger=logger,
        )

        non_terminal_timestep_indices = _get_non_terminal_timestep_indices(
            episode_end_indices=episode_dataset.episode_end_indices
        )

        action_noise_std = compute_noise_std(
            samples=episode_dataset.actions, noise_scale=config.action_noise_scale
        )  # (action_dimensionality,)

        inversion_outcomes = _run_synthetic_inversions(
            dynamics_model=dynamics_model,
            state_batch=torch.as_tensor(
                episode_dataset.states[non_terminal_timestep_indices]
            ).to(device=device),
            action_batch=torch.as_tensor(
                episode_dataset.actions[non_terminal_timestep_indices]
            ).to(device=device),
            synthetic_part_length=config.synthetic_part_length,
            action_noise_std=torch.as_tensor(action_noise_std).to(device=device),
            max_newton_iteration_count=config.maximum_newton_iterations,
        )

        augmentation_result = _create_augmented_episodes(
            inversion_outcomes=inversion_outcomes,
            episode_dataset=episode_dataset,
            anchor_positions=non_terminal_timestep_indices,
            expert_part_length=config.expert_part_length,
            label_error_bound_quantile=config.label_error_bound_quantile,
        )

        _log_augmentation_to_wandb(
            run=run,
            expert_episode_dataset=episode_dataset,
            augmentation_episode_dataset=augmentation_result.episode_dataset,
        )
        _log_summary_to_wandb(
            run=run,
            augmentation_result=augmentation_result,
            source_episode_count=len(episode_dataset.episode_end_indices),
            source_transition_count=len(episode_dataset.states),
        )


def _get_non_terminal_timestep_indices(
    *,
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
) -> npt.NDArray[np.int64]:  # (non_terminal_timestep_count,)
    mask = np.ones(shape=episode_end_indices[-1], dtype=bool)
    mask[episode_end_indices - 1] = False
    return np.flatnonzero(mask).astype(episode_end_indices.dtype, copy=False)


@torch.no_grad()
def _run_synthetic_inversions(
    *,
    dynamics_model: DynamicsModel,
    state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    action_batch: torch.Tensor,  # (anchor_count, action_dimensionality)
    synthetic_part_length: int,
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
    max_newton_iteration_count: int,
) -> InversionOutcomes:
    device = state_batch.device
    anchor_count, state_dimensionality = state_batch.shape
    action_dimensionality = action_batch.shape[1]

    anchor_state_batch = state_batch.clone()  # (anchor_count, state_dimensionality)

    predecessor_states = torch.empty(
        size=(anchor_count, synthetic_part_length, state_dimensionality), device=device
    )  # (anchor_count, synthetic_part_length, state_dimensionality)
    predecessor_actions = torch.empty(
        size=(anchor_count, synthetic_part_length, action_dimensionality), device=device
    )  # (anchor_count, synthetic_part_length, action_dimensionality)
    step_bounds = torch.empty(
        size=(anchor_count, synthetic_part_length), device=device
    )  # (anchor_count, synthetic_part_length)
    alive_mask = torch.ones(
        anchor_count, dtype=torch.bool, device=device
    )  # (anchor_count,)

    for step in tqdm(range(synthetic_part_length), desc="synthetic step", unit="step"):
        alive_anchor_index_batch = alive_mask.nonzero(as_tuple=False).squeeze(
            1
        )  # (alive_count,)

        alive_state_batch = anchor_state_batch[
            alive_anchor_index_batch
        ]  # (alive_count, state_dimensionality)

        alive_action_batch = action_batch[
            alive_anchor_index_batch
        ]  # (alive_count, action_dimensionality)
        alive_noisy_action_batch = alive_action_batch + torch.normal(
            mean=0.0, std=action_noise_std.expand_as(alive_action_batch)
        )  # (alive_count, action_dimensionality)

        predecessor_state_batch = solve_inverse_dynamics_state(
            dynamics_model=dynamics_model,
            initial_state_estimate_batch=alive_state_batch,
            action_batch=alive_noisy_action_batch,
            target_next_state_batch=alive_state_batch,
            max_newton_iteration_count=max_newton_iteration_count,
        )  # (alive_count, state_dimensionality)

        step_converged_batch = _check_inverse_dynamics_convergence(
            predecessor_state_batch=predecessor_state_batch,
            action_batch=alive_noisy_action_batch,
            state_batch=alive_state_batch,
            dynamics_model=dynamics_model,
        )

        error_bound_batch = _compute_label_error_bound(
            dynamics_model=dynamics_model,
            anchor_state_batch=alive_state_batch,
            action_batch=alive_noisy_action_batch,
            predecessor_state_batch=predecessor_state_batch,
        )  # (alive_count,)

        predecessor_states[alive_anchor_index_batch, step, :] = predecessor_state_batch
        predecessor_actions[alive_anchor_index_batch, step, :] = (
            alive_noisy_action_batch
        )
        step_bounds[alive_anchor_index_batch, step] = error_bound_batch

        alive_mask[alive_anchor_index_batch[~step_converged_batch]] = False
        anchor_state_batch[alive_anchor_index_batch[step_converged_batch]] = (
            predecessor_state_batch[step_converged_batch]
        )

    return InversionOutcomes(
        predecessor_states=predecessor_states,
        predecessor_actions=predecessor_actions,
        anchor_alive=alive_mask,
        step_bounds=step_bounds,
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
    inversion_outcomes: InversionOutcomes,
    episode_dataset: EpisodeDataset,
    anchor_positions: npt.NDArray[np.int64],
    expert_part_length: int,
    label_error_bound_quantile: float,
) -> AugmentationResult:

    converged_step_bounds = inversion_outcomes.step_bounds[
        inversion_outcomes.anchor_alive
    ].flatten()  # (alive_count * step,)
    if converged_step_bounds.numel() == 0:
        raise RuntimeError("No backward steps converged. ")

    bound_threshold = float(
        torch.quantile(input=converged_step_bounds, q=label_error_bound_quantile).item()
    )

    episode_accepted = inversion_outcomes.anchor_alive & (
        inversion_outcomes.step_bounds <= bound_threshold
    ).all(dim=1)  # (anchor_count,)
    episode_converged = inversion_outcomes.anchor_alive  # (anchor_count,)

    expert_part_length_per_anchor = _compute_expert_part_lengths(
        anchor_positions=anchor_positions,
        episode_ends=episode_dataset.episode_end_indices,
        target_expert_part_length=expert_part_length,
    )

    augmented_episode_dataset = _assemble_episodes(
        predecessor_states=inversion_outcomes.predecessor_states.cpu().numpy(),
        predecessor_actions=inversion_outcomes.predecessor_actions.cpu().numpy(),
        episode_accepted=episode_accepted.cpu().numpy(),
        anchor_positions=np.asarray(anchor_positions, dtype=np.int64),
        expert_part_length_per_anchor=expert_part_length_per_anchor,
        states=episode_dataset.states,
        actions=episode_dataset.actions,
    )

    anchor_count = inversion_outcomes.predecessor_actions.shape[0]
    accepted_count = int(episode_accepted.sum().item())
    not_converged_count = int((~episode_converged).sum().item())
    rejected_count = anchor_count - accepted_count - not_converged_count

    return AugmentationResult(
        episode_dataset=augmented_episode_dataset,
        bounds=converged_step_bounds.cpu().numpy(),
        bound_threshold=bound_threshold,
        counts=Counts(
            not_converged=not_converged_count,
            rejected=rejected_count,
            accepted=accepted_count,
        ),
        synthetic_part_length=inversion_outcomes.predecessor_states.shape[1],
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
    )
    episode_end_per_anchor = episode_ends[episode_index_per_anchor]
    available_per_anchor = episode_end_per_anchor - anchor_positions
    return np.minimum(target_expert_part_length, available_per_anchor).astype(np.int64)


def _assemble_episodes(
    *,
    predecessor_states: npt.NDArray[np.float32],  # (anchor, L, state_dim)
    predecessor_actions: npt.NDArray[np.float32],  # (anchor, L, action_dim)
    episode_accepted: npt.NDArray[np.bool_],  # (anchor,)
    anchor_positions: npt.NDArray[np.int64],  # (anchor,)
    expert_part_length_per_anchor: npt.NDArray[np.int64],  # (anchor,)
    states: npt.NDArray[np.float32],  # (expert_timestep_count, state_dim)
    actions: npt.NDArray[np.float32],  # (expert_timestep_count, action_dim)
) -> EpisodeDataset:
    """Stitch accepted anchors into flattened episodes (raw units).

    For accepted anchor at expert position p with synthetic predecessors
    [p_0, ..., p_{L-1}] (p_0 one step back from the anchor, p_{L-1} the
    furthest), the temporal episode is:

        states:  [p_{L-1}, ..., p_0, s_p, s_{p+1}, ..., s_{p+M-1}]
        actions: [a_{L-1}, ..., a_0, u_p, u_{p+1}, ..., u_{p+M-1}]

    i.e. predecessors reversed (oldest first) followed by M expert states,
    with each synthetic action a_t paired with the synthetic state it acts
    on (pusht convention: action[i] is taken at state[i]).
    """
    synthetic_part_length = predecessor_states.shape[1]

    episode_state_chunks: list[np.ndarray] = []
    episode_action_chunks: list[np.ndarray] = []
    episode_lengths: list[int] = []

    for anchor_index in np.nonzero(episode_accepted)[0]:
        anchor_position: int = anchor_positions[anchor_index]
        expert_part_length: int = expert_part_length_per_anchor[anchor_index]

        # Synthetic prefix in temporal order = predecessors reversed.
        synthetic_states = predecessor_states[anchor_index][
            ::-1
        ]  # (synthetic_part_length, state_dim)
        synthetic_actions = predecessor_actions[anchor_index][
            ::-1
        ]  # (synthetic_part_length, action_dim)

        expert_part_slice = slice(anchor_position, anchor_position + expert_part_length)
        expert_part_states = states[
            expert_part_slice
        ]  # (expert_part_length, state_dim)
        expert_part_actions = actions[
            expert_part_slice
        ]  # (expert_part_length, action_dim)

        episode_states = np.concatenate([synthetic_states, expert_part_states], axis=0)
        episode_actions = np.concatenate(
            [synthetic_actions, expert_part_actions], axis=0
        )

        episode_state_chunks.append(episode_states)
        episode_action_chunks.append(episode_actions)
        episode_lengths.append(synthetic_part_length + expert_part_length)

    if not episode_state_chunks:
        state_dim = predecessor_states.shape[2]
        action_dim = predecessor_actions.shape[2]
        return (
            np.zeros((0, state_dim), dtype=np.float32),
            np.zeros((0, action_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    augmented_states = np.concatenate(episode_state_chunks, axis=0).astype(
        np.float32
    )  # (total_timestep_count, state_dim)
    augmented_actions = np.concatenate(episode_action_chunks, axis=0).astype(
        np.float32
    )  # (total_timestep_count, action_dim)
    augmented_episode_ends = np.cumsum(
        np.asarray(episode_lengths, dtype=np.int64)
    )  # (episode_count,)

    return augmented_states, augmented_actions, augmented_episode_ends


if __name__ == "__main__":
    main()
