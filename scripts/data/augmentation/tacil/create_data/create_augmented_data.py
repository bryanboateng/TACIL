import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import tyro
import wandb
import zarr
from rich.pretty import pretty_repr
from scipy.stats import gaussian_kde
from tqdm import tqdm
from wandb import Run

from scripts.ashared import create_logger, log_device_info, set_seed
from scripts.data.augmentation.create_noise_augmentation import compute_noise_std
from scripts.data.augmentation.tacil.create_data.config import Config
from scripts.data.augmentation.tacil.create_data.inverse_dynamics import (
    CONVERGENCE_TOLERANCE,
    solve_inverse_dynamics_state,
)
from scripts.data.augmentation.tacil.dynamics.checkpoint import load_dynamics_checkpoint
from scripts.data.augmentation.tacil.dynamics_model import DynamicsModel
from scripts.shared.data import load_demonstration_data_from_artifact
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


@dataclass(frozen=True)
class AugmentationResult:
    """Output of an augmentation run, in flattened-episode form (raw units)."""

    augmented_states: np.ndarray  # (augmented_timestep_count, state_dimensionality)
    augmented_actions: np.ndarray  # (augmented_timestep_count, action_dimensionality)
    augmented_episode_ends: np.ndarray  # exclusive ends into augmented_states
    bounds: np.ndarray  # per-step normalized bounds over converged steps
    sigma: float
    num_accepted: int
    num_rejected: int
    num_not_converged: int


@dataclass(frozen=True)
class InversionOutcomes:
    """Dense per-(anchor, step) outcomes. step indexes backward from the anchor."""

    predecessor_states: torch.Tensor  # (anchor, step, state_dim)
    predecessor_actions: torch.Tensor  # (anchor, step, action_dim)
    step_converged: torch.Tensor  # (anchor, step) bool
    step_bounds: torch.Tensor  # (anchor, step) float, normalized


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(config: Config) -> None:
    set_seed(seed=config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    with wandb.init(
        project="abc",
        job_type="create-tacil-dataset",
        config=asdict(config),
        dir=OUTPUT_DIRECTORY_PATH,
    ) as run:
        dynamics_model = load_dynamics_checkpoint(
            path=Path(
                run.use_artifact(artifact_or_name=config.model_artifact).download()
            )
            / "dynamics.pt",
            device=device,
        )
        dynamics_model.eval()

        demonstration_data = load_demonstration_data_from_artifact(
            artifact=run.use_artifact(artifact_or_name=config.dataset_artifact),
            logger=logger,
        )

        non_terminal_timestep_indices = _get_non_terminal_timestep_indices(
            episode_end_indices=demonstration_data.episode_end_indices
        )

        action_noise_std = compute_noise_std(
            samples=demonstration_data.actions, noise_scale=config.action_noise_scale
        )  # (action_dimensionality,)

        inversion_outcomes = _run_synthetic_inversions(
            dynamics_model=dynamics_model,
            state_batch=torch.as_tensor(
                demonstration_data.states[non_terminal_timestep_indices]
            ).to(device=device),
            action_batch=torch.as_tensor(
                demonstration_data.actions[non_terminal_timestep_indices]
            ).to(device=device),
            trajectory_length=config.synthetic_length,
            action_noise_std=torch.as_tensor(action_noise_std).to(device=device),
            max_newton_iteration_count=config.maximum_newton_iterations,
        )

        result = create_augmented_episodes(
            inversion_outcomes=inversion_outcomes,
            states=np.asarray(demonstration_data.states),
            actions=np.asarray(demonstration_data.actions),
            episode_ends=np.asarray(demonstration_data.episode_end_indices),
            anchor_positions=non_terminal_timestep_indices,
            expert_tail_length=config.expert_length,
            label_error_bound_quantile=config.label_error_bound_quantile,
        )

        _log_augmentation_to_wandb(
            run=run,
            config=config,
            result=result,
            states=demonstration_data.states,
            actions=demonstration_data.actions,
            episode_ends=demonstration_data.episode_end_indices,
        )


def _log_augmentation_to_wandb(
    run: Run,
    config: Config,
    result: AugmentationResult,
    states: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_directory_path = Path(temporary_directory)
        combined_zarr_path = temporary_directory_path / "combined.zarr"

        write_combined_zarr(
            output_path=combined_zarr_path,
            states=np.asarray(states),
            actions=np.asarray(actions),
            episode_ends=np.asarray(episode_ends),
            augmented_states=result.augmented_states,
            augmented_actions=result.augmented_actions,
            augmented_episode_ends_relative=result.augmented_episode_ends,
        )

        _log_summary_to_wandb(
            run=run, result=result, quantile=config.label_error_bound_quantile
        )

        dataset_artifact = wandb.Artifact(
            name=f"tacil-dataset-{run.name}",
            type="dataset",
        )
        dataset_artifact.add_dir(str(combined_zarr_path))
        run.log_artifact(dataset_artifact).wait()


@torch.no_grad()
def create_augmented_episodes(
    inversion_outcomes: InversionOutcomes,
    states: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
    anchor_positions: np.ndarray,
    expert_tail_length: int,
    label_error_bound_quantile: float,
) -> AugmentationResult:

    converged_step_bounds = inversion_outcomes.step_bounds[
        inversion_outcomes.step_converged
    ]
    if converged_step_bounds.numel() == 0:
        raise RuntimeError("No backward steps converged. ")

    sigma = float(
        torch.quantile(input=converged_step_bounds, q=label_error_bound_quantile).item()
    )

    step_accepted = inversion_outcomes.step_converged & (
        inversion_outcomes.step_bounds <= sigma
    )
    episode_accepted = step_accepted.all(dim=1)  # (anchor_count,)
    episode_converged = inversion_outcomes.step_converged.all(dim=1)

    tail_length_per_anchor = _compute_tail_lengths(
        anchor_positions=anchor_positions,
        episode_ends=episode_ends,
        target_tail_length=expert_tail_length,
    )

    augmented_states, augmented_actions, augmented_episode_ends = _assemble_episodes(
        predecessor_states=inversion_outcomes.predecessor_states.cpu().numpy(),
        predecessor_actions=inversion_outcomes.predecessor_actions.cpu().numpy(),
        episode_accepted=episode_accepted.cpu().numpy(),
        anchor_positions=np.asarray(anchor_positions, dtype=np.int64),
        tail_length_per_anchor=tail_length_per_anchor,
        states=states,
        actions=actions,
    )

    anchor_count = inversion_outcomes.predecessor_actions.shape[0]
    accepted_count = int(episode_accepted.sum().item())
    not_converged_count = int((~episode_converged).sum().item())
    rejected_count = anchor_count - accepted_count - not_converged_count

    logger.info(
        "Anchors: %d | accepted: %d | rejected (converged, bound>sigma): %d | "
        "not converged: %d | sigma=%.4g (q=%.2f)",
        anchor_count,
        accepted_count,
        rejected_count,
        not_converged_count,
        sigma,
        label_error_bound_quantile,
    )

    return AugmentationResult(
        augmented_states=augmented_states,
        augmented_actions=augmented_actions,
        augmented_episode_ends=augmented_episode_ends,
        bounds=converged_step_bounds.cpu().numpy(),
        sigma=sigma,
        num_accepted=accepted_count,
        num_rejected=rejected_count,
        num_not_converged=not_converged_count,
    )


@torch.no_grad()
def _run_synthetic_inversions(
    dynamics_model: DynamicsModel,
    state_batch: torch.Tensor,  # (anchor_count, state_dimensionality)
    action_batch: torch.Tensor,  # (anchor_count, action_dimensionality)
    trajectory_length: int,
    action_noise_std: torch.Tensor,  # (action_dimensionality,)
    max_newton_iteration_count: int,
) -> InversionOutcomes:
    device = state_batch.device
    anchor_count, state_dimensionality = state_batch.shape
    action_dimensionality = action_batch.shape[1]

    anchor_state_batch = state_batch.clone()  # (anchor_count, state_dimensionality)

    predecessor_states = torch.empty(
        size=(anchor_count, trajectory_length, state_dimensionality), device=device
    )
    predecessor_actions = torch.empty(
        size=(anchor_count, trajectory_length, action_dimensionality), device=device
    )
    step_converged = torch.empty(
        size=(anchor_count, trajectory_length), dtype=torch.bool, device=device
    )
    step_bounds = torch.empty(size=(anchor_count, trajectory_length), device=device)

    for step in tqdm(range(trajectory_length), desc="synthetic step", unit="step"):
        noise = torch.normal(mean=0.0, std=action_noise_std.expand_as(action_batch))
        noisy_action_batch = action_batch + noise

        predecessor_state_batch = solve_inverse_dynamics_state(
            dynamics_model=dynamics_model,
            initial_state_estimate_batch=anchor_state_batch,
            action_batch=noisy_action_batch,
            target_next_state_batch=anchor_state_batch,
            max_newton_iteration_count=max_newton_iteration_count,
        )  # (anchor_count, state_dimensionality)

        reconstruction_error_batch: torch.Tensor = (
            dynamics_model.normalize_state(
                state_batch=dynamics_model.predict_next_state(
                    state_batch=predecessor_state_batch, action_batch=noisy_action_batch
                )
            )
            - dynamics_model.normalize_state(anchor_state_batch)
        ).norm(dim=-1)  # (anchor_count,)

        step_has_converged_batch = reconstruction_error_batch <= CONVERGENCE_TOLERANCE

        bound_batch = _compute_label_error_bound(
            dynamics_model=dynamics_model,
            anchor_state_batch=anchor_state_batch,
            action_batch=noisy_action_batch,
            predecessor_state_batch=predecessor_state_batch,
        )  # (anchor_count,)

        predecessor_states[:, step, :] = predecessor_state_batch
        predecessor_actions[:, step, :] = noisy_action_batch
        step_converged[:, step] = step_has_converged_batch
        step_bounds[:, step] = bound_batch

        anchor_state_batch = predecessor_state_batch

    return InversionOutcomes(
        predecessor_states=predecessor_states,
        predecessor_actions=predecessor_actions,
        step_converged=step_converged,
        step_bounds=step_bounds,
    )


def _assemble_episodes(
    predecessor_states: np.ndarray,  # (anchor, L, state_dim)
    predecessor_actions: np.ndarray,  # (anchor, L, action_dim)
    episode_accepted: np.ndarray,  # (anchor,) bool
    anchor_positions: np.ndarray,  # (anchor,) int
    tail_length_per_anchor: np.ndarray,  # (anchor,) int
    states: np.ndarray,  # (expert_timestep_count, state_dim)
    actions: np.ndarray,  # (expert_timestep_count, action_dim)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    synthetic_length = predecessor_states.shape[1]

    episode_state_chunks: list[np.ndarray] = []
    episode_action_chunks: list[np.ndarray] = []
    episode_lengths: list[int] = []

    for anchor_index in np.nonzero(episode_accepted)[0]:
        anchor_position = int(anchor_positions[anchor_index])
        tail_length = int(tail_length_per_anchor[anchor_index])

        # Synthetic prefix in temporal order = predecessors reversed.
        synthetic_states = predecessor_states[anchor_index][::-1]  # (L, state_dim)
        synthetic_actions = predecessor_actions[anchor_index][::-1]  # (L, action_dim)

        tail_slice = slice(anchor_position, anchor_position + tail_length)
        tail_states = states[tail_slice]  # (M, state_dim)
        tail_actions = actions[tail_slice]  # (M, action_dim)

        episode_states = np.concatenate([synthetic_states, tail_states], axis=0)
        episode_actions = np.concatenate([synthetic_actions, tail_actions], axis=0)

        episode_state_chunks.append(episode_states)
        episode_action_chunks.append(episode_actions)
        episode_lengths.append(synthetic_length + tail_length)

    if not episode_state_chunks:
        state_dim = predecessor_states.shape[2]
        action_dim = predecessor_actions.shape[2]
        return (
            np.zeros((0, state_dim), dtype=np.float32),
            np.zeros((0, action_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    augmented_states = np.concatenate(episode_state_chunks, axis=0).astype(np.float32)
    augmented_actions = np.concatenate(episode_action_chunks, axis=0).astype(np.float32)
    augmented_episode_ends = np.cumsum(np.asarray(episode_lengths, dtype=np.int64))

    return augmented_states, augmented_actions, augmented_episode_ends


def _get_non_terminal_timestep_indices(
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
) -> npt.NDArray[np.int64]:

    mask = np.ones(shape=episode_end_indices[-1], dtype=bool)
    mask[episode_end_indices - 1] = False
    return np.flatnonzero(mask).astype(episode_end_indices.dtype, copy=False)


def _compute_tail_lengths(
    anchor_positions: np.ndarray, episode_ends: np.ndarray, target_tail_length: int
) -> np.ndarray:
    anchor_positions = np.asarray(anchor_positions, dtype=np.int64)
    episode_ends = np.asarray(episode_ends, dtype=np.int64)

    episode_index_per_anchor = np.searchsorted(
        episode_ends, anchor_positions, side="right"
    )
    episode_end_per_anchor = episode_ends[episode_index_per_anchor]
    available_tail_per_anchor = episode_end_per_anchor - anchor_positions
    return np.minimum(target_tail_length, available_tail_per_anchor).astype(np.int64)


def _validate_arguments(
    synthetic_length: int, expert_tail_length: int, label_error_bound_quantile: float
) -> None:
    if synthetic_length < 1:
        raise ValueError(f"synthetic_length must be >= 1, got {synthetic_length}")
    if expert_tail_length < 1:
        raise ValueError(f"expert_tail_length must be >= 1, got {expert_tail_length}")
    if not 0.0 < label_error_bound_quantile <= 1.0:
        raise ValueError(
            f"label_error_bound_quantile must be in (0, 1], got "
            f"{label_error_bound_quantile}"
        )


def write_combined_zarr(
    output_path: Path,
    states: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
    augmented_states: np.ndarray,
    augmented_actions: np.ndarray,
    augmented_episode_ends_relative: np.ndarray,
) -> None:
    """Write expert + augmented episodes as a single zarr in pusht layout.

    `augmented_episode_ends_relative` are exclusive end indices INTO
    `augmented_states`; they're shifted by the expert state count to become
    absolute indices in the combined dataset.
    """
    states = states.astype(np.float32)
    actions = actions.astype(np.float32)
    augmented_states = augmented_states.astype(np.float32)
    augmented_actions = augmented_actions.astype(np.float32)

    all_states = np.concatenate([states, augmented_states], axis=0)
    all_actions = np.concatenate([actions, augmented_actions], axis=0)

    expert_total = states.shape[0]
    augmented_ends_absolute = expert_total + augmented_episode_ends_relative.astype(
        np.int64
    )
    all_ends = np.concatenate(
        [episode_ends.astype(np.int64), augmented_ends_absolute], axis=0
    )

    root = zarr.open_group(str(output_path), mode="w", zarr_format=2)
    data = root.create_group("data")
    data.create_array("state", data=all_states)
    data.create_array("action", data=all_actions)
    meta = root.create_group("meta")
    meta.create_array("episode_ends", data=all_ends)

    logger.info(
        "Wrote %d expert + %d augmented = %d states across %d episodes "
        "(%d augmented) to %s",
        expert_total,
        augmented_states.shape[0],
        all_states.shape[0],
        len(all_ends),
        len(augmented_ends_absolute),
        output_path,
    )


def _log_summary_to_wandb(
    run: Run, result: AugmentationResult, quantile: float
) -> None:
    run.summary["aug/num_accepted_episodes"] = result.num_accepted
    run.summary["aug/num_rejected"] = result.num_rejected
    run.summary["aug/num_not_converged"] = result.num_not_converged
    run.summary["aug/num_augmented_states"] = int(result.augmented_states.shape[0])
    run.summary["aug/sigma"] = result.sigma
    run.summary["aug/quantile"] = float(quantile)

    # Outcomes bar chart
    outcomes_table = wandb.Table(
        columns=["outcome", "count"],
        data=[
            ["Accepted", result.num_accepted],
            ["Rejected", result.num_rejected],
            ["Not converged", result.num_not_converged],
        ],
    )
    log_payload: dict = {
        "figures/outcomes": wandb.plot.bar(
            outcomes_table,
            label="outcome",
            value="count",
            title="Augmentation outcomes",
        )
    }

    # Bounds density line chart
    bounds = result.bounds
    if bounds.size > 0:
        max_x = float(np.percentile(bounds, q=99.5))
        x_values = np.linspace(0.0, max_x, num=500)
        y_values = gaussian_kde(bounds)(x_values)

        bounds_table = wandb.Table(
            columns=["bound", "density"],
            data=[[float(x), float(y)] for x, y in zip(x_values, y_values)],
        )
        log_payload["figures/bounds"] = wandb.plot.line(
            bounds_table,
            x="bound",
            y="density",
            title=f"Per-step bound distribution — σ={result.sigma:.4f} (q={quantile})",
        )

    run.log(log_payload)


@torch.no_grad()
def _compute_label_error_bound(
    dynamics_model: DynamicsModel,
    anchor_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    predecessor_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
) -> torch.Tensor:  # (batch_size,)
    local_lipschitz_constant_batch = dynamics_model.local_lipschitz_constants(
        state_batch=anchor_state_batch, action_batch=action_batch
    )

    normalized_step_distance_batch: torch.Tensor = torch.linalg.norm(
        dynamics_model.normalize_state(state_batch=predecessor_state_batch)
        - dynamics_model.normalize_state(state_batch=anchor_state_batch),
        dim=-1,
    )

    return local_lipschitz_constant_batch * normalized_step_distance_batch


if __name__ == "__main__":
    main()
