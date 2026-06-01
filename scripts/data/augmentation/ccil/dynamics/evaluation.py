from dataclasses import dataclass

import numpy as np
import torch
from numpy import typing as npt
from tqdm import tqdm

from scripts.data.augmentation.ccil.dynamics.data import Transitions
from scripts.data.augmentation.ccil.dynamics.inverse_dynamics import (
    CONVERGENCE_TOLERANCE,
    solve_inverse_dynamics_state,
)
from scripts.data.augmentation.ccil.dynamics_model import DynamicsModel


@dataclass(frozen=True)
class RolloutResult:
    reconstruction_error: float
    step_convergence_rate: float


@dataclass(frozen=True)
class EpisodeEvaluation:
    """Per-window outcomes for a single episode."""

    reconstruction_error_batch: (
        torch.Tensor
    )  # (window_count,) normalized start-state error
    step_convergence_rate_batch: (
        torch.Tensor
    )  # (window_count,) fraction of steps that converged


@torch.no_grad()
def evaluate_rollout(
    dynamics_model: DynamicsModel,
    episodes: list[Transitions],
    window_length: int,
    max_newton_iteration_count: int,
) -> RolloutResult:
    """Evaluate the model across every episode and return one dataset-level summary."""
    evaluations = [
        _evaluate_episode(
            dynamics_model=dynamics_model,
            episode=episode,
            window_length=window_length,
            max_newton_iteration_count=max_newton_iteration_count,
        )
        for episode in tqdm(episodes, desc="evaluation", unit="episode")
    ]
    return _aggregate_episode_evaluations(evaluations=evaluations)


@torch.no_grad()
def calculate_local_lipschitz_constants(
    dynamics_model: DynamicsModel,
    states: npt.NDArray[np.float32],  # (sample_count, state_dimensionality)
    actions: npt.NDArray[np.float32],  # (sample_count, action_dimensionality)
) -> np.ndarray:  # (sample_count,)
    """Local Lipschitz constant of the model at each (state, action) sample."""

    batch_size = 1_024

    model_device = next(dynamics_model.parameters()).device
    states_tensor = torch.from_numpy(states).to(device=model_device)
    actions_tensor = torch.from_numpy(actions).to(device=model_device)

    spectral_norm_batches: list[np.ndarray] = []
    for batch_start in tqdm(
        range(0, states_tensor.shape[0], batch_size), desc="lipschitz", unit="batch"
    ):
        batch_slice = slice(batch_start, batch_start + batch_size)
        spectral_norm_batches.append(
            dynamics_model.local_lipschitz_constants(
                state_batch=states_tensor[batch_slice],
                action_batch=actions_tensor[batch_slice],
            )
            .cpu()
            .numpy()
        )

    return np.concatenate(spectral_norm_batches)


def _element_from_each_window(
    sequence: torch.Tensor,  # (sequence_length, feature_dimensionality)
    position_in_window: int,
    window_count: int,
) -> torch.Tensor:  # (window_count, feature_dimensionality)
    """Pick one element at position_in_window per sliding window."""

    return sequence[position_in_window : position_in_window + window_count]


def _take_backward_step(
    dynamics_model: DynamicsModel,
    state_batch: torch.Tensor,  # (window_count, state_dimensionality)
    action_batch: torch.Tensor,  # (window_count, action_dimensionality)
    max_newton_iteration_count: int,
) -> tuple[
    torch.Tensor, torch.Tensor
]:  # (predecessor_state_batch, step_converged_batch)
    predecessor_state_batch = solve_inverse_dynamics_state(
        dynamics_model=dynamics_model,
        initial_state_estimate_batch=state_batch,
        action_batch=action_batch,
        target_next_state_batch=state_batch,
        max_newton_iteration_count=max_newton_iteration_count,
    )

    # Did it converge? Push the recovered predecessor forward and check it lands back on
    # the current state, in normalized space so the tolerance is scale-free.
    reconstructed_current_state_batch = dynamics_model.predict_next_state(
        state_batch=predecessor_state_batch, action_batch=action_batch
    )
    forward_residual_batch: torch.Tensor = (
        dynamics_model.normalize_state(reconstructed_current_state_batch)
        - dynamics_model.normalize_state(state_batch)
    ).norm(dim=-1)  # (window_count,)
    step_converged_batch = forward_residual_batch <= CONVERGENCE_TOLERANCE

    return predecessor_state_batch, step_converged_batch


@torch.no_grad()
def _roll_windows_backward(
    dynamics_model: DynamicsModel,
    state_sequence: torch.Tensor,  # (sequence_length, state_dimensionality)
    action_sequence: torch.Tensor,  # (sequence_length, action_dimensionality)
    window_length: int,
    max_newton_iteration_count: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Reconstruct the start of every sliding window by rolling it backward."""

    window_count = action_sequence.shape[0] - window_length + 1

    # Start at each window's final state i.e. the one point we actually know.
    current_state_batch = _element_from_each_window(
        sequence=state_sequence,
        position_in_window=window_length,
        window_count=window_count,
    )

    # Walk backward through the window
    step_converged_batches: list[torch.Tensor] = []
    for position_in_window in reversed(range(window_length)):
        action_batch = _element_from_each_window(
            sequence=action_sequence,
            position_in_window=position_in_window,
            window_count=window_count,
        )
        current_state_batch, step_converged_batch = _take_backward_step(
            dynamics_model=dynamics_model,
            state_batch=current_state_batch,
            action_batch=action_batch,
            max_newton_iteration_count=max_newton_iteration_count,
        )
        step_converged_batches.append(step_converged_batch)

    reconstructed_initial_state_batch = current_state_batch
    return reconstructed_initial_state_batch, step_converged_batches


@torch.no_grad()
def _evaluate_episode(
    dynamics_model: DynamicsModel,
    episode: Transitions,
    window_length: int,
    max_newton_iteration_count: int,
) -> EpisodeEvaluation:
    model_device = next(dynamics_model.parameters()).device

    # next_states[i] == states[i + 1], so the full trajectory s_0 .. s_T is the
    # per-step states followed by the final next_state.
    state_sequence = torch.from_numpy(
        np.concatenate([episode.states, episode.next_states[-1:]], axis=0)
    ).to(device=model_device)
    action_sequence = torch.from_numpy(episode.actions).to(device=model_device)

    reconstructed_initial_state_batch, step_converged_batches = _roll_windows_backward(
        dynamics_model=dynamics_model,
        state_sequence=state_sequence,
        action_sequence=action_sequence,
        window_length=window_length,
        max_newton_iteration_count=max_newton_iteration_count,
    )

    # Each window's target is its true initial state i.e. position 0 in the window.
    target_initial_state_batch = _element_from_each_window(
        sequence=state_sequence,
        position_in_window=0,
        window_count=reconstructed_initial_state_batch.shape[0],
    )

    reconstruction_error_batch = torch.linalg.norm(
        dynamics_model.normalize_state(reconstructed_initial_state_batch)
        - dynamics_model.normalize_state(target_initial_state_batch),
        dim=-1,
    )  # (window_count,)

    step_convergence_rate_batch = (
        torch.stack(step_converged_batches, dim=1).float().mean(dim=1)
    )  # (window_count,)

    return EpisodeEvaluation(
        reconstruction_error_batch=reconstruction_error_batch,
        step_convergence_rate_batch=step_convergence_rate_batch,
    )


def _aggregate_episode_evaluations(
    evaluations: list[EpisodeEvaluation],
) -> RolloutResult:
    """Pool per-episode results into one dataset-level summary.

    Concatenates the per-window errors and success rates across all episodes and
    averages them, so every window counts equally.
    """
    reconstruction_error_batch = torch.cat(
        [evaluation.reconstruction_error_batch for evaluation in evaluations]
    )
    step_convergence_rate_batch = torch.cat(
        [evaluation.step_convergence_rate_batch for evaluation in evaluations]
    )
    return RolloutResult(
        reconstruction_error=reconstruction_error_batch.mean().item(),
        step_convergence_rate=step_convergence_rate_batch.mean().item(),
    )
