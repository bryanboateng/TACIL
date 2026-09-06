from dataclasses import dataclass

import numpy as np
import torch
from numpy import typing as npt
from tqdm import tqdm

from scripts.data.augmentation.tacil.create_data.inverse_dynamics import (
    solve_inverse_dynamics,
)
from scripts.data.augmentation.tacil.create_data.state_distance import (
    calculate_state_distances,
)
from scripts.data.augmentation.tacil.create_dynamics.data import (
    TransitionTriples,
    build_windows,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)


@dataclass(frozen=True)
class ErrorMeasurements:
    forward_prediction_errors: torch.Tensor  # (window_count, step_count)
    backward_prediction_errors: torch.Tensor  # (window_count, step_count)
    backward_replay_errors: torch.Tensor  # (window_count, step_count)


def calculate_errors(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    validation_episodes: list[TransitionTriples],
    convergence_tolerance: float,
) -> ErrorMeasurements:
    device = next(dynamics_model_ensemble.parameters()).device
    state_windows, action_windows = _draw_windows(
        validation_episodes=validation_episodes
    )
    state_window_batch = torch.from_numpy(state_windows).to(device=device)
    action_window_batch = torch.from_numpy(action_windows).to(device=device)

    forward_prediction_errors = _calculate_forward(
        dynamics_model_ensemble=dynamics_model_ensemble,
        state_windows=state_window_batch,
        action_windows=action_window_batch,
    )
    backward_prediction_errors, backward_replay_errors = _calculate_backward(
        dynamics_model_ensemble=dynamics_model_ensemble,
        state_windows=state_window_batch,
        action_windows=action_window_batch,
        convergence_tolerance=convergence_tolerance,
    )
    return ErrorMeasurements(
        forward_prediction_errors=forward_prediction_errors,
        backward_prediction_errors=backward_prediction_errors,
        backward_replay_errors=backward_replay_errors,
    )


# ---- Private --------------------------------------------


def _draw_windows(
    *, validation_episodes: list[TransitionTriples]
) -> tuple[
    npt.NDArray[np.float32],  # (window_count, window_length + 1, state_dimensionality)
    npt.NDArray[np.float32],  # (window_count, window_length, action_dimensionality)
]:
    state_windows, action_windows = build_windows(
        episodes=validation_episodes, window_length=20
    )

    maximum_window_count = 1024

    window_count = state_windows.shape[0]
    if window_count <= maximum_window_count:
        return state_windows, action_windows

    window_indices = np.random.default_rng(seed=0).choice(
        window_count, size=maximum_window_count, replace=False
    )
    return state_windows[window_indices], action_windows[window_indices]


@torch.no_grad()
def _calculate_forward(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_windows: torch.Tensor,  # (window_count, step_count + 1, state_dimensionality)
    action_windows: torch.Tensor,  # (window_count, step_count, action_dimensionality)
) -> torch.Tensor:  # (window_count, step_count)
    step_count = action_windows.shape[1]

    current_state_batch = state_windows[:, 0, :]
    distance_columns: list[torch.Tensor] = []

    for step_index in range(step_count):
        current_state_batch = dynamics_model_ensemble.predict_next_state(
            state_batch=current_state_batch,
            action_batch=action_windows[:, step_index, :],
        )
        distance_columns.append(
            calculate_state_distances(
                minuend_batch=current_state_batch,
                subtrahend_batch=state_windows[:, step_index + 1, :],
                dynamics_model_ensemble=dynamics_model_ensemble,
            )
        )

    return torch.stack(tensors=distance_columns, dim=1)


@torch.no_grad()
def _calculate_backward(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_windows: torch.Tensor,  # (window_count, step_count + 1, state_dimensionality)
    action_windows: torch.Tensor,  # (window_count, step_count, action_dimensionality)
    convergence_tolerance: float,
) -> tuple[
    torch.Tensor,  # prediction errors, (window_count, step_count)
    torch.Tensor,  # replay errors, (window_count, step_count)
]:
    window_count, step_count, _ = action_windows.shape
    device = state_windows.device

    prediction_errors = torch.full(
        size=(window_count, step_count), fill_value=float("nan"), device=device
    )
    replay_errors = torch.full_like(input=prediction_errors, fill_value=float("nan"))

    window_indices = torch.arange(window_count, device=device)  # (active_count,)
    current_state_batch = state_windows[:, -1, :]

    for walk_step_index in tqdm(
        iterable=range(step_count), desc="walking windows backward", unit="step"
    ):
        if window_indices.numel() == 0:
            break

        window_step_index = step_count - 1 - walk_step_index

        inverse_dynamics_solution = solve_inverse_dynamics(
            dynamics_model_ensemble=dynamics_model_ensemble,
            initial_state_estimate_batch=current_state_batch,
            action_batch=action_windows[window_indices, window_step_index, :],
            target_next_state_batch=current_state_batch,
            convergence_tolerance=convergence_tolerance,
        )

        converged_mask_batch = (
            inverse_dynamics_solution.converged_batch
        )  # (active_count,)
        window_indices = window_indices[converged_mask_batch]
        current_state_batch = inverse_dynamics_solution.state_batch[
            converged_mask_batch
        ]

        prediction_errors[window_indices, walk_step_index] = calculate_state_distances(
            minuend_batch=current_state_batch,
            subtrahend_batch=state_windows[window_indices, window_step_index, :],
            dynamics_model_ensemble=dynamics_model_ensemble,
        )
        replay_errors[window_indices, walk_step_index] = (
            dynamics_model_ensemble.calculate_replay_error(
                start_state_batch=current_state_batch,
                action_batch_list=[
                    action_windows[window_indices, replayed_step_index, :]
                    for replayed_step_index in range(window_step_index, step_count)
                ],
                end_state_batch=state_windows[window_indices, -1, :],
            )
        )

    return prediction_errors, replay_errors
