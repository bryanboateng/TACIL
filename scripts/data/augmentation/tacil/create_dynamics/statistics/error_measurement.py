from dataclasses import dataclass

import numpy as np
import torch
from numpy import typing as npt
from tqdm import tqdm

from scripts.data.augmentation.tacil.create_data.inverse_dynamics import (
    solve_inverse_dynamics,
)
from scripts.data.augmentation.tacil.create_dynamics.data import (
    TransitionSet,
    build_windows,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.pusht_state import subtract_states


@dataclass(frozen=True)
class ErrorMeasurements:
    forward_prediction_errors: torch.Tensor  # (window_count, step_count)
    backward_prediction_errors: torch.Tensor  # (window_count, step_count)
    backward_replay_errors: torch.Tensor  # (window_count, step_count)


def measure_errors(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    validation_episodes: list[TransitionSet],
    convergence_tolerance: float,
) -> ErrorMeasurements:
    device = next(dynamics_model_ensemble.parameters()).device
    state_windows, action_windows = _draw_windows(
        validation_episodes=validation_episodes
    )
    state_window_batch = torch.from_numpy(state_windows).to(device=device)
    action_window_batch = torch.from_numpy(action_windows).to(device=device)

    forward_prediction_errors = _measure_forward(
        dynamics_model_ensemble=dynamics_model_ensemble,
        state_windows=state_window_batch,
        action_windows=action_window_batch,
    )
    backward_prediction_errors, backward_replay_errors = _measure_backward(
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
    *, validation_episodes: list[TransitionSet]
) -> tuple[
    npt.NDArray[
        np.float32
    ],  # (window_count, _WINDOW_STEP_COUNT + 1, state_dimensionality)
    npt.NDArray[
        np.float32
    ],  # (window_count, _WINDOW_STEP_COUNT, action_dimensionality)
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
def _measure_forward(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_windows: torch.Tensor,  # (window_count, step_count + 1, state_dimensionality)
    action_windows: torch.Tensor,  # (window_count, step_count, action_dimensionality)
) -> torch.Tensor:  # (window_count, step_count)
    step_count = action_windows.shape[1]

    current_state_batch = state_windows[:, 0, :]
    difference_columns: list[torch.Tensor] = []

    for step_index in range(step_count):
        action_batch = action_windows[:, step_index, :]
        current_state_batch = dynamics_model_ensemble.predict_next_state(
            state_batch=current_state_batch, action_batch=action_batch
        )
        true_next_state_batch = state_windows[:, step_index + 1, :]
        difference_columns.append(
            dynamics_model_ensemble.state_normalizer.normalize_difference(
                difference_batch=subtract_states(
                    minuend_batch=current_state_batch,
                    subtrahend_batch=true_next_state_batch,
                )
            )
        )

    return torch.linalg.norm(torch.stack(tensors=difference_columns, dim=1), dim=-1)


@torch.no_grad()
def _measure_backward(
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

    current_state_batch = state_windows[:, -1, :].clone()
    active_mask_batch = torch.ones(window_count, dtype=torch.bool, device=device)

    prediction_errors = torch.full(
        size=(window_count, step_count), fill_value=float("nan"), device=device
    )
    replay_errors = torch.full_like(input=prediction_errors, fill_value=float("nan"))

    for step_index in tqdm(
        iterable=range(step_count), desc="walking windows backward", unit="step"
    ):
        active_indices = active_mask_batch.nonzero().squeeze(dim=1)
        if active_indices.numel() == 0:
            break

        window_index = step_count - 1 - step_index
        action_batch = action_windows[active_indices, window_index, :]
        successor_state_batch = current_state_batch[active_indices]

        inverse_dynamics_solution = solve_inverse_dynamics(
            dynamics_model_ensemble=dynamics_model_ensemble,
            initial_state_estimate_batch=successor_state_batch,
            action_batch=action_batch,
            target_next_state_batch=successor_state_batch,
            convergence_tolerance=convergence_tolerance,
        )

        converged_mask_batch = inverse_dynamics_solution.converged_batch
        active_mask_batch[active_indices[~converged_mask_batch]] = False
        converged_indices = active_indices[converged_mask_batch]
        if converged_indices.numel() == 0:
            break

        predecessor_state_batch = inverse_dynamics_solution.state_batch[
            converged_mask_batch
        ]

        batch = state_windows[converged_indices, window_index, :]
        prediction_errors[converged_indices, step_index] = torch.linalg.norm(
            dynamics_model_ensemble.state_normalizer.normalize_difference(
                difference_batch=subtract_states(
                    minuend_batch=predecessor_state_batch, subtrahend_batch=batch
                )
            ),
            dim=-1,
        )
        replay_errors[converged_indices, step_index] = (
            dynamics_model_ensemble.replay_error(
                start_state_batch=predecessor_state_batch,
                action_batch_list=[
                    action_windows[converged_indices, replayed_window_index, :]
                    for replayed_window_index in range(window_index, step_count)
                ],
                end_state_batch=state_windows[converged_indices, -1, :],
            )
        )

        current_state_batch[converged_indices] = predecessor_state_batch

    return prediction_errors, replay_errors
