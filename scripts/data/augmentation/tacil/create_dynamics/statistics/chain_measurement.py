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

# Fixed measurement shape: long enough to show the growth trend, short enough
# to stay affordable, and heavily overlapping windows make more chains
# redundant anyway.
_CHAIN_STEP_COUNT = 20
_CHAIN_COUNT = 1024
_SUBSAMPLE_SEED = 0


@dataclass(frozen=True)
class ForwardChainMeasurements:
    """Open-loop rollouts of the ensemble mean, one chain per window.

    Entry [c, s] belongs to chain c after s + 1 steps: the model has taken s + 1
    steps on its own predictions. All values are in normalized state units.
    """

    # Distance between the model's prediction and the real state.
    prediction_errors: torch.Tensor  # (chain_count, step_count)


@dataclass(frozen=True)
class BackwardChainMeasurements:
    """Backward walks from the windows' final states, one chain per window.

    Entry [c, s] belongs to chain c after s + 1 steps: the Newton solver has
    stepped s + 1 states back from the anchor. A chain ends at its first
    non-converged solve; entries past that point stay NaN.
    """

    # Distance between the solved predecessor and the real predecessor.
    prediction_errors: torch.Tensor  # (chain_count, step_count)
    # Every member replays the chain's actions from the solved predecessor, and
    # this is how far their endpoints land from the state the chain started at.
    # The limit is read against this value.
    #
    # It falls to zero when the members collapse onto one function, because the
    # solver fits the mean and identical members all replay onto the anchor.
    # That is the run-is-worthless alarm.
    replay_errors: torch.Tensor  # (chain_count, step_count)


def draw_chain_windows(
    *, validation_episodes: list[TransitionSet]
) -> tuple[
    npt.NDArray[
        np.float32
    ],  # (chain_count, _CHAIN_STEP_COUNT + 1, state_dimensionality)
    npt.NDArray[np.float32],  # (chain_count, _CHAIN_STEP_COUNT, action_dimensionality)
]:
    return _subsample(
        *build_windows(episodes=validation_episodes, window_length=_CHAIN_STEP_COUNT)
    )


@torch.no_grad()
def measure_forward_chains(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_windows: torch.Tensor,  # (chain_count, step_count + 1, state_dimensionality)
    action_windows: torch.Tensor,  # (chain_count, step_count, action_dimensionality)
) -> ForwardChainMeasurements:
    step_count = action_windows.shape[1]

    current_state_batch = state_windows[:, 0, :]
    difference_columns: list[torch.Tensor] = []

    for step_index in range(step_count):
        action_batch = action_windows[:, step_index, :]
        current_state_batch = dynamics_model_ensemble.predict_next_state(
            state_batch=current_state_batch, action_batch=action_batch
        )
        batch = state_windows[:, step_index + 1, :]
        difference_columns.append(
            dynamics_model_ensemble.state_normalizer.normalize_difference(
                difference_batch=subtract_states(
                    minuend_batch=current_state_batch, subtrahend_batch=batch
                )
            )
        )

    return ForwardChainMeasurements(
        prediction_errors=torch.linalg.norm(
            torch.stack(tensors=difference_columns, dim=1), dim=-1
        )
    )


@torch.no_grad()
def measure_backward_chains(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_windows: torch.Tensor,  # (chain_count, step_count + 1, state_dimensionality)
    action_windows: torch.Tensor,  # (chain_count, step_count, action_dimensionality)
    convergence_tolerance: float,
) -> BackwardChainMeasurements:
    chain_count, step_count, _ = action_windows.shape
    device = state_windows.device

    current_state_batch = state_windows[:, -1, :].clone()
    active_mask_batch = torch.ones(chain_count, dtype=torch.bool, device=device)

    prediction_errors = torch.full(
        size=(chain_count, step_count), fill_value=float("nan"), device=device
    )
    replay_errors = torch.full_like(input=prediction_errors, fill_value=float("nan"))

    for step_index in tqdm(
        iterable=range(step_count), desc="backward diagnostic chains", unit="step"
    ):
        active_indices = active_mask_batch.nonzero().squeeze(dim=1)
        if active_indices.numel() == 0:
            break

        # The action that led into the chain's current state, and therefore
        # out of the predecessor the solver is asked for.
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
                chain_action_batches=[
                    action_windows[converged_indices, replayed_window_index, :]
                    for replayed_window_index in range(window_index, step_count)
                ],
                end_state_batch=state_windows[converged_indices, -1, :],
            )
        )

        current_state_batch[converged_indices] = predecessor_state_batch

    return BackwardChainMeasurements(
        prediction_errors=prediction_errors, replay_errors=replay_errors
    )


# ---- Private --------------------------------------------


def _subsample(
    state_windows: npt.NDArray[np.float32],  # (window_count, length + 1, state_dim)
    action_windows: npt.NDArray[np.float32],  # (window_count, length, action_dim)
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Cap the number of windows. They overlap heavily (stride 1), so a random
    subset loses no coverage, only redundancy."""
    window_count = state_windows.shape[0]
    if window_count <= _CHAIN_COUNT:
        return state_windows, action_windows

    window_indices = np.random.default_rng(seed=_SUBSAMPLE_SEED).choice(
        window_count, size=_CHAIN_COUNT, replace=False
    )
    return state_windows[window_indices], action_windows[window_indices]
