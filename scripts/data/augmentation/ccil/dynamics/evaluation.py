from dataclasses import dataclass
from logging import Logger

import numpy as np
import torch
from numpy import typing as npt
from tqdm.auto import tqdm

from scripts.data.augmentation.ccil.dynamics.data import (
    Episode,
    NoValidWindowsError,
    build_windows,
)
from scripts.data.augmentation.ccil.dynamics.inverse_dynamics import (
    solve_inverse_dynamics_state,
)
from scripts.data.augmentation.ccil.dynamics_model import DynamicsModel

# Raw-space residual under which a backward Newton step counts as "converged" for the
# newton_step_success_rate diagnostic. Raw units because it measures solver health
# (does predict_next_state return to the target), not normalized model accuracy.
_STEP_SUCCESS_TOLERANCE: float = 1e-3


@dataclass(frozen=True)
class BackwardRolloutResult:
    error: float
    newton_step_success_rate: float


@torch.no_grad()
def evaluate_forward_rollout(
    dynamics_model: DynamicsModel,
    episodes: list[Episode],
    horizon_lengths: tuple[int, ...],
    logger: Logger,
) -> dict[int, float]:
    model_device = next(dynamics_model.parameters()).device

    results: dict[int, float] = {}
    for horizon_length in tqdm(horizon_lengths, desc="forward rollout", unit="horizon"):
        try:
            (
                state_windows_np,  # (window_count, horizon_length + 1, state_dimensionality)
                action_windows_np,  # (window_count, horizon_length, action_dimensionality)
            ) = build_windows(episodes=episodes, window_length=horizon_length)
        except NoValidWindowsError:
            logger.warning(
                "Skipping rollout horizon %d: no episode is long enough.",
                horizon_length,
            )
            continue

        state_windows = torch.from_numpy(state_windows_np).to(device=model_device)
        action_windows = torch.from_numpy(action_windows_np).to(device=model_device)

        predicted_states = state_windows[:, 0]  # (window_count, state_dimensionality)
        for step in range(horizon_length):
            predicted_states = dynamics_model.predict_next_state(
                predicted_states, action_windows[:, step]
            )  # (window_count, state_dimensionality)

        target_states = state_windows[:, horizon_length]

        # Error is measured in normalized state space, consistent with the training
        # loss, so every dimension contributes on equal footing.
        errors: torch.Tensor = torch.linalg.norm(
            dynamics_model.normalize_state(predicted_states)
            - dynamics_model.normalize_state(target_states),
            dim=-1,
        )  # (window_count,)

        results[horizon_length] = errors.mean().item()

    return results


@torch.no_grad()
def evaluate_backward_rollout(
    dynamics_model: DynamicsModel,
    episodes: list[Episode],
    horizon_lengths: tuple[int, ...],
    max_newton_iteration_count: int,
    action_noise_scale: float,
    description: str,
) -> dict[int, BackwardRolloutResult]:
    model_device = next(dynamics_model.parameters()).device

    reconstruction_errors_per_horizon: dict[int, list[torch.Tensor]] = {
        horizon_length: [] for horizon_length in horizon_lengths
    }
    step_success_rates_per_horizon: dict[int, list[torch.Tensor]] = {
        horizon_length: [] for horizon_length in horizon_lengths
    }

    for episode in tqdm(episodes, desc=description, unit="episode"):
        episode_length: int = episode.states.shape[0]

        all_states = torch.from_numpy(
            np.concatenate([episode.states, episode.next_states[-1:]], axis=0)
        ).to(device=model_device)  # (episode_length + 1, state_dimensionality)

        actions = torch.from_numpy(episode.actions).to(device=model_device)

        for horizon_length in horizon_lengths:
            window_count = episode_length - horizon_length + 1

            if window_count <= 0:
                continue

            current_state_batch = all_states[
                horizon_length : horizon_length + window_count
            ]  # (window_count, state_dimensionality)

            step_converged_list: list[torch.Tensor] = []

            for timestep_index in range(horizon_length):
                action_index = horizon_length - timestep_index - 1

                action_batch = actions[
                    action_index : action_index + window_count
                ]  # (window_count, action_dimensionality)

                action_batch = (
                    action_batch
                    + torch.randn_like(action_batch)
                    * dynamics_model.action_std
                    * action_noise_scale
                )

                predecessor_state_batch = solve_inverse_dynamics_state(
                    dynamics_model=dynamics_model,
                    initial_state_estimate_batch=current_state_batch,
                    action_batch=action_batch,
                    target_next_state_batch=current_state_batch,
                    max_newton_iteration_count=max_newton_iteration_count,
                )  # (window_count, state_dimensionality)

                forward_residual = (
                    dynamics_model.predict_next_state(
                        predecessor_state_batch, action_batch
                    )
                    - current_state_batch
                ).norm(dim=-1)  # (window_count,)

                step_converged_list.append(forward_residual <= 1e-3)

                current_state_batch = predecessor_state_batch

            target_state_batch = all_states[:window_count]

            # Normalized state space, consistent with the forward rollout and loss.
            reconstruction_errors = torch.linalg.norm(
                dynamics_model.normalize_state(current_state_batch)
                - dynamics_model.normalize_state(target_state_batch),
                dim=-1,
            )  # (window_count,)

            reconstruction_errors_per_horizon[horizon_length].append(
                reconstruction_errors
            )

            step_success_rates = (
                torch.stack(step_converged_list, dim=1).float().mean(dim=1)
            )  # (window_count,)

            step_success_rates_per_horizon[horizon_length].append(step_success_rates)

    results: dict[int, BackwardRolloutResult] = {}
    for horizon_length in horizon_lengths:
        per_window_reconstruction_errors = reconstruction_errors_per_horizon[
            horizon_length
        ]
        per_window_step_success_rates = step_success_rates_per_horizon[horizon_length]

        if not per_window_reconstruction_errors:
            continue

        all_reconstruction_errors = torch.cat(per_window_reconstruction_errors)
        all_step_success_rates = torch.cat(per_window_step_success_rates)

        results[horizon_length] = BackwardRolloutResult(
            error=all_reconstruction_errors.mean().item(),
            newton_step_success_rate=all_step_success_rates.mean().item(),
        )

    return results


@torch.no_grad()
def calculate_local_lipschitz_constants(
    dynamics_model: DynamicsModel,
    states: npt.NDArray[np.float32],  # (sample_count, state_dimensionality)
    actions: npt.NDArray[np.float32],  # (sample_count, action_dimensionality)
    batch_size: int,
) -> np.ndarray:  # (sample_count,)
    model_device = next(dynamics_model.parameters()).device

    normalized_states = dynamics_model.normalize_state(
        state_batch=torch.from_numpy(states).to(device=model_device)
    )  # (sample_count, state_dimensionality)
    normalized_actions = dynamics_model.normalize_action(
        action_batch=torch.from_numpy(actions).to(device=model_device)
    )  # (sample_count, action_dimensionality)

    normalized_inputs = torch.cat(
        [normalized_states, normalized_actions], dim=-1
    )  # (sample_count, state_dimensionality + action_dimensionality)

    state_dimensionality = normalized_states.shape[-1]

    def _normalized_residual(normalized_input: torch.Tensor) -> torch.Tensor:
        return dynamics_model.normalized_residual_from_normalized(
            normalized_state_batch=normalized_input[:state_dimensionality],
            normalized_action_batch=normalized_input[state_dimensionality:],
        )

    batched_jacobian = torch.vmap(torch.func.jacrev(_normalized_residual))

    spectral_norm_batches: list[np.ndarray] = []
    for batch_start in tqdm(
        range(0, normalized_inputs.shape[0], batch_size), desc="lipschitz", unit="batch"
    ):
        jacobian_batch = batched_jacobian(
            normalized_inputs[batch_start : batch_start + batch_size]
        )  # (batch_size, state_dimensionality, state_dimensionality + action_dimensionality)

        spectral_norms: torch.Tensor = torch.linalg.matrix_norm(
            jacobian_batch, ord=2
        )  # (batch_size,)
        spectral_norm_batches.append(spectral_norms.cpu().numpy())

    return np.concatenate(spectral_norm_batches)  # (sample_count,)
