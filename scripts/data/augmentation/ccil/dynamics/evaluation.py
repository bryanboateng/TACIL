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
class BackwardRolloutResult:
    reconstruction_error: float
    newton_convergence_rate: float


type ReconstructionErrors = torch.Tensor
type StepSuccessRates = torch.Tensor

type EpisodeEvaluation = tuple[ReconstructionErrors, StepSuccessRates]


@torch.no_grad()
def backward_rollout_window_batch(
    dynamics_model: DynamicsModel,
    states: torch.Tensor,  # (N, state_dimensionality)
    actions: torch.Tensor,  # (N, action_dimensionality)
    window_length: int,
    max_newton_iteration_count: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:

    window_count = actions.shape[0] - window_length + 1

    final_state_batch = states[
        window_length : window_length + window_count
    ]  # (window_count, state_dimensionality)

    current_state_batch = final_state_batch

    step_converged_list: list[torch.Tensor] = []

    for timestep_index in range(window_length):
        action_index = window_length - timestep_index - 1

        action_batch = actions[
            action_index : action_index + window_count
        ]  # (window_count, action_dimensionality)

        predecessor_state_batch = solve_inverse_dynamics_state(
            dynamics_model=dynamics_model,
            initial_state_estimate_batch=current_state_batch,
            action_batch=action_batch,
            target_next_state_batch=current_state_batch,
            max_newton_iteration_count=max_newton_iteration_count,
        )  # (window_count, state_dimensionality)

        forward_residual_batch: torch.Tensor = (
            dynamics_model.normalize_state(
                dynamics_model.predict_next_state(
                    state_batch=predecessor_state_batch, action_batch=action_batch
                )
            )
            - dynamics_model.normalize_state(current_state_batch)
        ).norm(dim=-1)  # (window_count,)

        step_converged_list.append(forward_residual_batch <= CONVERGENCE_TOLERANCE)

        current_state_batch = predecessor_state_batch

    return current_state_batch, step_converged_list


@torch.no_grad()
def evaluate_episode(
    dynamics_model: DynamicsModel,
    episode: Transitions,
    window_length: int,
    max_newton_iteration_count: int,
) -> EpisodeEvaluation:
    model_device = next(dynamics_model.parameters()).device

    window_count = episode.states.shape[0] - window_length + 1

    states = torch.from_numpy(
        np.concatenate([episode.states, episode.next_states[-1:]], axis=0)
    ).to(device=model_device)
    actions = torch.from_numpy(episode.actions).to(device=model_device)

    reconstructed_initial_state_batch, step_converged_list = (
        backward_rollout_window_batch(
            dynamics_model=dynamics_model,
            states=states,
            actions=actions,
            window_length=window_length,
            max_newton_iteration_count=max_newton_iteration_count,
        )
    )

    target_initial_state_batch = states[:window_count]

    reconstruction_error_batch = torch.linalg.norm(
        dynamics_model.normalize_state(reconstructed_initial_state_batch)
        - dynamics_model.normalize_state(target_initial_state_batch),
        dim=-1,
    )  # (window_count,)

    step_success_rates = (
        torch.stack(step_converged_list, dim=1).float().mean(dim=1)
    )  # (window_count,)

    return reconstruction_error_batch, step_success_rates


def aggregate_episode_evaluations(
    evaluations: list[EpisodeEvaluation],
) -> BackwardRolloutResult:

    reconstruction_errors = torch.cat(
        [reconstruction_errors for reconstruction_errors, _ in evaluations]
    )

    step_success_rates = torch.cat(
        [step_success_rates for _, step_success_rates in evaluations]
    )

    return BackwardRolloutResult(
        reconstruction_error=reconstruction_errors.mean().item(),
        newton_convergence_rate=step_success_rates.mean().item(),
    )


@torch.no_grad()
def evaluate_rollout(
    dynamics_model: DynamicsModel,
    episodes: list[Transitions],
    window_length: int,
    max_newton_iteration_count: int,
) -> BackwardRolloutResult:

    evaluations = [
        evaluate_episode(
            dynamics_model=dynamics_model,
            episode=episode,
            window_length=window_length,
            max_newton_iteration_count=max_newton_iteration_count,
        )
        for episode in tqdm(episodes, desc="evaluation", unit="episode")
    ]

    return aggregate_episode_evaluations(evaluations=evaluations)


@torch.no_grad()
def calculate_local_lipschitz_constants(
    dynamics_model: DynamicsModel,
    states: npt.NDArray[np.float32],  # (sample_count, state_dimensionality)
    actions: npt.NDArray[np.float32],  # (sample_count, action_dimensionality)
    batch_size: int,
) -> np.ndarray:  # (sample_count,)
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
