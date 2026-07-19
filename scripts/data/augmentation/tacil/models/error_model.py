from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from scripts.data.augmentation.tacil.models.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.data import EpisodeStore

_SIGNAL_CHUNK_SIZE = 4096


@torch.no_grad()
def calculate_median_step_distance(
    *, episode_store: EpisodeStore, dynamics_model_ensemble: DynamicsModelEnsemble
) -> float:
    """Typical per-step motion of the expert, in normalized state units."""

    device = next(dynamics_model_ensemble.parameters()).device

    normalized_state_batch = dynamics_model_ensemble.normalize_state(
        state_batch=torch.as_tensor(data=episode_store.states).to(device=device)
    )  # (total_timestep_count, state_dimensionality)

    # For every consecutive pair of states (s_t, s_{t+1}), compute how far
    # apart they are in space.
    # This is "how much did the state change in one step" for every step across every episode.
    #
    # The shape is (total_timestep_count - 1,).
    # I.e. one distance per consecutive pair of states.
    step_distance_batch = torch.linalg.norm(
        normalized_state_batch[1:] - normalized_state_batch[:-1], dim=1
    )  # (total_timestep_count - 1,)

    # Because all episodes inside the episode store are concatenated into
    # one long tensor, some "consecutive" pairs are not actually consecutive
    # in time.
    # They are the last state of one episode paired with the first state of
    # the next episode.
    # We need to exclude those pairs.
    episode_boundary_mask = np.zeros(shape=len(step_distance_batch), dtype=bool)
    episode_boundary_mask[episode_store.episode_end_indices[:-1] - 1] = True

    return float(
        step_distance_batch[
            ~torch.as_tensor(data=episode_boundary_mask).to(device=device)
        ]
        .median()
        .item()
    )


@dataclass(frozen=True)
class ErrorModelSignals:
    disagreement_batch: torch.Tensor  # (transition_count,)
    sensitivity_scaled_step_distance_batch: torch.Tensor  # (transition_count,)
    true_error_batch: torch.Tensor  # (transition_count,)


@torch.no_grad()
def compute_error_model_signals(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_batch: torch.Tensor,  # (transition_count, state_dimensionality)
    action_batch: torch.Tensor,  # (transition_count, action_dimensionality)
    next_state_batch: torch.Tensor,  # (transition_count, state_dimensionality)
) -> ErrorModelSignals:
    disagreement_chunks: list[torch.Tensor] = []
    sensitivity_scaled_step_distance_chunks: list[torch.Tensor] = []
    true_error_chunks: list[torch.Tensor] = []

    for chunk_start in range(0, state_batch.shape[0], _SIGNAL_CHUNK_SIZE):
        chunk = slice(chunk_start, chunk_start + _SIGNAL_CHUNK_SIZE)

        disagreement_chunks.append(
            dynamics_model_ensemble.disagreements_about_next_state(
                state_batch=state_batch[chunk], action_batch=action_batch[chunk]
            )
        )
        sensitivity_scaled_step_distance_chunks.append(
            dynamics_model_ensemble.sensitivity_scaled_step_distances(
                state_batch=state_batch[chunk],
                next_state_batch=next_state_batch[chunk],
                action_batch=action_batch[chunk],
            )
        )
        true_error_chunks.append(
            torch.linalg.norm(
                dynamics_model_ensemble.normalize_state(
                    state_batch=dynamics_model_ensemble.predict_next_state(
                        state_batch=state_batch[chunk], action_batch=action_batch[chunk]
                    )
                )
                - dynamics_model_ensemble.normalize_state(
                    state_batch=next_state_batch[chunk]
                ),
                dim=-1,
            )
        )

    return ErrorModelSignals(
        disagreement_batch=torch.cat(tensors=disagreement_chunks),
        sensitivity_scaled_step_distance_batch=torch.cat(
            tensors=sensitivity_scaled_step_distance_chunks
        ),
        true_error_batch=torch.cat(tensors=true_error_chunks),
    )


@dataclass(frozen=True)
class ErrorModel:
    disagreement_weight: float
    sensitivity_scaled_step_distance_weight: float
    conformal_scale: float
    quantile: float
    uncentered_r_squared: float
    fit_transition_count: int
    calibration_transition_count: int

    def estimate_step_errors(
        self,
        *,
        disagreement_batch: torch.Tensor,  # (batch_size,)
        sensitivity_scaled_step_distance_batch: torch.Tensor,  # (batch_size,)
    ) -> torch.Tensor:  # (batch_size,)
        return self.conformal_scale * (
            self.disagreement_weight * disagreement_batch
            + self.sensitivity_scaled_step_distance_weight
            * sensitivity_scaled_step_distance_batch
        )


def fit_error_model(
    *, error_model_signals: ErrorModelSignals, quantile: float, split_seed: int
) -> ErrorModel:
    """Fit the weights on one half, conformalize the scale on the other.

    The conformal quantile is only honest on data the weights were not fitted
    to, hence the disjoint halves. On exchangeable in-distribution transitions
    the result covers the true error with probability at least `quantile`.
    """
    disagreements = (
        error_model_signals.disagreement_batch.detach().cpu().numpy().astype(np.float64)
    )
    sensitivity_scaled_step_distances = (
        error_model_signals.sensitivity_scaled_step_distance_batch.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    true_errors = (
        error_model_signals.true_error_batch.detach().cpu().numpy().astype(np.float64)
    )

    transition_count = true_errors.shape[0]
    if transition_count < 4:
        raise ValueError(
            f"Cannot fit and calibrate an error model on {transition_count} "
            f"transitions."
        )

    permutation = np.random.default_rng(seed=split_seed).permutation(transition_count)
    fit_indices = permutation[: transition_count // 2]  # (transition_count // 2,)
    calibration_indices = permutation[transition_count // 2 :]

    feature_matrix = np.stack(
        [disagreements[fit_indices], sensitivity_scaled_step_distances[fit_indices]],
        axis=1,
    )  # (fit_count, 2)
    weights, sum_of_squared_errors = _solve_nonnegative_least_squares_through_origin(
        feature_matrix=feature_matrix, target=true_errors[fit_indices]
    )
    if weights.max() <= 0.0:
        raise ValueError(
            "Neither the disagreement nor the sensitivity-scaled step "
            "distance explains the validation error; "
            "the error model would be identically zero. Inspect the dynamics "
            "ensemble before augmenting."
        )

    fitted_calibration_values = (
        weights[0] * disagreements[calibration_indices]
        + weights[1] * sensitivity_scaled_step_distances[calibration_indices]
    )  # (calibration_count,)
    conformal_scale = float(
        np.quantile(
            true_errors[calibration_indices]
            / np.maximum(fitted_calibration_values, 1e-12),
            q=quantile,
        )
    )

    target_sum_of_squares = float(true_errors[fit_indices] @ true_errors[fit_indices])
    return ErrorModel(
        disagreement_weight=float(weights[0]),
        sensitivity_scaled_step_distance_weight=float(weights[1]),
        conformal_scale=conformal_scale,
        quantile=quantile,
        uncentered_r_squared=1.0
        - sum_of_squared_errors / max(target_sum_of_squares, 1e-30),
        fit_transition_count=int(fit_indices.shape[0]),
        calibration_transition_count=int(calibration_indices.shape[0]),
    )


def _solve_nonnegative_least_squares_through_origin(
    *,
    feature_matrix: npt.NDArray[np.float64],  # (sample_count, 2)
    target: npt.NDArray[np.float64],  # (sample_count,)
) -> tuple[npt.NDArray[np.float64], float]:  # weights (2,), sum of squared errors
    """Closed-form active-set solve: with two features the optimum is either
    the unconstrained solution (if feasible) or on one of the axes, so trying
    those four candidates and keeping the feasible one with the smallest
    residual is exact."""
    gram = feature_matrix.T @ feature_matrix  # (2, 2)
    correlation = feature_matrix.T @ target  # (2,)

    candidate_weight_vectors = [np.zeros(2)]
    for feature_index in range(2):
        if gram[feature_index, feature_index] > 0.0:
            single_feature_weights = np.zeros(2)
            single_feature_weights[feature_index] = max(
                correlation[feature_index] / gram[feature_index, feature_index], 0.0
            )
            candidate_weight_vectors.append(single_feature_weights)
    if np.linalg.det(gram) > 1e-30:
        unconstrained_weights = np.linalg.solve(gram, correlation)
        if (unconstrained_weights >= 0.0).all():
            candidate_weight_vectors.append(unconstrained_weights)

    def sum_of_squared_errors(weights: npt.NDArray[np.float64]) -> float:
        return float(
            target @ target - 2.0 * weights @ correlation + weights @ gram @ weights
        )

    best_weights = min(candidate_weight_vectors, key=sum_of_squared_errors)
    return best_weights, sum_of_squared_errors(best_weights)
