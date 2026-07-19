from dataclasses import dataclass

import numpy as np
import scipy.optimize
import torch

from scripts.data.augmentation.tacil.models.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)


@dataclass(frozen=True)
class ErrorModelTrainingData:
    """One row per held-out transition: the two predictors the error model
    combines, and the true one-step prediction error they should explain."""

    disagreements: torch.Tensor  # (transition_count,)
    sensitivity_scaled_step_distances: torch.Tensor  # (transition_count,)
    true_errors: torch.Tensor  # (transition_count,)


@torch.no_grad()
def compute_error_model_training_data(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    states: torch.Tensor,  # (transition_count, state_dimensionality)
    actions: torch.Tensor,  # (transition_count, action_dimensionality)
    next_states: torch.Tensor,  # (transition_count, state_dimensionality)
) -> ErrorModelTrainingData:
    batch_size = 4_096

    disagreement_batches: list[torch.Tensor] = []
    sensitivity_scaled_step_distance_batches: list[torch.Tensor] = []
    true_error_batches: list[torch.Tensor] = []

    for batch_start_index in range(0, states.shape[0], batch_size):
        batch_slice = slice(batch_start_index, batch_start_index + batch_size)

        disagreement_batches.append(
            dynamics_model_ensemble.disagreements_about_next_state(
                state_batch=states[batch_slice], action_batch=actions[batch_slice]
            )
        )
        sensitivity_scaled_step_distance_batches.append(
            dynamics_model_ensemble.sensitivity_scaled_step_distances(
                state_batch=states[batch_slice],
                next_state_batch=next_states[batch_slice],
                action_batch=actions[batch_slice],
            )
        )
        true_error_batches.append(
            torch.linalg.norm(
                dynamics_model_ensemble.normalize_state(
                    state_batch=dynamics_model_ensemble.predict_next_state(
                        state_batch=states[batch_slice],
                        action_batch=actions[batch_slice],
                    )
                )
                - dynamics_model_ensemble.normalize_state(
                    state_batch=next_states[batch_slice]
                ),
                dim=-1,
            )
        )

    return ErrorModelTrainingData(
        disagreements=torch.cat(tensors=disagreement_batches),
        sensitivity_scaled_step_distances=torch.cat(
            tensors=sensitivity_scaled_step_distance_batches
        ),
        true_errors=torch.cat(tensors=true_error_batches),
    )


@dataclass(frozen=True)
class ErrorModel:
    disagreement_weight: float
    sensitivity_scaled_step_distance_weight: float
    # Multiplier that turns the weighted signal sum from an average-case guess
    # into an overestimate that holds 90% of the time (split-conformal
    # calibration).
    safety_factor: float

    def estimate_step_errors(
        self,
        *,
        disagreement_batch: torch.Tensor,  # (batch_size,)
        sensitivity_scaled_step_distance_batch: torch.Tensor,  # (batch_size,)
    ) -> torch.Tensor:  # (batch_size,)
        return self.safety_factor * (
            self.disagreement_weight * disagreement_batch
            + self.sensitivity_scaled_step_distance_weight
            * sensitivity_scaled_step_distance_batch
        )


def fit_error_model(
    *, training_data: ErrorModelTrainingData, split_seed: int
) -> tuple[ErrorModel, float]:
    """Returns the fitted model and its uncentered r-squared — the fraction of
    the true error the two signals explain. The r-squared is a pure fit-quality
    diagnostic; nothing computes with it."""
    disagreements = (
        training_data.disagreements.detach().cpu().numpy().astype(np.float64)
    )
    sensitivity_scaled_step_distances = (
        training_data.sensitivity_scaled_step_distances.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    true_errors = training_data.true_errors.detach().cpu().numpy().astype(np.float64)

    transition_count = true_errors.shape[0]
    if transition_count < 4:
        raise ValueError(
            f"Cannot fit and calibrate an error model on {transition_count} "
            f"transitions."
        )

    permutation = np.random.default_rng(seed=split_seed).permutation(transition_count)
    fit_indices = permutation[: transition_count // 2]
    calibration_indices = permutation[transition_count // 2 :]

    weights, residual_norm = scipy.optimize.nnls(
        A=np.stack(
            arrays=[
                disagreements[fit_indices],
                sensitivity_scaled_step_distances[fit_indices],
            ],
            axis=1,
        ),
        b=true_errors[fit_indices],
    )
    if weights.max() <= 0.0:
        raise ValueError(
            "Neither the disagreement nor the sensitivity-scaled step "
            "distance explains the validation error."
            "The error model would be identically zero."
        )
    disagreement_weight = weights[0]
    sensitivity_scaled_step_distances_weight = weights[1]

    fitted_calibration_values = (
        disagreement_weight * disagreements[calibration_indices]
        + sensitivity_scaled_step_distances_weight
        * sensitivity_scaled_step_distances[calibration_indices]
    )  # (calibration_count,)
    safety_factor = float(
        np.quantile(
            a=true_errors[calibration_indices]
            / np.maximum(fitted_calibration_values, 1e-12),
            q=0.9,
        )
    )

    target_sum_of_squares = float(true_errors[fit_indices] @ true_errors[fit_indices])
    uncentered_r_squared = 1.0 - float(residual_norm) ** 2 / max(
        target_sum_of_squares, 1e-30
    )

    return (
        ErrorModel(
            disagreement_weight=float(disagreement_weight),
            sensitivity_scaled_step_distance_weight=float(
                sensitivity_scaled_step_distances_weight
            ),
            safety_factor=safety_factor,
        ),
        uncentered_r_squared,
    )
