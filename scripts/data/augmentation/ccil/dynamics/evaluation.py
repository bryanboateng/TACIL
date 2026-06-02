import numpy as np
import torch
from numpy import typing as npt

from scripts.data.augmentation.ccil.dynamics.losses import compute_loss
from scripts.data.augmentation.ccil.dynamics_model import DynamicsModel


@torch.no_grad()
def calculate_validation_loss(
    dynamics_model: DynamicsModel,
    validation_state_windows: npt.NDArray,  # (window_count, window_length + 1, state_dimensionality)
    validation_action_windows: npt.NDArray,  # (window_count, window_length, action_dimensionality)
    batch_size: int,
) -> float:

    model_device = next(dynamics_model.parameters()).device

    state_windows_tensor = torch.from_numpy(validation_state_windows).to(
        device=model_device
    )
    action_windows_tensor = torch.from_numpy(validation_action_windows).to(
        device=model_device
    )

    window_count = state_windows_tensor.shape[0]
    total_loss = 0.0
    batch_count = 0

    dynamics_model.eval()
    for batch_start in range(0, window_count, batch_size):
        batch_slice = slice(batch_start, batch_start + batch_size)
        batch_loss = compute_loss(
            dynamics_model=dynamics_model,
            state_window_batch=state_windows_tensor[batch_slice],
            action_window_batch=action_windows_tensor[batch_slice],
        )
        total_loss += batch_loss.item()
        batch_count += 1

    return total_loss / batch_count


@torch.no_grad()
def calculate_local_lipschitz_constants(
    dynamics_model: DynamicsModel,
    states: npt.NDArray[np.float32],  # (sample_count, state_dimensionality)
    actions: npt.NDArray[np.float32],  # (sample_count, action_dimensionality)
    batch_size: int,
) -> np.ndarray:  # (sample_count,)
    """Local Lipschitz constant of the model at each (state, action) sample."""

    dynamics_model.eval()

    model_device = next(dynamics_model.parameters()).device
    states_tensor = torch.from_numpy(states).to(device=model_device)
    actions_tensor = torch.from_numpy(actions).to(device=model_device)

    spectral_norm_batches: list[np.ndarray] = []
    for batch_start in range(0, states_tensor.shape[0], batch_size):
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
