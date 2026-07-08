import numpy as np
import torch
from numpy import typing as npt

from scripts.data.augmentation.tacil.create_dynamics.losses import compute_loss
from scripts.data.augmentation.tacil.models.dynamics_model import DynamicsModel
from scripts.data.augmentation.tacil.models.normalizers import Normalizers


@torch.no_grad()
def calculate_validation_loss(
    *,
    dynamics_model: DynamicsModel,
    normalizers: Normalizers,
    state_windows: npt.NDArray[
        np.float32
    ],  # (window_count, window_length + 1, state_dimensionality)
    action_windows: npt.NDArray[
        np.float32
    ],  # (window_count, window_length, action_dimensionality)
    batch_size: int,
) -> float:

    device = next(dynamics_model.parameters()).device

    state_windows_tensor = torch.from_numpy(ndarray=state_windows).to(
        device=device
    )  # (window_count, window_length + 1, state_dimensionality)
    action_windows_tensor = torch.from_numpy(ndarray=action_windows).to(
        device=device
    )  # (window_count, window_length, action_dimensionality)

    window_count = state_windows_tensor.shape[0]
    total_loss = 0.0

    dynamics_model.eval()
    for batch_start in range(0, window_count, batch_size):
        batch_slice = slice(batch_start, batch_start + batch_size)
        state_window_batch = state_windows_tensor[
            batch_slice
        ]  # (batch_size, window_length + 1, state_dimensionality)
        batch_loss = compute_loss(
            dynamics_model=dynamics_model,
            normalizers=normalizers,
            state_window_batch=state_window_batch,
            action_window_batch=action_windows_tensor[batch_slice],
        )
        total_loss += batch_loss.item() * state_window_batch.shape[0]

    return total_loss / window_count


@torch.no_grad()
def calculate_local_lipschitz_constants(
    *,
    dynamics_model: DynamicsModel,
    normalizers: Normalizers,
    states: npt.NDArray[np.float32],  # (sample_count, state_dimensionality)
    actions: npt.NDArray[np.float32],  # (sample_count, action_dimensionality)
    batch_size: int,
) -> npt.NDArray[np.float32]:  # (sample_count,)
    """Local Lipschitz constant of the network at each (state, action) sample."""

    dynamics_model.eval()

    device = next(dynamics_model.parameters()).device
    states_tensor = torch.from_numpy(ndarray=states).to(device=device)
    actions_tensor = torch.from_numpy(ndarray=actions).to(device=device)

    local_lipschitz_constant_batches: list[npt.NDArray[np.float32]] = []
    for batch_start in range(0, states_tensor.shape[0], batch_size):
        batch_slice = slice(batch_start, batch_start + batch_size)
        local_lipschitz_constant_batches.append(
            dynamics_model.local_lipschitz_constants(
                normalizers=normalizers,
                state_batch=states_tensor[batch_slice],
                action_batch=actions_tensor[batch_slice],
            )
            .cpu()
            .numpy()
        )

    return np.concatenate(local_lipschitz_constant_batches)
