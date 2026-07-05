import numpy as np
import torch
from numpy import typing as npt

from scripts.data.augmentation.tacil.dynamics.losses import compute_loss
from scripts.data.augmentation.tacil.dynamics_network import DynamicsNetwork
from scripts.data.augmentation.tacil.normalizers import Normalizers


@torch.no_grad()
def calculate_validation_loss(
    *,
    network: DynamicsNetwork,
    normalizers: Normalizers,
    validation_state_windows: npt.NDArray[
        np.float32
    ],  # (window_count, window_length + 1, state_dimensionality)
    validation_action_windows: npt.NDArray[
        np.float32
    ],  # (window_count, window_length, action_dimensionality)
    batch_size: int,
) -> float:

    device = next(network.parameters()).device

    state_windows_tensor = torch.from_numpy(validation_state_windows).to(
        device=device
    )  # (window_count, window_length + 1, state_dimensionality)
    action_windows_tensor = torch.from_numpy(validation_action_windows).to(
        device=device
    )  # (window_count, window_length, action_dimensionality)

    window_count = state_windows_tensor.shape[0]
    total_loss = 0.0

    network.eval()
    for batch_start in range(0, window_count, batch_size):
        batch_slice = slice(batch_start, batch_start + batch_size)
        batch_state_windows = state_windows_tensor[
            batch_slice
        ]  # (batch_size, window_length + 1, state_dimensionality)
        batch_loss = compute_loss(
            network=network,
            normalizers=normalizers,
            state_window_batch=batch_state_windows,
            action_window_batch=action_windows_tensor[batch_slice],
        )
        total_loss += batch_loss.item() * batch_state_windows.shape[0]

    return total_loss / window_count


@torch.no_grad()
def calculate_local_lipschitz_constants(
    *,
    network: DynamicsNetwork,
    normalizers: Normalizers,
    states: npt.NDArray[np.float32],  # (sample_count, state_dimensionality)
    actions: npt.NDArray[np.float32],  # (sample_count, action_dimensionality)
    batch_size: int,
) -> npt.NDArray[np.float32]:  # (sample_count,)
    """Local Lipschitz constant of the network at each (state, action) sample."""

    network.eval()

    device = next(network.parameters()).device
    states_tensor = torch.from_numpy(states).to(device=device)
    actions_tensor = torch.from_numpy(actions).to(device=device)

    spectral_norm_batches: list[npt.NDArray[np.float32]] = []
    for batch_start in range(0, states_tensor.shape[0], batch_size):
        batch_slice = slice(batch_start, batch_start + batch_size)
        spectral_norm_batches.append(
            network.local_lipschitz_constants(
                normalizers=normalizers,
                state_batch=states_tensor[batch_slice],
                action_batch=actions_tensor[batch_slice],
            )
            .cpu()
            .numpy()
        )

    return np.concatenate(spectral_norm_batches)
