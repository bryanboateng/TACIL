from enum import StrEnum
from pathlib import Path

import torch

from scripts.data.augmentation.tacil.dynamics_ensemble import DynamicsEnsemble
from scripts.data.augmentation.tacil.dynamics_network import DynamicsNetwork
from scripts.data.augmentation.tacil.normalizers import Normalizers


def save_dynamics_ensemble(*, ensemble: DynamicsEnsemble, directory: Path) -> None:
    _save_normalizers(
        normalizers=ensemble.normalizers,
        state_dimensionality=ensemble.state_dimensionality,
        action_dimensionality=ensemble.action_dimensionality,
        path=directory / _NORMALIZERS_FILE_NAME,
    )
    for network_index, network in enumerate(iterable=ensemble.networks):
        _save_network(network=network, path=directory / f"network_{network_index}.pt")


def load_dynamics_ensemble(
    *, directory: Path, device: torch.device
) -> DynamicsEnsemble:
    network_file_paths = sorted(
        directory.glob(pattern="network_*.pt"),
        key=lambda path: int(path.stem.removeprefix("network_")),
    )

    if len(network_file_paths) == 0:
        raise FileNotFoundError(f"No dynamics network checkpoints found in {directory}")

    return DynamicsEnsemble(
        normalizers=_load_normalizers(
            path=directory / _NORMALIZERS_FILE_NAME, device=device
        ),
        networks=[
            _load_network(path=network_file_path, device=device)
            for network_file_path in network_file_paths
        ],
    ).to(device=device)


# ---- Private --------------------------------------------

_NORMALIZERS_FILE_NAME = "normalizers.pt"


class _CheckpointKeys(StrEnum):
    STATE_DICTIONARY = "state_dictionary"
    STATE_DIMENSIONALITY = "state_dimensionality"
    ACTION_DIMENSIONALITY = "action_dimensionality"
    HIDDEN_DIMENSIONALITIES = "hidden_dimensionalities"
    LIPSCHITZ_CONSTANT_UPPER_BOUND = "lipschitz_constant_upper_bound"


def _save_normalizers(
    *,
    normalizers: Normalizers,
    state_dimensionality: int,
    action_dimensionality: int,
    path: Path,
) -> None:
    torch.save(
        obj={
            _CheckpointKeys.STATE_DICTIONARY: normalizers.state_dict(),
            _CheckpointKeys.STATE_DIMENSIONALITY: state_dimensionality,
            _CheckpointKeys.ACTION_DIMENSIONALITY: action_dimensionality,
        },
        f=path,
    )


def _load_normalizers(*, path: Path, device: torch.device) -> Normalizers:
    checkpoint = torch.load(f=path, map_location=device, weights_only=False)
    normalizers = Normalizers.placeholder(
        state_dimensionality=checkpoint[_CheckpointKeys.STATE_DIMENSIONALITY],
        action_dimensionality=checkpoint[_CheckpointKeys.ACTION_DIMENSIONALITY],
    )
    normalizers.load_state_dict(state_dict=checkpoint[_CheckpointKeys.STATE_DICTIONARY])
    return normalizers.to(device=device)


def _save_network(*, network: DynamicsNetwork, path: Path) -> None:
    torch.save(
        obj={
            _CheckpointKeys.STATE_DICTIONARY: network.state_dict(),
            _CheckpointKeys.STATE_DIMENSIONALITY: network.state_dimensionality,
            _CheckpointKeys.ACTION_DIMENSIONALITY: network.action_dimensionality,
            _CheckpointKeys.HIDDEN_DIMENSIONALITIES: network.hidden_dimensionalities,
            _CheckpointKeys.LIPSCHITZ_CONSTANT_UPPER_BOUND: network.lipschitz_constant_upper_bound,
        },
        f=path,
    )


def _load_network(*, path: Path, device: torch.device) -> DynamicsNetwork:
    checkpoint = torch.load(f=path, map_location=device, weights_only=False)
    network = DynamicsNetwork(
        state_dimensionality=checkpoint[_CheckpointKeys.STATE_DIMENSIONALITY],
        action_dimensionality=checkpoint[_CheckpointKeys.ACTION_DIMENSIONALITY],
        hidden_dimensionalities=tuple(
            checkpoint[_CheckpointKeys.HIDDEN_DIMENSIONALITIES]
        ),
        lipschitz_constant_upper_bound=checkpoint[
            _CheckpointKeys.LIPSCHITZ_CONSTANT_UPPER_BOUND
        ],
    )
    network.load_state_dict(state_dict=checkpoint[_CheckpointKeys.STATE_DICTIONARY])
    return network.to(device=device)
