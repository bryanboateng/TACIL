import json
from dataclasses import asdict, fields
from enum import StrEnum
from pathlib import Path

import torch

from scripts.data.augmentation.tacil.models.dynamics_model import DynamicsModel
from scripts.data.augmentation.tacil.models.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.data.augmentation.tacil.models.error_model import ErrorModel
from scripts.data.augmentation.tacil.models.normalizers import Normalizers


def save_dynamics_model_ensemble(
    *, dynamics_model_ensemble: DynamicsModelEnsemble, directory_path: Path
) -> None:
    _save_normalizers(
        normalizers=dynamics_model_ensemble.normalizers,
        state_dimensionality=dynamics_model_ensemble.state_dimensionality,
        action_dimensionality=dynamics_model_ensemble.action_dimensionality,
        path=directory_path / _NORMALIZERS_FILE_NAME,
    )
    for index, dynamics_model in enumerate(
        iterable=dynamics_model_ensemble.dynamics_models
    ):
        _save_dynamics_model(
            dynamics_model=dynamics_model,
            path=directory_path / f"dynamics_model_{index}.pt",
        )


def load_dynamics_model_ensemble(
    *, directory_path: Path, device: torch.device
) -> DynamicsModelEnsemble:
    dynamics_model_file_paths = sorted(
        directory_path.glob(pattern="dynamics_model_*.pt"),
        key=lambda path: int(path.stem.removeprefix("dynamics_model_")),
    )

    if len(dynamics_model_file_paths) == 0:
        raise FileNotFoundError(
            f"No dynamics model checkpoints found in {directory_path}"
        )

    return DynamicsModelEnsemble(
        normalizers=_load_normalizers(
            path=directory_path / _NORMALIZERS_FILE_NAME, device=device
        ),
        dynamics_models=[
            _load_dynamics_model(path=dynamics_model_file_path, device=device)
            for dynamics_model_file_path in dynamics_model_file_paths
        ],
    ).to(device=device)


def save_error_model(*, error_model: ErrorModel, directory_path: Path) -> None:
    (directory_path / _ERROR_MODEL_FILE_NAME).write_text(
        data=json.dumps(obj=asdict(error_model), indent=2)
    )


def load_error_model(*, directory_path: Path) -> ErrorModel:
    error_model_file_path = directory_path / _ERROR_MODEL_FILE_NAME
    if not error_model_file_path.exists():
        raise FileNotFoundError(
            f"{error_model_file_path} not found; this dynamics ensemble "
            f"predates the calibrated error model. Retrain it with the "
            f"current training script."
        )
    saved_values = json.loads(error_model_file_path.read_text())
    # Take exactly the declared fields: extra keys from older checkpoints are
    # dropped, a missing key still fails loudly.
    return ErrorModel(
        **{field.name: saved_values[field.name] for field in fields(ErrorModel)}
    )


# ---- Private --------------------------------------------

_NORMALIZERS_FILE_NAME = "normalizers.pt"
_ERROR_MODEL_FILE_NAME = "error_model.json"


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


def _save_dynamics_model(*, dynamics_model: DynamicsModel, path: Path) -> None:
    torch.save(
        obj={
            _CheckpointKeys.STATE_DICTIONARY: dynamics_model.state_dict(),
            _CheckpointKeys.STATE_DIMENSIONALITY: dynamics_model.state_dimensionality,
            _CheckpointKeys.ACTION_DIMENSIONALITY: dynamics_model.action_dimensionality,
            _CheckpointKeys.HIDDEN_DIMENSIONALITIES: dynamics_model.hidden_dimensionalities,
            _CheckpointKeys.LIPSCHITZ_CONSTANT_UPPER_BOUND: dynamics_model.lipschitz_constant_upper_bound,
        },
        f=path,
    )


def _load_dynamics_model(*, path: Path, device: torch.device) -> DynamicsModel:
    checkpoint = torch.load(f=path, map_location=device, weights_only=False)
    dynamics_model = DynamicsModel(
        state_dimensionality=checkpoint[_CheckpointKeys.STATE_DIMENSIONALITY],
        action_dimensionality=checkpoint[_CheckpointKeys.ACTION_DIMENSIONALITY],
        hidden_dimensionalities=tuple(
            checkpoint[_CheckpointKeys.HIDDEN_DIMENSIONALITIES]
        ),
        lipschitz_constant_upper_bound=checkpoint[
            _CheckpointKeys.LIPSCHITZ_CONSTANT_UPPER_BOUND
        ],
    )
    dynamics_model.load_state_dict(
        state_dict=checkpoint[_CheckpointKeys.STATE_DICTIONARY]
    )
    return dynamics_model.to(device=device)
