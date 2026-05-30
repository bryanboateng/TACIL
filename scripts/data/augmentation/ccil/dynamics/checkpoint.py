from enum import StrEnum
from pathlib import Path

import torch
import wandb
from wandb import Run

from scripts.data.augmentation.ccil.dynamics_model import DynamicsModel


class CheckpointKeys(StrEnum):
    STATE_DICT = "state_dict"
    STATE_DIMENSIONALITY = "state_dimensionality"
    ACTION_DIMENSIONALITY = "action_dimensionality"
    HIDDEN_DIMENSIONALITIES = "hidden_dimensionalities"
    ACTIVATION_FUNCTION_NAME = "activation_function_name"
    LIPSCHITZ_CONSTANT_UPPER_BOUND = "lipschitz_constant_upper_bound"


def save_checkpoint(
    dynamics_model: DynamicsModel, output_directory_path: Path, run: Run
) -> None:
    dynamics_checkpoint_path = output_directory_path / "dynamics.pt"
    save_dynamics_checkpoint(
        dynamics_model=dynamics_model, path=dynamics_checkpoint_path
    )

    model_artifact = wandb.Artifact(
        name="dynamics_model",
        type="model",
        metadata={
            CheckpointKeys.STATE_DIMENSIONALITY: dynamics_model.state_dimensionality,
            CheckpointKeys.ACTION_DIMENSIONALITY: dynamics_model.action_dimensionality,
            CheckpointKeys.HIDDEN_DIMENSIONALITIES: list(
                dynamics_model.hidden_dimensionalities
            ),
            CheckpointKeys.ACTIVATION_FUNCTION_NAME: dynamics_model.activation_function_name,
            CheckpointKeys.LIPSCHITZ_CONSTANT_UPPER_BOUND: dynamics_model.lipschitz_constant_upper_bound,
        },
    )
    model_artifact.add_file(
        local_path=str(dynamics_checkpoint_path), name="dynamics.pt"
    )
    run.log_artifact(artifact_or_path=model_artifact)


def save_dynamics_checkpoint(dynamics_model: DynamicsModel, path: Path) -> None:
    torch.save(
        obj={
            CheckpointKeys.STATE_DICT: dynamics_model.state_dict(),
            CheckpointKeys.STATE_DIMENSIONALITY: dynamics_model.state_dimensionality,
            CheckpointKeys.ACTION_DIMENSIONALITY: dynamics_model.action_dimensionality,
            CheckpointKeys.HIDDEN_DIMENSIONALITIES: dynamics_model.hidden_dimensionalities,
            CheckpointKeys.ACTIVATION_FUNCTION_NAME: dynamics_model.activation_function_name,
            CheckpointKeys.LIPSCHITZ_CONSTANT_UPPER_BOUND: dynamics_model.lipschitz_constant_upper_bound,
        },
        f=path,
    )


def load_dynamics_checkpoint(path: Path, device: torch.device) -> DynamicsModel:
    checkpoint = torch.load(f=path, map_location=device, weights_only=False)
    dynamics_model = DynamicsModel(
        state_dimensionality=checkpoint[CheckpointKeys.STATE_DIMENSIONALITY],
        action_dimensionality=checkpoint[CheckpointKeys.ACTION_DIMENSIONALITY],
        hidden_dimensionalities=tuple(
            checkpoint[CheckpointKeys.HIDDEN_DIMENSIONALITIES]
        ),
        activation_function_name=checkpoint[CheckpointKeys.ACTIVATION_FUNCTION_NAME],
        lipschitz_constant_upper_bound=checkpoint[
            CheckpointKeys.LIPSCHITZ_CONSTANT_UPPER_BOUND
        ],
    )
    dynamics_model.load_state_dict(checkpoint[CheckpointKeys.STATE_DICT])
    return dynamics_model.to(device=device)
