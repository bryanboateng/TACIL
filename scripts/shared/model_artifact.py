import tempfile
from pathlib import Path

import torch
import wandb
from wandb import Run

POLICY_ARTIFACT_TYPE = "policy"
DYNAMICS_MODEL_ARTIFACT_TYPE = "dynamics_model"

_STATE_DICT_FILE_NAME = "state_dict.pt"


def upload_model(*, model: torch.nn.Module, artifact_type: str, run: Run) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        state_dict_path = Path(temporary_directory) / _STATE_DICT_FILE_NAME
        torch.save(obj=model.state_dict(), f=state_dict_path)

        artifact = wandb.Artifact(
            name=f"{artifact_type.replace('_', '-')}-{run.id}", type=artifact_type
        )
        artifact.add_file(local_path=str(state_dict_path))
        run.log_artifact(artifact_or_path=artifact)


def download_model_state_dict(
    *, artifact: wandb.Artifact, device: torch.device
) -> dict[str, torch.Tensor]:
    return torch.load(
        f=Path(artifact.download()) / _STATE_DICT_FILE_NAME, map_location=device
    )
