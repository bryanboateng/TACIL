from dataclasses import dataclass

from scripts.environments.registry import EnvironmentName
from scripts.evaluation.config import EvaluationConfig
from scripts.shared.dataset_artifact import build_dataset_name_prefix


@dataclass(frozen=True)
class RunConfig[PolicyConfigT, AugmentationT]:
    seed: int

    environment: EnvironmentName

    wandb_project: str

    expert_dataset_artifact: str

    evaluation: EvaluationConfig

    policy: PolicyConfigT
    augmentation: AugmentationT | None

    record_videos: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("wandb_project", self.wandb_project),
            ("expert_dataset_artifact", self.expert_dataset_artifact),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty.")

        prefix = build_dataset_name_prefix(environment=self.environment)
        if not self.expert_dataset_artifact.startswith(prefix):
            raise ValueError(
                f"expert_dataset_artifact {self.expert_dataset_artifact!r} does not "
                f"belong to {self.environment}; it has to start with {prefix!r}."
            )
