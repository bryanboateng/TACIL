from dataclasses import dataclass

from scripts.evaluation.config import EvaluationConfig


@dataclass(frozen=True)
class RunConfig[PolicyConfigT, AugmentationT]:
    seed: int

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
