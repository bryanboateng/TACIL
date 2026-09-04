from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig[PolicyConfigT, AugmentationT]:
    seed: int

    wandb_project: str

    expert_dataset_artifact: str

    policy: PolicyConfigT
    augmentation: AugmentationT | None

    evaluate_all_conditions: bool = False

    def __post_init__(self) -> None:
        # An empty name reaches wandb as a project of its own, or asks for an
        # artifact that cannot exist, and only fails once the run is under way.
        for name, value in (
            ("wandb_project", self.wandb_project),
            ("expert_dataset_artifact", self.expert_dataset_artifact),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty.")
