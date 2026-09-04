from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig[PolicyConfigT, AugmentationT]:
    seed: int

    wandb_project: str

    expert_dataset_artifact: str

    policy: PolicyConfigT
    augmentation: AugmentationT | None

    # Mirror the expert data about the goal's axis of symmetry before anything
    # else happens. This is a stage, not a strategy: the reflection is exact,
    # so what comes out is expert data, and whatever augmentation follows sees
    # twice as much of it -- twice the transitions to fit the dynamics model
    # on, twice the anchors to grow chains from, twice the episodes to stitch
    # between. On its own, with no augmentation, it is the mirror strategy.
    mirror_expert_data: bool = False

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
