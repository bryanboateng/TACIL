from dataclasses import dataclass

from scripts.environments.peg_insertion import condition as peg_insertion_condition
from scripts.environments.pusht import condition as pusht_condition
from scripts.environments.registry import EnvironmentName
from scripts.evaluation.config import VALIDATION_START_SEED, EvaluationConfig
from scripts.policies.bc import config as bc_config
from scripts.policies.dp import config as dp_config
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

        if (
            isinstance(self.policy, (bc_config.Config, dp_config.Config))
            and self.policy.validation is not None
        ):
            match self.environment:
                case EnvironmentName.PUSH_T:
                    conditions = pusht_condition.ALL_CONDITIONS
                case EnvironmentName.PEG_INSERTION:
                    conditions = peg_insertion_condition.ALL_CONDITIONS
            validation_end = (
                VALIDATION_START_SEED
                + len(conditions) * self.policy.validation.episode_count_per_condition
            )
            evaluation_end = (
                self.evaluation.start_episode_seed
                + self.evaluation.episode_count_per_condition
            )
            if (
                self.evaluation.start_episode_seed < validation_end
                and evaluation_end > VALIDATION_START_SEED
            ):
                raise ValueError(
                    "Validation and final evaluation episode seeds must not overlap"
                )

        prefix = build_dataset_name_prefix(environment=self.environment)
        if not self.expert_dataset_artifact.startswith(prefix):
            raise ValueError(
                f"expert_dataset_artifact {self.expert_dataset_artifact!r} does not "
                f"belong to {self.environment}; it has to start with {prefix!r}."
            )
