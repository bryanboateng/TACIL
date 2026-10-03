from dataclasses import asdict, dataclass

from scripts.environments.registry import EnvironmentName
from scripts.evaluation.config import EvaluationConfig, resolve_evaluation_stage


@dataclass(frozen=True)
class RunConfig[PolicyConfigT, AugmentationT]:
    seed: int

    environment: EnvironmentName

    wandb_project: str

    expert_dataset_artifact: str

    policy: PolicyConfigT
    augmentation: AugmentationT | None

    evaluation: EvaluationConfig

    def __post_init__(self) -> None:
        for name, value in (
            ("wandb_project", self.wandb_project),
            ("expert_dataset_artifact", self.expert_dataset_artifact),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty.")


def build_run_record[PolicyConfigT, AugmentationT](
    *,
    config: RunConfig[PolicyConfigT, AugmentationT],
    policy_name: str,
    augmentation_name: str | None,
) -> dict[str, object]:
    if (config.augmentation is None) != (augmentation_name is None):
        raise ValueError(
            f"augmentation_name is {augmentation_name!r}, which does not match an "
            f"augmentation of {type(config.augmentation).__name__}."
        )

    stage = resolve_evaluation_stage(stage=config.evaluation.stage)

    record = asdict(obj=config)
    record["policy_name"] = policy_name
    record["evaluation"] = {
        "stage": stage.name,
        "start_episode_seed": stage.start_episode_seed,
        "episode_count_per_condition": stage.episode_count_per_condition,
        "parallel_environment_count": config.evaluation.parallel_environment_count,
    }
    if augmentation_name is not None:
        record["augmentation"] = {"name": augmentation_name} | record["augmentation"]
    return record
