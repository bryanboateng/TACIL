import copy
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
import tyro
import wandb
from rich.pretty import pretty_repr
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from wandb import Run

from scripts.evaluation.core.condition import Condition, FloorCondition
from scripts.evaluation.core.scoring import (
    RewardStatistics,
    reward_statistics,
    run_scored_episodes,
)
from scripts.policies.bc.config import Config
from scripts.policies.bc.dataset import Dataset
from scripts.policies.bc.policy import (
    BehaviorCloningPolicy,
    load_behavior_cloning_policy,
    save_behavior_cloning_policy,
)
from scripts.policies.shared.checkpoint import Checkpoint
from scripts.shared.data import load_episode_store
from scripts.shared.log import create_logger, log_device_info
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH
from scripts.shared.seed import set_seed

logger = create_logger()


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(*, config: Config) -> None:
    set_seed(seed=config.seed)
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    loss_function = torch.nn.MSELoss()

    with (
        wandb.init(
            project=config.wandb_project,
            job_type="train-behavior-cloning-policy",
            config=asdict(config),
            dir=OUTPUT_DIRECTORY_PATH,
        ) as run,
        tempfile.TemporaryDirectory() as checkpoint_directory,
    ):
        checkpoint_directory_path = Path(checkpoint_directory)

        dataset = Dataset(
            episode_store=load_episode_store(
                episode_file_path=Path(
                    run.use_artifact(
                        artifact_or_name=config.dataset_artifact
                    ).download()
                )
            )
        )

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.optimization.batch_size,
            shuffle=True,
            pin_memory=device.type == "cuda",
        )

        policy = BehaviorCloningPolicy(
            observation_dimensionality=config.dimensionality.observation,
            action_dimensionality=config.dimensionality.action,
            hidden_dimensionality=config.dimensionality.hidden,
        ).to(device=device)

        policy.initialize_normalizers(
            observation_batch=dataset.states.to(device=device),
            action_batch=dataset.actions.to(device=device),
        )

        optimizer = Adam(
            params=policy.parameters(), lr=config.optimization.learning_rate
        )

        best_checkpoint: Checkpoint | None = None

        for epoch_index in tqdm(
            iterable=range(config.optimization.epoch_count), desc="Epoch", unit="epoch"
        ):
            policy.train()
            for observation_batch, action_batch in data_loader:
                observation_batch = observation_batch.to(
                    device=device
                )  # (batch_size, observation_dimensionality)
                action_batch = action_batch.to(
                    device=device
                )  # (batch_size, action_dimensionality)

                optimizer.zero_grad()

                predicted_action_batch = policy.predict_action(
                    observation=observation_batch
                )  # (batch_size, action_dimensionality)
                loss = loss_function(
                    policy.normalize_action(action=predicted_action_batch),
                    policy.normalize_action(action=action_batch),
                )

                loss.backward()

                optimizer.step()

                run.log(data={"train/loss": loss.item()})

            periodic_eval_is_due = (
                (epoch_index + 1) % config.evaluation.periodic_interval_epoch_count == 0
            ) or ((epoch_index + 1) == config.optimization.epoch_count)
            if not periodic_eval_is_due:
                continue

            checkpoint_path = checkpoint_directory_path / "checkpoint.pt"
            save_behavior_cloning_policy(policy=policy, path=checkpoint_path)
            validation_reward = _evaluate(
                checkpoint_path=checkpoint_path,
                episode_count=config.evaluation.episode_count,
                device=device,
            )
            run.log(
                data={
                    "validation/reward_mean": validation_reward.mean,
                    "validation/reward_std": validation_reward.std,
                    "epoch": epoch_index,
                }
            )

            if (
                best_checkpoint is None
                or validation_reward.mean > best_checkpoint.validation_reward_mean
            ):
                best_checkpoint = Checkpoint(
                    validation_reward_mean=validation_reward.mean,
                    validation_reward_std=validation_reward.std,
                    epoch=epoch_index,
                    state_dict=copy.deepcopy(policy.state_dict()),
                )

        if best_checkpoint is None:
            raise ValueError("No checkpoint was recorded")

        run.summary["selected/epoch"] = best_checkpoint.epoch
        run.summary["selected/reward_mean"] = best_checkpoint.validation_reward_mean
        run.summary["selected/reward_std"] = best_checkpoint.validation_reward_std

        policy.load_state_dict(state_dict=best_checkpoint.state_dict)
        _upload_policy(policy=policy, run=run)


def _upload_policy(*, policy: BehaviorCloningPolicy, run: Run) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
        save_behavior_cloning_policy(policy=policy, path=checkpoint_path)

        model_artifact = wandb.Artifact(
            name=f"behavior-cloning-policy-{run.name}", type="model"
        )
        model_artifact.add_file(local_path=str(checkpoint_path), name="checkpoint.pt")
        run.log_artifact(model_artifact).wait()


def _evaluate(
    *, checkpoint_path: Path, episode_count: int, device: torch.device
) -> RewardStatistics:
    policy_factory = load_behavior_cloning_policy(
        checkpoint_path=checkpoint_path, device=device
    )
    return reward_statistics(
        episode_scores=run_scored_episodes(
            policy_factory=policy_factory,
            start_episode_seed=100_000,
            episode_count=episode_count,
            condition=Condition(
                floor_condition=FloorCondition.GRIPPY, disturbance=None
            ),
        )
    )


if __name__ == "__main__":
    main()
