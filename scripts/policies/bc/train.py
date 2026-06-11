"""Behavior-Cloning training. Nothing is written to the project tree: the best checkpoint
is kept in memory and, exactly like the dynamics-model trainer, serialized once at the end
into a TemporaryDirectory and pushed to W&B as a config-hashed artifact (no metadata).

The periodic eval needs a policy file to roll out, so it writes into a TemporaryDirectory
that is discarded immediately -- nothing persists locally either way.

Normalization is baked into the policy (see bc/policy.py): the dataset returns RAW
observations/actions, the model normalizes internally, and the training loss is computed
in normalized action space. The whole policy (weights + normalization buffers) is what
gets checkpointed.

One run = one model seed. Run 3x with different `seed` and average the final numbers for
error bars (that aggregation happens outside this script).

Goes to: scripts/policies/bc/train.py
"""

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

from scripts.ashared import create_logger, log_device_info, set_seed
from scripts.evaluation.bryan.evaluation import RewardStatistics, evaluate_policy
from scripts.policies.bc.config import Config
from scripts.policies.bc.dataset import SingleStepDataset
from scripts.policies.bc.policy import (
    BehaviorCloningPolicy,
    load_behavior_cloning_policy,
    save_behavior_cloning_policy,
)
from scripts.policies.shared.stat import Checkpoint
from scripts.shared.data import load_demonstration_data_from_artifact
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

logger = create_logger()


def main():
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))
    _run(config=config)


def _run(config: Config) -> None:
    set_seed(seed=config.seed)
    device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    log_device_info(device=device, logger=logger)

    loss_function = torch.nn.MSELoss()

    with (
        wandb.init(
            project="abc",
            job_type="train-behavior-cloning-policy",
            config=asdict(config),
            dir=OUTPUT_DIRECTORY_PATH,
        ) as run,
        tempfile.TemporaryDirectory() as checkpoint_directory,
    ):
        checkpoint_directory_path = Path(checkpoint_directory)

        dataset = SingleStepDataset(
            demonstration_data=load_demonstration_data_from_artifact(
                artifact=run.use_artifact(artifact_or_name=config.dataset_artifact),
                logger=logger,
            )
        )

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.optimization.batch_size,
            shuffle=True,
            pin_memory=device.type == "cuda",
        )

        # Built after the dataset because it needs the dataset's fitted (training-split)
        # normalization ranges, which it bakes in as buffers.
        policy = BehaviorCloningPolicy(
            observation_dimensionality=config.dimensionality.observation,
            action_dimensionality=config.dimensionality.action,
            hidden_dimensionality=config.dimensionality.hidden,
            observation_normalization_range=dataset.observation_normalization_range,
            action_normalization_range=dataset.action_normalization_range,
        ).to(device=device)

        optimizer = Adam(
            params=policy.parameters(), lr=config.optimization.learning_rate
        )

        best_checkpoint: Checkpoint | None = None

        for epoch_index in tqdm(
            iterable=range(config.optimization.epoch_count), desc="Epoch", unit="epoch"
        ):
            policy.train()
            for observation_batch, action_batch in data_loader:
                observation_batch = observation_batch.to(device=device)
                action_batch = action_batch.to(device=device)

                optimizer.zero_grad()

                # The dataset returns RAW values; the model normalizes the observation,
                # and we compare its normalized prediction against the normalized target.
                predicted_normalized_action = policy(
                    policy.normalize_observation(observation_batch)
                )
                target_normalized_action = policy.normalize_action(action_batch)
                loss = loss_function(
                    predicted_normalized_action, target_normalized_action
                )

                loss.backward()

                optimizer.step()

                run.log(data={"train/loss": loss.item()})

            periodic_eval_is_due = (
                epoch_index + 1
            ) % config.evaluation.periodic_interval_epoch_count == 0 or (
                epoch_index + 1
            ) == config.optimization.epoch_count
            if not periodic_eval_is_due:
                continue

            checkpoint_path = checkpoint_directory_path / "checkpoint.pt"
            save_behavior_cloning_policy(policy=policy, path=checkpoint_path)
            validation_result = _evaluate(
                checkpoint_path=checkpoint_path,
                episode_seeds=config.evaluation.episode_seeds,
                device=device,
            )
            run.log(
                data={
                    "validation/max_reward_mean": validation_result.mean,
                    "epoch": epoch_index,
                }
            )

            if (
                best_checkpoint is None
                or validation_result.mean > best_checkpoint.validation_max_reward_mean
            ):
                best_checkpoint = Checkpoint(
                    validation_max_reward_mean=validation_result.mean,
                    epoch=epoch_index,
                    state_dict=copy.deepcopy(policy.state_dict()),
                )

        if best_checkpoint is None:
            raise ValueError("No checkpoint was recorded")

        policy.load_state_dict(state_dict=best_checkpoint.state_dict)
        logger.info(
            "Selected best epoch %d (validation mean %.4f).",
            best_checkpoint.epoch,
            best_checkpoint.validation_max_reward_mean,
        )

        _report_final_and_log_artifact(
            policy=policy, best_epoch=best_checkpoint.epoch, run=run
        )


def _report_final_and_log_artifact(
    policy: BehaviorCloningPolicy, best_epoch: int, run: Run
) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "policy.pt"
        save_behavior_cloning_policy(policy=policy, path=checkpoint_path)

        run.summary["selected/epoch"] = best_epoch

        model_artifact = wandb.Artifact(
            name=f"behavior-cloning-policy-{run.name}", type="model"
        )
        model_artifact.add_file(local_path=str(checkpoint_path), name="policy.pt")
        run.log_artifact(model_artifact).wait()


def _evaluate(
    checkpoint_path: Path, episode_seeds: tuple[int, ...], device: torch.device
) -> RewardStatistics:
    policy_factory = load_behavior_cloning_policy(
        checkpoint_path=checkpoint_path, device=device
    )
    return evaluate_policy(
        policy_factory=policy_factory,
        episode_seeds=episode_seeds,
        disturber_config=None,
        description="Validation",
    )


def _write_final_summary(run: Run, protocol: str, result: RewardStatistics) -> None:
    run.summary[f"final_{protocol}/max_reward_mean"] = result.mean
    run.summary[f"final_{protocol}/max_reward_std"] = result.std


if __name__ == "__main__":
    main()
