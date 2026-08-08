import copy
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from wandb import Run

from scripts.data.augmentation.ccil.f import TransitionBatch
from scripts.evaluation.core.condition import Condition
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
from scripts.shared.log import create_logger
from scripts.shared.seed import set_seed

logger = create_logger()

# Every key this stage logs starts with it, so a stage run and a pipeline run
# report the same numbers under the same names.
STAGE_PREFIX = "3_policy-"


def train_behavior_cloning_policy(
    *,
    config: Config,
    transitions: TransitionBatch,
    checkpoint_directory_path: Path,
    device: torch.device,
    run: Run,
) -> Path:
    set_seed(seed=config.seed)

    loss_function = torch.nn.MSELoss()

    dataset = Dataset(transitions=transitions)

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

    optimizer = Adam(params=policy.parameters(), lr=config.optimization.learning_rate)

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

            run.log(data={f"{STAGE_PREFIX}training/loss": loss.item()})

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
            condition=Condition(
                floor_condition=config.evaluation.floor_condition,
                disturbance_level=config.evaluation.disturbance_level,
            ),
            device=device,
        )
        run.log(
            data={
                f"{STAGE_PREFIX}validation/reward_mean": validation_reward.mean,
                f"{STAGE_PREFIX}validation/reward_std": validation_reward.std,
                f"{STAGE_PREFIX}epoch": epoch_index,
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

    run.summary[f"{STAGE_PREFIX}selected/epoch"] = best_checkpoint.epoch
    run.summary[f"{STAGE_PREFIX}selected/reward_mean"] = (
        best_checkpoint.validation_reward_mean
    )
    run.summary[f"{STAGE_PREFIX}selected/reward_std"] = (
        best_checkpoint.validation_reward_std
    )

    policy.load_state_dict(state_dict=best_checkpoint.state_dict)

    final_path = checkpoint_directory_path / "checkpoint.pt"
    save_behavior_cloning_policy(policy=policy, path=final_path)
    return final_path


def _evaluate(
    *,
    checkpoint_path: Path,
    episode_count: int,
    condition: Condition,
    device: torch.device,
) -> RewardStatistics:
    policy_factory = load_behavior_cloning_policy(
        checkpoint_path=checkpoint_path, device=device
    )
    return reward_statistics(
        episode_records=run_scored_episodes(
            policy_factory=policy_factory,
            start_episode_seed=100_000,
            episode_count=episode_count,
            condition=condition,
        )
    )
