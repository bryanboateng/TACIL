import copy

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from wandb import Run

from scripts.evaluation.core.condition import Condition
from scripts.evaluation.core.scoring import reward_statistics, run_scored_episodes
from scripts.policies.bc.config import Config
from scripts.policies.bc.dataset import Dataset
from scripts.policies.bc.policy import (
    BehaviorCloningPolicyModel,
    create_behavior_cloning_policy_factory,
)
from scripts.policies.shared.checkpoint import Checkpoint
from scripts.shared.f import TransitionBatch
from scripts.shared.log import create_logger
from scripts.shared.normalization import MinMaxNormalizer, ZScoreNormalizer
from scripts.shared.pusht_state import ACTION_DIMENSIONALITY, STATE_DIMENSIONALITY
from scripts.shared.seeding import set_seed

logger = create_logger()

STAGE_NAME = "3_policy"


def train_behavior_cloning_policy(
    *,
    config: Config,
    transitions: TransitionBatch,
    condition: Condition,
    seed: int,
    device: torch.device,
    run: Run,
) -> BehaviorCloningPolicyModel:
    set_seed(seed=seed)

    loss_function = torch.nn.MSELoss()

    run.summary[f"{STAGE_NAME}/transition_count"] = len(transitions.states)

    dataset = Dataset(transitions=transitions)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.optimization.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    policy_model = BehaviorCloningPolicyModel(
        observation_dimensionality=STATE_DIMENSIONALITY,
        action_dimensionality=ACTION_DIMENSIONALITY,
        hidden_layer_dimensionality=config.hidden_layer_dimensionality,
        observation_normalizer=ZScoreNormalizer.from_values(
            value_batch=dataset.states.to(device=device)
        ),
        action_normalizer=MinMaxNormalizer.from_values(
            value_batch=dataset.actions.to(device=device)
        ),
    ).to(device=device)

    optimizer = Adam(
        params=policy_model.parameters(), lr=config.optimization.learning_rate
    )

    best_checkpoint: Checkpoint | None = None

    for epoch_index in tqdm(
        iterable=range(config.optimization.epoch_count), desc="Epoch", unit="epoch"
    ):
        policy_model.train()
        for observation_batch, action_batch in data_loader:
            observation_batch = observation_batch.to(
                device=device
            )  # (batch_size, observation_dimensionality)
            action_batch = action_batch.to(
                device=device
            )  # (batch_size, action_dimensionality)

            optimizer.zero_grad()

            predicted_action_batch = policy_model.predict_action(
                observation_batch=observation_batch
            )  # (batch_size, action_dimensionality)

            loss = loss_function(
                input=policy_model.normalize_action(
                    action_batch=predicted_action_batch
                ),
                target=policy_model.normalize_action(action_batch=action_batch),
            )

            loss.backward()

            optimizer.step()

            run.log(data={f"{STAGE_NAME}/1_training/loss": loss.item()})

        periodic_evaluation_is_due = (
            (epoch_index + 1) % config.evaluation.periodic_interval_epoch_count == 0
        ) or ((epoch_index + 1) == config.optimization.epoch_count)
        if not periodic_evaluation_is_due:
            continue

        validation_reward = reward_statistics(
            episode_records=run_scored_episodes(
                policy_factory=create_behavior_cloning_policy_factory(
                    policy_model=policy_model, device=device
                ),
                start_episode_seed=100_000,
                episode_count=config.evaluation.episode_count,
                condition=condition,
            )
        )
        run.log(
            data={
                f"{STAGE_NAME}/2_validation/reward_mean": validation_reward.mean,
                f"{STAGE_NAME}/2_validation/reward_std": validation_reward.std,
                f"{STAGE_NAME}/epoch": epoch_index,
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
                state_dict=copy.deepcopy(policy_model.state_dict()),
            )

    if best_checkpoint is None:
        raise ValueError("No checkpoint was recorded")

    run.summary[f"{STAGE_NAME}/3_selected/epoch"] = best_checkpoint.epoch
    run.summary[f"{STAGE_NAME}/3_selected/reward_mean"] = (
        best_checkpoint.validation_reward_mean
    )
    run.summary[f"{STAGE_NAME}/3_selected/reward_std"] = (
        best_checkpoint.validation_reward_std
    )

    policy_model.load_state_dict(state_dict=best_checkpoint.state_dict)
    return policy_model
