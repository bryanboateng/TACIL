import copy

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from wandb import Run

from scripts.evaluation.core.condition import VALIDATION_CONDITION
from scripts.evaluation.core.scoring import (
    compute_reward_statistics,
    run_scored_episodes,
)
from scripts.policies.bc.config import Config
from scripts.policies.bc.dataset import Dataset
from scripts.policies.bc.policy import (
    BehaviorCloningPolicyModel,
    create_behavior_cloning_policy_factory,
)
from scripts.policies.shared.checkpoint import Checkpoint
from scripts.shared.log import create_logger
from scripts.shared.normalization import MinMaxNormalizer, ZScoreNormalizer
from scripts.shared.seeding import set_seed
from scripts.shared.transition_batch import TransitionBatch

logger = create_logger()

STAGE_NAME = "3_policy"

STEP_METRIC_NAME = f"{STAGE_NAME}/step"


def train_behavior_cloning_policy(
    *,
    config: Config,
    transitions: TransitionBatch,
    seed: int,
    device: torch.device,
    run: Run,
) -> BehaviorCloningPolicyModel:
    set_seed(seed=seed)
    _define_step_metrics(run=run)

    loss_function = torch.nn.MSELoss()

    run.summary[f"{STAGE_NAME}/transition_count"] = len(transitions.states)

    dataset = Dataset(transitions=transitions)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.optimization.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    # The transitions state their own widths, so the environment's layout never
    # has to be named here.
    policy_model = BehaviorCloningPolicyModel(
        observation_dimensionality=transitions.states.shape[1],
        action_dimensionality=transitions.actions.shape[1],
        hidden_layer_dimensionality=config.hidden_layer_dimensionality,
        observation_normalizer=ZScoreNormalizer(
            value_batch=dataset.states.to(device=device)
        ),
        action_normalizer=MinMaxNormalizer(
            value_batch=dataset.actions.to(device=device)
        ),
    ).to(device=device)

    optimizer = Adam(
        params=policy_model.parameters(), lr=config.optimization.learning_rate
    )

    best_checkpoint: Checkpoint | None = None
    global_step = 0

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
                input=policy_model.action_normalizer.normalize(
                    value_batch=predicted_action_batch
                ),
                target=policy_model.action_normalizer.normalize(
                    value_batch=action_batch
                ),
            )

            loss.backward()

            gradient_norm = torch.norm(
                input=torch.stack(
                    tensors=[
                        parameter.grad.norm()
                        for parameter in policy_model.parameters()
                        if parameter.grad is not None
                    ]
                )
            )

            optimizer.step()

            run.log(
                data={
                    f"{STAGE_NAME}/1_training/loss": loss.item(),
                    f"{STAGE_NAME}/1_training/gradient_norm": gradient_norm.item(),
                    STEP_METRIC_NAME: global_step,
                }
            )
            global_step += 1

        if (epoch_index + 1) % config.validation.interval_epoch_count != 0:
            continue

        validation_reward = compute_reward_statistics(
            episode_records=run_scored_episodes(
                policy_factory=create_behavior_cloning_policy_factory(
                    policy_model=policy_model, device=device
                ),
                start_episode_seed=100_000,
                episode_count=config.validation.episode_count,
                condition=VALIDATION_CONDITION,
            )
        )
        run.log(
            data={
                f"{STAGE_NAME}/2_validation/reward_mean": validation_reward.mean,
                f"{STAGE_NAME}/2_validation/reward_std": validation_reward.std,
                f"{STAGE_NAME}/epoch": epoch_index,
                STEP_METRIC_NAME: global_step,
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

    run.summary[f"{STAGE_NAME}/completed_step_count"] = global_step
    run.summary[f"{STAGE_NAME}/3_best_checkpoint/epoch"] = best_checkpoint.epoch
    run.summary[f"{STAGE_NAME}/3_best_checkpoint/validation_reward_mean"] = (
        best_checkpoint.validation_reward_mean
    )
    run.summary[f"{STAGE_NAME}/3_best_checkpoint/validation_reward_std"] = (
        best_checkpoint.validation_reward_std
    )

    policy_model.load_state_dict(state_dict=best_checkpoint.state_dict)
    return policy_model


def _define_step_metrics(*, run: Run) -> None:
    run.define_metric(name=STEP_METRIC_NAME)
    for metric_pattern in ("epoch", "1_training/*", "2_validation/*"):
        run.define_metric(
            name=f"{STAGE_NAME}/{metric_pattern}", step_metric=STEP_METRIC_NAME
        )
