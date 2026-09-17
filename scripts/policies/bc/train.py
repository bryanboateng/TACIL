import copy

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from wandb import Run

from scripts.evaluation.core.scoring import (
    calculate_reward_statistics,
    run_scored_episodes_over_conditions,
)
from scripts.policies.bc.config import Config
from scripts.policies.bc.dataset import Dataset
from scripts.policies.bc.policy import (
    BehaviorCloningPolicyModel,
    create_behavior_cloning_policy_factory,
)
from scripts.policies.shared.checkpoint import Checkpoint
from scripts.shared import stage_names
from scripts.shared.log import create_logger
from scripts.shared.normalization import MinMaxNormalizer, ZScoreNormalizer
from scripts.shared.seeding import set_seed
from scripts.shared.state_action_pairs import StateActionPairs

logger = create_logger()

STAGE_NAME = stage_names.POLICY

STEP_METRIC_NAME = f"{STAGE_NAME}/0_step"
EPOCH_METRIC_NAME = f"{STAGE_NAME}/0_epoch"

TRAINING_SECTION = f"{STAGE_NAME}/1_training"
VALIDATION_SECTION = f"{STAGE_NAME}/2_validation"


def train_behavior_cloning_policy(
    *,
    config: Config,
    state_action_pairs: StateActionPairs,
    seed: int,
    device: torch.device,
    run: Run,
) -> BehaviorCloningPolicyModel:
    set_seed(seed=seed)
    _define_step_metrics(run=run)

    loss_function = torch.nn.MSELoss()

    run.summary[f"{STAGE_NAME}/0_dataset/1_transition_count"] = len(
        state_action_pairs.states
    )

    dataset = Dataset(state_action_pairs=state_action_pairs)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.optimization.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    model = BehaviorCloningPolicyModel(
        observation_dimensionality=state_action_pairs.states.shape[1],
        action_dimensionality=state_action_pairs.actions.shape[1],
        hidden_layer_dimensionality=config.hidden_layer_dimensionality,
        observation_normalizer=ZScoreNormalizer(
            value_batch=dataset.states.to(device=device)
        ),
        action_normalizer=MinMaxNormalizer(
            value_batch=dataset.actions.to(device=device)
        ),
    ).to(device=device)

    optimizer = Adam(params=model.parameters(), lr=config.optimization.learning_rate)

    best_checkpoint: Checkpoint | None = None
    global_step = 0

    for epoch_index in tqdm(
        iterable=range(config.optimization.epoch_count), desc="Epoch", unit="epoch"
    ):
        model.train()
        for observation_batch, target_action_batch in data_loader:
            observation_batch = observation_batch.to(
                device=device
            )  # (batch_size, observation_dimensionality)
            target_action_batch = target_action_batch.to(
                device=device
            )  # (batch_size, action_dimensionality)

            optimizer.zero_grad()

            predicted_action_batch = model(
                observation_batch, config.observation_noise
            )  # (batch_size, action_dimensionality)

            loss = loss_function(
                input=model.action_normalizer.normalize(
                    value_batch=predicted_action_batch
                ),
                target=model.action_normalizer.normalize(
                    value_batch=target_action_batch
                ),
            )

            loss.backward()

            gradient_norm = torch.norm(
                input=torch.stack(
                    tensors=[
                        parameter.grad.norm()
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]
                )
            )

            optimizer.step()

            run.log(
                data={
                    f"{TRAINING_SECTION}/1_loss": loss.item(),
                    f"{TRAINING_SECTION}/2_gradient_norm": gradient_norm.item(),
                    STEP_METRIC_NAME: global_step,
                }
            )
            global_step += 1

        if (
            config.validation is None
            or (epoch_index + 1) % config.validation.interval_epoch_count != 0
        ):
            continue

        with torch.no_grad():
            validation_reward = calculate_reward_statistics(
                episode_records=run_scored_episodes_over_conditions(
                    policy_factory=create_behavior_cloning_policy_factory(
                        policy_model=model, device=device
                    ),
                    episode_count_per_condition=config.validation.episode_count_per_condition,
                )
            )
        run.log(
            data={
                f"{VALIDATION_SECTION}/1_reward_mean": validation_reward.mean,
                f"{VALIDATION_SECTION}/2_reward_std": validation_reward.std,
                EPOCH_METRIC_NAME: epoch_index,
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
                state_dict=copy.deepcopy(model.state_dict()),
            )

    if config.validation is None:
        return model

    if best_checkpoint is None:
        raise ValueError("No checkpoint was recorded")

    run.summary[f"{STAGE_NAME}/3_best_checkpoint/1_epoch"] = best_checkpoint.epoch
    run.summary[f"{STAGE_NAME}/3_best_checkpoint/2_validation_reward_mean"] = (
        best_checkpoint.validation_reward_mean
    )
    run.summary[f"{STAGE_NAME}/3_best_checkpoint/3_validation_reward_std"] = (
        best_checkpoint.validation_reward_std
    )

    model.load_state_dict(state_dict=best_checkpoint.state_dict)
    return model


def _define_step_metrics(*, run: Run) -> None:
    run.define_metric(name=STEP_METRIC_NAME)
    for name in (EPOCH_METRIC_NAME, f"{TRAINING_SECTION}/*", f"{VALIDATION_SECTION}/*"):
        run.define_metric(name=name, step_metric=STEP_METRIC_NAME)
