import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from wandb import Run

from scripts.evaluation.core.scoring import (
    calculate_reward_statistics,
    run_scored_episodes_over_conditions,
)
from scripts.policies.mopo.buffer import ReplayBuffer, concatenate_batches
from scripts.policies.mopo.config import Config
from scripts.policies.mopo.dynamics import Dynamics
from scripts.policies.mopo.dynamics_training import train_dynamics
from scripts.policies.mopo.policy import (
    Critic,
    GaussianActor,
    MopoPolicy,
    create_mopo_policy_factory,
)
from scripts.policies.mopo.transitions import Transitions
from scripts.shared import stage_names
from scripts.shared.log import create_logger
from scripts.shared.normalization import MinMaxNormalizer, ZScoreNormalizer
from scripts.shared.pusht_reward import calculate_termination
from scripts.shared.seeding import set_seed

logger = create_logger()

STAGE_NAME = stage_names.POLICY

STEP_METRIC_NAME = f"{STAGE_NAME}/0_step"
EPOCH_METRIC_NAME = f"{STAGE_NAME}/0_epoch"

TRAINING_SECTION = f"{STAGE_NAME}/1_training"
VALIDATION_SECTION = f"{STAGE_NAME}/2_validation"
ROLLOUT_SECTION = f"{STAGE_NAME}/3_rollout"


def train_mopo_policy(
    *,
    config: Config,
    transitions: Transitions,
    state_normalizer: ZScoreNormalizer,
    action_normalizer: MinMaxNormalizer,
    seed: int,
    device: torch.device,
    run: Run,
) -> MopoPolicy:
    set_seed(seed=seed)
    _define_step_metrics(run=run)

    dynamics = train_dynamics(
        config=config.dynamics,
        rollout_penalty_coefficient=config.rollout.penalty_coefficient,
        transitions=transitions,
        termination_function=lambda state_batch: calculate_termination(
            state_batch=state_normalizer.denormalize(
                value_batch=torch.from_numpy(state_batch).float()
            ).numpy()
        ),
        seed=seed,
        device=device,
        run=run,
    )

    generator = np.random.default_rng(seed=seed)

    state_dimensionality = transitions.states.shape[1]
    action_dimensionality = transitions.actions.shape[1]

    policy = _create_policy(
        config=config,
        dynamics=dynamics,
        state_dimensionality=state_dimensionality,
        action_dimensionality=action_dimensionality,
        device=device,
    )
    actor_lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=policy.actor_optimizer, T_max=config.optimization.epoch_count
    )

    real_buffer = ReplayBuffer(
        capacity=len(transitions.states),
        state_dimensionality=state_dimensionality,
        action_dimensionality=action_dimensionality,
        device=device,
    )
    real_buffer.add(transitions=transitions)

    imagined_buffer = ReplayBuffer(
        capacity=config.rollout.start_state_count
        * config.rollout.length
        * config.rollout.retained_epoch_count,
        state_dimensionality=state_dimensionality,
        action_dimensionality=action_dimensionality,
        device=device,
    )

    real_batch_size = int(
        config.optimization.batch_size * config.optimization.real_data_ratio
    )
    imagined_batch_size = config.optimization.batch_size - real_batch_size

    best_checkpoint: _Checkpoint | None = None
    global_step = 0

    for epoch_index in tqdm(
        iterable=range(config.optimization.epoch_count), desc="Epoch", unit="epoch"
    ):
        policy.train()
        for _ in range(config.optimization.step_count_per_epoch):
            if global_step % config.rollout.interval_step_count == 0:
                imagined_transitions, rollout_statistics = policy.imagine_rollout(
                    start_state_batch=real_buffer.sample_states(
                        count=config.rollout.start_state_count, generator=generator
                    ),
                    length=config.rollout.length,
                    generator=generator,
                )
                imagined_buffer.add(transitions=imagined_transitions)
                run.log(
                    data={
                        f"{ROLLOUT_SECTION}/1_transition_count": rollout_statistics.transition_count,
                        f"{ROLLOUT_SECTION}/2_reward_mean": rollout_statistics.reward_mean,
                        f"{ROLLOUT_SECTION}/3_penalty_mean": rollout_statistics.penalty_mean,
                        STEP_METRIC_NAME: global_step,
                    }
                )

            learning_statistics = policy.learn(
                batch=concatenate_batches(
                    first=real_buffer.sample(
                        count=real_batch_size, generator=generator
                    ),
                    second=imagined_buffer.sample(
                        count=imagined_batch_size, generator=generator
                    ),
                )
            )

            run.log(
                data={
                    f"{TRAINING_SECTION}/1_actor_loss": learning_statistics.actor_loss,
                    f"{TRAINING_SECTION}/2_critic_loss": learning_statistics.critic_loss,
                    f"{TRAINING_SECTION}/3_temperature_loss": learning_statistics.temperature_loss,
                    f"{TRAINING_SECTION}/4_temperature": learning_statistics.temperature,
                    STEP_METRIC_NAME: global_step,
                }
            )
            global_step += 1

        actor_lr_schedule.step()

        if (epoch_index + 1) % config.validation.interval_epoch_count != 0:
            continue

        validation_reward = calculate_reward_statistics(
            episode_records=run_scored_episodes_over_conditions(
                policy_factory=create_mopo_policy_factory(
                    policy=policy,
                    state_normalizer=state_normalizer,
                    action_normalizer=action_normalizer,
                    device=device,
                ),
                start_episode_seed=100_000,
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
            best_checkpoint = _Checkpoint(
                validation_reward_mean=validation_reward.mean,
                validation_reward_std=validation_reward.std,
                epoch=epoch_index,
                actor_state_dict=copy.deepcopy(policy.actor.state_dict()),
            )

    if best_checkpoint is None:
        raise ValueError("No checkpoint was recorded")

    run.summary[f"{STAGE_NAME}/4_best_checkpoint/1_epoch"] = best_checkpoint.epoch
    run.summary[f"{STAGE_NAME}/4_best_checkpoint/2_validation_reward_mean"] = (
        best_checkpoint.validation_reward_mean
    )
    run.summary[f"{STAGE_NAME}/4_best_checkpoint/3_validation_reward_std"] = (
        best_checkpoint.validation_reward_std
    )

    policy.actor.load_state_dict(state_dict=best_checkpoint.actor_state_dict)
    return policy


# ---- Private --------------------------------------------


@dataclass(frozen=True)
class _Checkpoint:
    validation_reward_mean: float
    validation_reward_std: float
    epoch: int
    actor_state_dict: dict[str, torch.Tensor]


def _create_policy(
    *,
    config: Config,
    dynamics: Dynamics,
    state_dimensionality: int,
    action_dimensionality: int,
    device: torch.device,
) -> MopoPolicy:
    actor = GaussianActor(
        state_dimensionality=state_dimensionality,
        action_dimensionality=action_dimensionality,
        hidden_layer_dimensionalities=config.optimization.hidden_layer_dimensionalities,
    ).to(device=device)
    first_critic = Critic(
        state_dimensionality=state_dimensionality,
        action_dimensionality=action_dimensionality,
        hidden_layer_dimensionalities=config.optimization.hidden_layer_dimensionalities,
    ).to(device=device)
    second_critic = Critic(
        state_dimensionality=state_dimensionality,
        action_dimensionality=action_dimensionality,
        hidden_layer_dimensionalities=config.optimization.hidden_layer_dimensionalities,
    ).to(device=device)

    log_temperature = torch.zeros(1, requires_grad=True, device=device)

    return MopoPolicy(
        actor=actor,
        first_critic=first_critic,
        second_critic=second_critic,
        actor_optimizer=Adam(
            params=actor.parameters(), lr=config.optimization.actor_learning_rate
        ),
        first_critic_optimizer=Adam(
            params=first_critic.parameters(),
            lr=config.optimization.critic_learning_rate,
        ),
        second_critic_optimizer=Adam(
            params=second_critic.parameters(),
            lr=config.optimization.critic_learning_rate,
        ),
        log_temperature=log_temperature,
        temperature_optimizer=Adam(
            params=[log_temperature], lr=config.optimization.temperature_learning_rate
        ),
        target_entropy=-float(action_dimensionality),
        dynamics=dynamics,
        discount_factor=config.optimization.discount_factor,
        target_update_rate=config.optimization.target_update_rate,
        device=device,
    )


def _define_step_metrics(*, run: Run) -> None:
    run.define_metric(name=STEP_METRIC_NAME)
    for name in (
        EPOCH_METRIC_NAME,
        f"{TRAINING_SECTION}/*",
        f"{VALIDATION_SECTION}/*",
        f"{ROLLOUT_SECTION}/*",
    ):
        run.define_metric(name=name, step_metric=STEP_METRIC_NAME)
