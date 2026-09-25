import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch.optim import Adam
from tqdm import tqdm
from wandb import Run

from scripts.policies.mopo.buffer import ReplayBuffer, concatenate_transitions
from scripts.policies.mopo.config import Config
from scripts.policies.mopo.dynamics.dynamics import Dynamics, TerminationFunction
from scripts.policies.mopo.dynamics.dynamics_training import train_dynamics
from scripts.policies.mopo.policy import CriticEnsemble, GaussianActor, MopoPolicy
from scripts.policies.mopo.transitions import Transitions
from scripts.policies.shared.checkpoint import (
    Checkpoint,
    beats_best_checkpoint,
    summarize_best_checkpoint,
)
from scripts.policies.shared.metrics import (
    EPOCH_METRIC_NAME,
    STAGE_NAME,
    STEP_METRIC_NAME,
    TRAINING_SECTION,
    VALIDATION_SECTION,
    define_step_metrics,
)
from scripts.shared.log import create_logger
from scripts.shared.seeding import set_seed

logger = create_logger()

ROLLOUT_SECTION = f"{STAGE_NAME}/3_rollout"


@dataclass(frozen=True)
class ValidationScore:
    mean: float
    std: float
    success_rate: float


def train_mopo_policy(
    *,
    config: Config,
    transitions: Transitions,
    termination_function: TerminationFunction,
    score_policy: Callable[[MopoPolicy], ValidationScore],
    selection_metric: Literal["reward", "success_then_reward"],
    seed: int,
    device: torch.device,
    run: Run,
) -> MopoPolicy:
    set_seed(seed=seed)
    define_step_metrics(
        run=run, sections=(TRAINING_SECTION, VALIDATION_SECTION, ROLLOUT_SECTION)
    )

    dynamics = train_dynamics(
        config=config.dynamics,
        rollout_penalty_coefficient=config.rollout.penalty_coefficient,
        transitions=transitions,
        termination_function=termination_function,
        seed=seed,
        device=device,
        run=run,
    )

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

    real_data_batch_size = int(
        config.optimization.batch_size * config.optimization.real_data_ratio
    )
    imagined_data_batch_size = config.optimization.batch_size - real_data_batch_size

    best_checkpoint: Checkpoint | None = None
    global_step = 0

    for epoch_index in tqdm(
        iterable=range(config.optimization.epoch_count), desc="Epoch", unit="epoch"
    ):
        policy.train()
        rollout_duration = 0.0
        epoch_start_timepoint = time.perf_counter()
        for _ in range(config.optimization.step_count_per_epoch):
            if global_step % config.rollout.interval_step_count == 0:
                rollout_start_timepoint = time.perf_counter()
                imagined_transitions, rollout_metrics = policy.imagine_rollout(
                    start_state_batch=real_buffer.sample_states(
                        count=config.rollout.start_state_count
                    ),
                    length=config.rollout.length,
                )
                imagined_buffer.add(transitions=imagined_transitions)
                rollout_duration += time.perf_counter() - rollout_start_timepoint
                run.log(
                    data={
                        f"{ROLLOUT_SECTION}/1_transition_count": rollout_metrics.transition_count,
                        f"{ROLLOUT_SECTION}/2_reward_mean": rollout_metrics.reward_mean,
                        f"{ROLLOUT_SECTION}/3_penalty_mean": rollout_metrics.penalty_mean,
                        STEP_METRIC_NAME: global_step,
                    }
                )

            policy.learn(
                batch=concatenate_transitions(
                    first=real_buffer.sample(count=real_data_batch_size),
                    second=imagined_buffer.sample(count=imagined_data_batch_size),
                )
            )
            global_step += 1

        epoch_metrics = policy.get_epoch_metrics()
        training_metrics = {
            f"{TRAINING_SECTION}/1_actor_loss": epoch_metrics.actor_loss,
            f"{TRAINING_SECTION}/2_critic_loss": epoch_metrics.critic_loss,
            f"{TRAINING_SECTION}/3_temperature_loss": epoch_metrics.temperature_loss,
            f"{TRAINING_SECTION}/4_temperature": epoch_metrics.temperature,
        }
        # Read after the averages, because that is where the device catches up.
        epoch_seconds = time.perf_counter() - epoch_start_timepoint

        run.log(
            data={
                **training_metrics,
                f"{TRAINING_SECTION}/5_learning_seconds": epoch_seconds
                - rollout_duration,
                f"{ROLLOUT_SECTION}/4_seconds": rollout_duration,
                EPOCH_METRIC_NAME: epoch_index,
                STEP_METRIC_NAME: global_step,
            }
        )

        actor_lr_schedule.step()

        if (epoch_index + 1) % config.validation.interval_epoch_count != 0:
            continue

        validation_score = score_policy(policy)
        policy.train()
        run.log(
            data={
                f"{VALIDATION_SECTION}/1_reward_mean": validation_score.mean,
                f"{VALIDATION_SECTION}/2_reward_std": validation_score.std,
                f"{VALIDATION_SECTION}/3_success_rate": validation_score.success_rate,
                EPOCH_METRIC_NAME: epoch_index,
                STEP_METRIC_NAME: global_step,
            }
        )

        if beats_best_checkpoint(
            best_checkpoint=best_checkpoint,
            selection_metric=selection_metric,
            validation_reward_mean=validation_score.mean,
            validation_success_rate=validation_score.success_rate,
        ):
            best_checkpoint = Checkpoint(
                validation_reward_mean=validation_score.mean,
                validation_reward_std=validation_score.std,
                validation_success_rate=validation_score.success_rate,
                epoch=epoch_index,
                state_dict=copy.deepcopy(policy.actor.state_dict()),
            )

    if best_checkpoint is None:
        raise ValueError("No checkpoint was recorded")

    summarize_best_checkpoint(run=run, best_checkpoint=best_checkpoint)

    if config.validation.kept_policy == "best":
        policy.actor.load_state_dict(state_dict=best_checkpoint.state_dict)
    return policy


# ---- Private --------------------------------------------


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
    critic = CriticEnsemble(
        state_dimensionality=state_dimensionality,
        action_dimensionality=action_dimensionality,
        hidden_layer_dimensionalities=config.optimization.hidden_layer_dimensionalities,
        critic_count=2,
    ).to(device=device)
    target_critic = copy.deepcopy(x=critic).eval()

    log_temperature = torch.zeros(1, requires_grad=True, device=device)

    card_is_too_old_for_compilation = (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device=device) < (7, 0)
    )
    compile_learning_step = config.optimization.compile_learning_step
    if compile_learning_step and card_is_too_old_for_compilation:
        logger.warning("Current device does not support torch.compile")
        compile_learning_step = False

    return MopoPolicy(
        actor=actor,
        actor_optimizer=Adam(
            params=actor.parameters(),
            lr=config.optimization.actor_learning_rate,
            fused=True,
        ),
        critic=critic,
        critic_optimizer=Adam(
            params=critic.parameters(),
            lr=config.optimization.critic_learning_rate,
            fused=True,
        ),
        target_critic=target_critic,
        target_critic_update_rate=config.optimization.target_critic_update_rate,
        log_temperature=log_temperature,
        temperature_optimizer=Adam(
            params=[log_temperature],
            lr=config.optimization.temperature_learning_rate,
            fused=True,
        ),
        target_entropy=-float(action_dimensionality),
        dynamics=dynamics,
        discount_factor=config.optimization.discount_factor,
        compile_learning_step=compile_learning_step,
        device=device,
    )
