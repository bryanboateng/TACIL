import copy
import time
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from wandb import Run

from scripts.environments.environment import Environment
from scripts.evaluation.config import VALIDATION_START_SEED
from scripts.evaluation.core.scoring import (
    ScoringSession,
    calculate_reward_statistics,
    calculate_success_statistics,
)
from scripts.policies.mopo.buffer import ReplayBuffer, concatenate_transitions
from scripts.policies.mopo.config import Config, ValidationConfig
from scripts.policies.mopo.dynamics.dynamics import Dynamics, TerminationFunction
from scripts.policies.mopo.dynamics.dynamics_training import train_dynamics
from scripts.policies.mopo.policy import (
    CriticEnsemble,
    GaussianActor,
    MopoPolicy,
    create_mopo_policy_factory,
)
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
from scripts.policies.shared.validation import Validation
from scripts.shared.log import create_logger
from scripts.shared.model_artifact import DYNAMICS_MODEL_ARTIFACT_TYPE, upload_model
from scripts.shared.normalization import MinMaxNormalizer, ZScoreNormalizer
from scripts.shared.seeding import set_seed

logger = create_logger()

ROLLOUT_SECTION = f"{STAGE_NAME}/3_rollout"


@dataclass(frozen=True)
class NormalizedEnvironment:
    environment: Environment
    state_normalizer: ZScoreNormalizer
    action_normalizer: MinMaxNormalizer


def create_termination_function(
    *, normalized_environment: NormalizedEnvironment
) -> TerminationFunction:
    def calculate_termination(
        normalized_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size,)
        state_batch = (
            normalized_environment.state_normalizer.denormalize(
                value_batch=normalized_state_batch.cpu()
            )
            .numpy()
            .astype(np.float32)
        )  # (batch_size, state_dimensionality)

        return torch.from_numpy(
            normalized_environment.environment.calculate_termination(
                state_batch=state_batch
            )
        ).to(device=normalized_state_batch.device)

    return calculate_termination


def train_mopo_policy(
    *,
    config: Config,
    transitions: Transitions,
    termination_function: TerminationFunction,
    validation: Validation[ValidationConfig, NormalizedEnvironment] | None,
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
    if config.dynamics.save_model:
        upload_model(
            model=dynamics.model_ensemble,
            artifact_type=DYNAMICS_MODEL_ARTIFACT_TYPE,
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

    with ExitStack() as scoring_resources:
        scoring_session: ScoringSession | None = None

        for epoch_index in tqdm(
            iterable=range(config.optimization.epoch_count),
            desc="Epoch",
            unit="epoch",
            dynamic_ncols=True,
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

            if (
                validation is None
                or (epoch_index + 1) % validation.settings.interval_epoch_count != 0
            ):
                continue

            normalized_environment = validation.environment

            if scoring_session is None:
                scoring_session = scoring_resources.enter_context(
                    ScoringSession(
                        environment=normalized_environment.environment,
                        parallel_environment_count=validation.settings.parallel_environment_count,
                    )
                )

            validation_start_timepoint = time.perf_counter()
            with torch.no_grad():
                episode_records = scoring_session.roll_out_episodes_with_separate_seeds_per_condition(
                    policy_factory=create_mopo_policy_factory(
                        policy=policy,
                        state_normalizer=normalized_environment.state_normalizer,
                        action_normalizer=normalized_environment.action_normalizer,
                        device=device,
                    ),
                    start_episode_seed=VALIDATION_START_SEED,
                    episode_count_per_condition=validation.settings.episode_count_per_condition,
                )
                validation_reward = calculate_reward_statistics(
                    episode_records=episode_records
                )
            validation_success = calculate_success_statistics(
                episode_records=episode_records
            ).rate
            policy.train()
            run.log(
                data={
                    f"{VALIDATION_SECTION}/1_reward_mean": validation_reward.mean,
                    f"{VALIDATION_SECTION}/2_reward_std": validation_reward.std,
                    f"{VALIDATION_SECTION}/3_success_rate": validation_success,
                    f"{VALIDATION_SECTION}/4_seconds": time.perf_counter()
                    - validation_start_timepoint,
                    EPOCH_METRIC_NAME: epoch_index,
                    STEP_METRIC_NAME: global_step,
                }
            )

            if beats_best_checkpoint(
                best_checkpoint=best_checkpoint,
                selection_metric=normalized_environment.environment.checkpoint_selection_metric,
                validation_reward_mean=validation_reward.mean,
                validation_success_rate=validation_success,
            ):
                best_checkpoint = Checkpoint(
                    validation_reward_mean=validation_reward.mean,
                    validation_reward_std=validation_reward.std,
                    validation_success_rate=validation_success,
                    epoch=epoch_index,
                    state_dict=copy.deepcopy(policy.actor.state_dict()),
                )

    if validation is None:
        return policy

    if best_checkpoint is None:
        raise ValueError("No checkpoint was recorded")

    summarize_best_checkpoint(run=run, best_checkpoint=best_checkpoint)

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
