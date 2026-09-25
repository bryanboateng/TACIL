import copy
from contextlib import ExitStack
from time import perf_counter

import torch
from torch.optim import Adam
from tqdm.auto import tqdm
from wandb import Run

from scripts.environments.environment import Environment
from scripts.evaluation.core.scoring import (
    ScoringSession,
    calculate_reward_statistics,
    calculate_success_statistics,
)
from scripts.policies.bc.config import Config
from scripts.policies.bc.dataset import Dataset
from scripts.policies.bc.policy import (
    BehaviorCloningPolicyModel,
    create_behavior_cloning_policy_factory,
)
from scripts.policies.shared.checkpoint import (
    Checkpoint,
    calculate_checkpoint_selection_key,
    summarize_best_checkpoint,
)
from scripts.shared import stage_names
from scripts.shared.log import create_logger
from scripts.shared.normalization import MinMaxNormalizer, ZScoreNormalizer
from scripts.shared.seeding import set_seed
from scripts.shared.state_action_transitions import StateActionTransitions

logger = create_logger()

STAGE_NAME = stage_names.POLICY

STEP_METRIC_NAME = f"{STAGE_NAME}/0_step"
EPOCH_METRIC_NAME = f"{STAGE_NAME}/0_epoch"

TRAINING_SECTION = f"{STAGE_NAME}/1_training"
VALIDATION_SECTION = f"{STAGE_NAME}/2_validation"


def train_behavior_cloning_policy(
    *,
    config: Config,
    environment: Environment | None,
    state_action_transitions: StateActionTransitions,
    seed: int,
    device: torch.device,
    run: Run,
) -> BehaviorCloningPolicyModel:
    set_seed(seed=seed)
    _define_step_metrics(run=run)

    loss_function = torch.nn.MSELoss()

    run.summary[f"{STAGE_NAME}/0_dataset/1_transition_count"] = len(
        state_action_transitions.states
    )

    dataset = Dataset(state_action_transitions=state_action_transitions)

    # These low-dimensional datasets are small enough to batch directly on the
    # training device, without fetching and collating each individual sample.
    states = dataset.states.to(device=device)
    actions = dataset.actions.to(device=device)

    model = BehaviorCloningPolicyModel(
        observation_dimensionality=state_action_transitions.states.shape[1],
        action_dimensionality=state_action_transitions.actions.shape[1],
        hidden_layer_dimensionality=config.hidden_layer_dimensionality,
        observation_normalizer=ZScoreNormalizer(value_batch=states),
        action_normalizer=MinMaxNormalizer(value_batch=actions),
        excluded_observation_indices=config.excluded_observation_indices,
    ).to(device=device)

    optimizer = Adam(params=model.parameters(), lr=config.optimization.learning_rate)

    best_checkpoint: Checkpoint | None = None
    global_step = 0

    with ExitStack() as scoring_resources:
        scoring_session: ScoringSession | None = None
        for epoch_index in tqdm(
            iterable=range(config.optimization.epoch_count), desc="Epoch", unit="epoch"
        ):
            model.train()
            training_start = perf_counter()
            epoch_loss = torch.zeros((), device=device)
            permutation = torch.randperm(len(dataset), device=device)
            for batch_index, batch_indices in enumerate(
                permutation.split(config.optimization.batch_size)
            ):
                observation_batch = states[batch_indices]
                target_action_batch = actions[batch_indices]

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

                # A gradient snapshot per epoch avoids thousands of small kernels
                # and device synchronizations solely for telemetry.
                if batch_index == 0:
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

                epoch_loss += loss.detach() * len(batch_indices)
                global_step += 1

            mean_loss = (epoch_loss / len(dataset)).item()
            run.log(
                data={
                    f"{TRAINING_SECTION}/1_loss": mean_loss,
                    f"{TRAINING_SECTION}/2_gradient_norm": gradient_norm.item(),
                    f"{TRAINING_SECTION}/3_seconds": perf_counter() - training_start,
                    EPOCH_METRIC_NAME: epoch_index,
                    STEP_METRIC_NAME: global_step,
                }
            )

            if (
                config.validation is None
                or (epoch_index + 1) % config.validation.interval_epoch_count != 0
            ):
                continue

            if environment is None:
                raise ValueError(
                    "Scoring the policy while it trains needs an environment to score "
                    "it in, but none was given."
                )

            if scoring_session is None:
                scoring_session = scoring_resources.enter_context(
                    ScoringSession(
                        environment=environment,
                        parallel_environment_count=config.validation.parallel_environment_count,
                    )
                )

            validation_start = perf_counter()
            with torch.no_grad():
                episode_records = scoring_session.run_over_conditions(
                    policy_factory=create_behavior_cloning_policy_factory(
                        policy_model=model, device=device
                    ),
                    episode_count_per_condition=config.validation.episode_count_per_condition,
                )
                validation_reward = calculate_reward_statistics(
                    episode_records=episode_records
                )
            validation_success = calculate_success_statistics(
                episode_records=episode_records
            ).rate
            run.log(
                data={
                    f"{VALIDATION_SECTION}/1_reward_mean": validation_reward.mean,
                    f"{VALIDATION_SECTION}/2_reward_std": validation_reward.std,
                    f"{VALIDATION_SECTION}/3_success_rate": validation_success,
                    f"{VALIDATION_SECTION}/4_seconds": perf_counter()
                    - validation_start,
                    EPOCH_METRIC_NAME: epoch_index,
                    STEP_METRIC_NAME: global_step,
                }
            )

            if best_checkpoint is None or calculate_checkpoint_selection_key(
                selection_metric=environment.checkpoint_selection_metric,
                validation_reward_mean=validation_reward.mean,
                validation_success_rate=validation_success,
            ) > calculate_checkpoint_selection_key(
                selection_metric=environment.checkpoint_selection_metric,
                validation_reward_mean=best_checkpoint.validation_reward_mean,
                validation_success_rate=best_checkpoint.validation_success_rate,
            ):
                best_checkpoint = Checkpoint(
                    validation_reward_mean=validation_reward.mean,
                    validation_reward_std=validation_reward.std,
                    validation_success_rate=validation_success,
                    epoch=epoch_index,
                    step=global_step,
                    state_dict=copy.deepcopy(model.state_dict()),
                )

    if config.validation is None:
        return model

    if best_checkpoint is None:
        raise ValueError("No checkpoint was recorded")

    assert environment is not None
    summarize_best_checkpoint(
        run=run, best_checkpoint=best_checkpoint, environment=environment
    )

    model.load_state_dict(state_dict=best_checkpoint.state_dict)
    return model


def _define_step_metrics(*, run: Run) -> None:
    run.define_metric(name=STEP_METRIC_NAME)
    for name in (EPOCH_METRIC_NAME, f"{TRAINING_SECTION}/*", f"{VALIDATION_SECTION}/*"):
        run.define_metric(name=name, step_metric=STEP_METRIC_NAME)
