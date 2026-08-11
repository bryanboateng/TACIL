import numpy as np
import torch
import wandb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from wandb import Run

from scripts.data.augmentation.tacil.create_dynamics.config import DynamicsConfig
from scripts.data.augmentation.tacil.create_dynamics.data import (
    TransitionSet,
    build_windows,
    concatenate_episodes,
    extract_transitions,
)
from scripts.data.augmentation.tacil.create_dynamics.diagnostics import (
    diagnose_dynamics_model_ensemble,
)
from scripts.data.augmentation.tacil.create_dynamics.dynamics_model_training import (
    MemberEpochMetrics,
    MemberTrainingState,
    TrainingContext,
    create_member_training_state,
    load_best_member_model,
    train_member_for_one_epoch,
)
from scripts.data.augmentation.tacil.models.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.data.augmentation.tacil.models.network_input import network_features
from scripts.data.augmentation.tacil.models.normalizers import Normalizers
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.log import create_logger
from scripts.shared.pusht_state import subtract_states
from scripts.shared.seeding import set_seed

logger = create_logger()

STAGE_NAME = "1_dynamics"

EPOCH_METRIC_NAME = f"{STAGE_NAME}/epoch"


def train_dynamics_model_ensemble(
    *,
    config: DynamicsConfig,
    episode_store: EpisodeStore,
    seed: int,
    device: torch.device,
    run: Run,
) -> DynamicsModelEnsemble:
    set_seed(seed=seed)
    _define_epoch_metrics(run=run)

    training_episodes, validation_episodes = train_test_split(
        extract_transitions(episode_store=episode_store),
        test_size=0.2,
        random_state=0,
        shuffle=True,
    )

    training_transitions = concatenate_episodes(episodes=training_episodes)
    validation_transitions = concatenate_episodes(episodes=validation_episodes)

    train_state_windows, train_action_windows = build_windows(
        episodes=training_episodes, window_length=config.window_length
    )
    validation_state_windows, validation_action_windows = build_windows(
        episodes=validation_episodes, window_length=config.window_length
    )
    run.summary[f"{STAGE_NAME}/4_window_count/training"] = train_action_windows.shape[0]
    run.summary[f"{STAGE_NAME}/4_window_count/validation"] = (
        validation_action_windows.shape[0]
    )

    normalizers = _create_normalizers(transitions=training_transitions, device=device)

    training_context = TrainingContext(
        config=config,
        normalizers=normalizers,
        training_transitions=training_transitions,
        validation_transitions=validation_transitions,
        train_state_windows=torch.from_numpy(train_state_windows).to(device=device),
        train_action_windows=torch.from_numpy(train_action_windows).to(device=device),
        validation_state_windows=validation_state_windows,
        validation_action_windows=validation_action_windows,
    )

    member_training_states = [
        create_member_training_state(
            member_seed=int(member_seed),
            config=config,
            transitions=training_transitions,
            device=device,
        )
        for member_seed in np.random.SeedSequence(entropy=seed).generate_state(
            n_words=config.ensemble_size
        )
    ]

    for epoch_index in tqdm(range(500), desc="training ensemble", unit="epoch"):
        _log_epoch(
            epoch_index=epoch_index,
            per_member_metrics=[
                train_member_for_one_epoch(
                    state=member_training_state,
                    epoch_index=epoch_index,
                    context=training_context,
                )
                for member_training_state in member_training_states
            ],
            run=run,
        )

    _log_checkpoint_summary(member_states=member_training_states, run=run)

    dynamics_model_ensemble = DynamicsModelEnsemble(
        normalizers=normalizers,
        dynamics_models=[
            load_best_member_model(member_training_state=member_training_state)
            for member_training_state in member_training_states
        ],
    )
    dynamics_model_ensemble.eval()

    diagnose_dynamics_model_ensemble(
        dynamics_model_ensemble=dynamics_model_ensemble,
        training_episodes=training_episodes,
        validation_episodes=validation_episodes,
        episode_store=episode_store,
        run=run,
    )

    return dynamics_model_ensemble


def _define_epoch_metrics(*, run: Run) -> None:
    run.define_metric(name=EPOCH_METRIC_NAME)
    for metric_pattern in ("1_loss/*", "2_gradient_norm/*", "8_lipschitz/*"):
        run.define_metric(
            name=f"{STAGE_NAME}/{metric_pattern}", step_metric=EPOCH_METRIC_NAME
        )


def _create_normalizers(
    transitions: TransitionSet, device: torch.device
) -> Normalizers:
    state_batch = torch.from_numpy(transitions.states).to(device=device)
    next_state_batch = torch.from_numpy(transitions.next_states).to(device=device)
    action_batch = torch.from_numpy(transitions.actions).to(device=device)
    normalizers = Normalizers.from_transitions(
        state_batch=state_batch,
        network_feature_batch=network_features(
            state_batch=state_batch, action_batch=action_batch
        ),
        state_displacement_batch=subtract_states(
            minuend_batch=next_state_batch, subtrahend_batch=state_batch
        ),
    ).to(device=device)
    return normalizers


def _log_epoch(
    *, epoch_index: int, per_member_metrics: list[MemberEpochMetrics], run: Run
) -> None:
    logged_metrics: dict[str, float | wandb.Histogram] = {
        EPOCH_METRIC_NAME: epoch_index
    }
    for member_index, metrics in enumerate(iterable=per_member_metrics):
        member = f"member_{member_index:02d}"
        logged_metrics[f"{STAGE_NAME}/1_loss/training/{member}"] = metrics.training_loss
        logged_metrics[f"{STAGE_NAME}/1_loss/validation/{member}"] = (
            metrics.validation_loss
        )
        logged_metrics[f"{STAGE_NAME}/2_gradient_norm/{member}"] = metrics.gradient_norm

    for split, member_split_lipschitz in (
        (
            "training",
            [
                metrics.training_local_lipschitz_constants
                for metrics in per_member_metrics
            ],
        ),
        (
            "validation",
            [
                metrics.validation_local_lipschitz_constants
                for metrics in per_member_metrics
            ],
        ),
    ):
        logged_metrics[f"{STAGE_NAME}/8_lipschitz/{split}"] = wandb.Histogram(
            sequence=np.concatenate(member_split_lipschitz).tolist()
        )

    run.log(data=logged_metrics)


def _log_checkpoint_summary(
    *, member_states: list[MemberTrainingState], run: Run
) -> None:
    checkpoints = []
    for member_state in member_states:
        if member_state.best_checkpoint is None:
            raise ValueError("No checkpoint was recorded")
        checkpoints.append(member_state.best_checkpoint)

    validation_losses = np.array(
        [checkpoint.validation_loss for checkpoint in checkpoints]
    )
    epochs = np.array([checkpoint.epoch for checkpoint in checkpoints])

    run.summary[f"{STAGE_NAME}/3_selected/loss_mean"] = float(validation_losses.mean())
    run.summary[f"{STAGE_NAME}/3_selected/epoch_mean"] = float(epochs.mean())
    # The spread over the members. It shows how much the result depends on the
    # initialization of one member.
    run.summary[f"{STAGE_NAME}/3_selected/loss_std"] = _member_spread(
        values=validation_losses
    )
    run.summary[f"{STAGE_NAME}/3_selected/epoch_std"] = _member_spread(values=epochs)


def _member_spread(*, values: np.ndarray) -> float:
    # ddof=1, because the members are a sample of the possible initializations.
    # A single member has no spread.
    return float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0
