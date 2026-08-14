from dataclasses import dataclass

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
from scripts.data.augmentation.tacil.models import network_input
from scripts.data.augmentation.tacil.models.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.data.augmentation.tacil.models.network_input import raw_network_input
from scripts.shared.distribution_summary import summarize_distribution
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.log import create_logger
from scripts.shared.normalization import ZScoreNormalizer
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
        random_state=seed,
        shuffle=True,
    )

    training_transitions = concatenate_episodes(episodes=training_episodes)
    validation_transitions = concatenate_episodes(episodes=validation_episodes)

    validation_state_windows, validation_action_windows = build_windows(
        episodes=validation_episodes, window_length=config.window_length
    )
    run.summary[f"{STAGE_NAME}/4_window_count/validation"] = (
        validation_action_windows.shape[0]
    )

    normalizers = _create_normalizers(transitions=training_transitions, device=device)

    training_context = TrainingContext(
        config=config,
        training_transitions=training_transitions,
        validation_transitions=validation_transitions,
        validation_state_windows=validation_state_windows,
        validation_action_windows=validation_action_windows,
    )

    member_training_states = []
    member_window_counts: list[int] = []
    for member_seed in np.random.SeedSequence(entropy=seed).generate_state(
        n_words=config.ensemble_size
    ):
        # Bootstrap: each member trains on its own resample of the training
        # episodes, drawn with replacement and equally large. Every member
        # sees a full-size dataset, but each one has different gaps.
        episode_indices = np.random.default_rng(seed=int(member_seed)).integers(
            low=0, high=len(training_episodes), size=len(training_episodes)
        )
        member_state_windows, member_action_windows = build_windows(
            episodes=[training_episodes[index] for index in episode_indices],
            window_length=config.window_length,
        )
        member_window_counts.append(member_action_windows.shape[0])

        member_training_states.append(
            create_member_training_state(
                member_seed=int(member_seed),
                config=config,
                transitions=training_transitions,
                network_input_normalizer=normalizers.network_input,
                state_displacement_normalizer=normalizers.state_displacement,
                train_state_windows=torch.from_numpy(member_state_windows).to(
                    device=device
                ),
                train_action_windows=torch.from_numpy(member_action_windows).to(
                    device=device
                ),
                device=device,
            )
        )
    run.summary[f"{STAGE_NAME}/4_window_count/training"] = float(
        np.mean(member_window_counts)
    )

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
        state_normalizer=normalizers.state,
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


@dataclass
class _Normalizers:
    state: ZScoreNormalizer
    network_input: ZScoreNormalizer
    state_displacement: ZScoreNormalizer


def _create_normalizers(
    *, transitions: TransitionSet, device: torch.device
) -> _Normalizers:
    state_batch = torch.from_numpy(transitions.states).to(device=device)
    next_state_batch = torch.from_numpy(transitions.next_states).to(device=device)
    action_batch = torch.from_numpy(transitions.actions).to(device=device)
    return _Normalizers(
        state=ZScoreNormalizer(value_batch=state_batch).to(device=device),
        network_input=ZScoreNormalizer(
            value_batch=raw_network_input(
                state_batch=state_batch, action_batch=action_batch
            )
        ).to(device=device),
        state_displacement=ZScoreNormalizer(
            value_batch=subtract_states(
                minuend_batch=next_state_batch, subtrahend_batch=state_batch
            )
        ).to(device=device),
    )


def _log_epoch(
    *, epoch_index: int, per_member_metrics: list[MemberEpochMetrics], run: Run
) -> None:
    logged_metrics: dict[str, float | wandb.Histogram] = {
        EPOCH_METRIC_NAME: epoch_index
    }
    # Summarized over the members rather than one curve each: twenty curves per
    # quantity are unreadable, and what the curves are watched for — does it
    # come down, do the members stay together — the summary says too.
    for name, member_values in (
        ("1_loss/training", [m.training_loss for m in per_member_metrics]),
        ("1_loss/validation", [m.validation_loss for m in per_member_metrics]),
        ("2_gradient_norm", [m.gradient_norm for m in per_member_metrics]),
    ):
        summary = summarize_distribution(
            values=np.asarray(member_values, dtype=np.float64)
        )
        for label, value in summary.items():
            logged_metrics[f"{STAGE_NAME}/{name}/{label}"] = value

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

    # Each member keeps the epoch with its lowest validation loss, and this is
    # what those kept epochs look like across the members. The spread shows how
    # much the result depends on the initialization of a single one.
    for name, values in (("validation_loss", validation_losses), ("epoch", epochs)):
        for label, value in summarize_distribution(values=values).items():
            run.summary[f"{STAGE_NAME}/3_best_checkpoint/{name}_{label}"] = value
