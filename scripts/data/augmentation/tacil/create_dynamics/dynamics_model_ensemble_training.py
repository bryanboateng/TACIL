from dataclasses import dataclass

import numpy as np
import torch
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
from scripts.data.augmentation.tacil.create_dynamics.dynamics_model_training import (
    TrainingContext,
    create_member_training_state,
    load_best_member_model,
    train_member_for_one_epoch,
)
from scripts.data.augmentation.tacil.create_dynamics.statistics.logging import (
    define_epoch_metrics,
    log_checkpoint_summary,
    log_ensemble_statistics,
    log_epoch,
    log_window_counts,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.data.augmentation.tacil.dynamics.network_input import (
    create_raw_network_input,
)
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.log import create_logger
from scripts.shared.normalization import ZScoreNormalizer
from scripts.shared.pusht_state import subtract_states
from scripts.shared.seeding import set_seed

logger = create_logger()


def train_dynamics_model_ensemble(
    *,
    config: DynamicsConfig,
    episode_store: EpisodeStore,
    seed: int,
    device: torch.device,
    run: Run,
) -> DynamicsModelEnsemble:
    set_seed(seed=seed)
    define_epoch_metrics(run=run)

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
    log_window_counts(
        member_training_window_counts=member_window_counts,
        validation_window_count=validation_action_windows.shape[0],
        run=run,
    )

    for epoch_index in tqdm(range(500), desc="training ensemble", unit="epoch"):
        log_epoch(
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

    log_checkpoint_summary(member_states=member_training_states, run=run)

    dynamics_model_ensemble = DynamicsModelEnsemble(
        state_normalizer=normalizers.state,
        dynamics_models=[
            load_best_member_model(member_training_state=member_training_state)
            for member_training_state in member_training_states
        ],
    )
    dynamics_model_ensemble.eval()

    log_ensemble_statistics(
        dynamics_model_ensemble=dynamics_model_ensemble,
        training_episodes=training_episodes,
        validation_episodes=validation_episodes,
        episode_store=episode_store,
        run=run,
    )

    return dynamics_model_ensemble


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
            value_batch=create_raw_network_input(
                state_batch=state_batch, action_batch=action_batch
            )
        ).to(device=device),
        state_displacement=ZScoreNormalizer(
            value_batch=subtract_states(
                minuend_batch=next_state_batch, subtrahend_batch=state_batch
            )
        ).to(device=device),
    )
