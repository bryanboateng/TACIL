from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from wandb import Run

from scripts.data.augmentation.tacil.create_dynamics.config import DynamicsConfig
from scripts.data.augmentation.tacil.create_dynamics.data import (
    Transitions,
    build_windows,
    concatenate_transitions,
    extract_transitions,
)
from scripts.data.augmentation.tacil.create_dynamics.dynamics_model_training import (
    MemberTrainingState,
    TrainingContext,
    create_member_training_state,
    train_member_for_one_epoch,
)
from scripts.data.augmentation.tacil.create_dynamics.statistics.wandb_logging import (
    define_epoch_metrics,
    log_best_checkpoint_summary,
    log_ensemble_statistics,
    log_epoch,
    log_window_counts,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model.dynamics_model import (
    DynamicsModel,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.environments.state_space import StateSpace
from scripts.environments.tacil_dynamics_plotter import TacilDynamicsPlotter
from scripts.environments.tacil_network_input import TacilNetworkInputBuilder
from scripts.shared.episodes import StateActionEpisodes
from scripts.shared.normalization import ZScoreNormalizer
from scripts.shared.seeding import set_seed


def train_dynamics_model_ensemble(
    *,
    config: DynamicsConfig,
    state_space: StateSpace,
    network_input_builder: TacilNetworkInputBuilder,
    dynamics_plotter: TacilDynamicsPlotter,
    episodes: StateActionEpisodes,
    seed: int,
    device: torch.device,
    run: Run,
) -> DynamicsModelEnsemble:
    set_seed(seed=seed)
    define_epoch_metrics(run=run)

    training_transitions_per_episode, validation_transitions_per_episode = (
        train_test_split(
            extract_transitions(episodes=episodes),
            test_size=0.2,
            random_state=seed,
            shuffle=True,
        )
    )

    training_transitions = concatenate_transitions(
        transitions_per_episode=training_transitions_per_episode
    )
    validation_transitions = concatenate_transitions(
        transitions_per_episode=validation_transitions_per_episode
    )

    (
        validation_state_windows,  # (window_count, window_length + 1, state_dimensionality)
        validation_action_windows,  # (window_count, window_length, action_dimensionality)
    ) = build_windows(
        transitions_per_episode=validation_transitions_per_episode,
        window_length=config.window_length,
    )

    normalizers = _create_normalizers(
        transitions=training_transitions,
        state_space=state_space,
        network_input_builder=network_input_builder,
        device=device,
    )

    training_context = TrainingContext(
        config=config,
        training_transitions=training_transitions,
        validation_transitions=validation_transitions,
        validation_state_windows=torch.from_numpy(validation_state_windows).to(
            device=device
        ),
        validation_action_windows=torch.from_numpy(validation_action_windows).to(
            device=device
        ),
    )

    member_training_states = []
    member_window_counts: list[int] = []
    for member_seed in np.random.SeedSequence(entropy=seed).generate_state(
        n_words=config.ensemble_size
    ):
        episode_indices = np.random.default_rng(seed=int(member_seed)).integers(
            low=0,
            high=len(training_transitions_per_episode),
            size=len(training_transitions_per_episode),
        )  # (episode_count,)
        (
            member_state_windows,  # (window_count, window_length + 1, state_dimensionality)
            member_action_windows,  # (window_count, window_length, action_dimensionality)
        ) = build_windows(
            transitions_per_episode=[
                training_transitions_per_episode[index] for index in episode_indices
            ],
            window_length=config.window_length,
        )
        member_window_counts.append(member_action_windows.shape[0])

        member_training_states.append(
            create_member_training_state(
                member_seed=int(member_seed),
                config=config,
                state_space=state_space,
                network_input_builder=network_input_builder,
                network_input_normalizer=normalizers.network_input,
                state_displacement_normalizer=normalizers.state_displacement,
                state_windows=torch.from_numpy(member_state_windows).to(device=device),
                action_windows=torch.from_numpy(member_action_windows).to(
                    device=device
                ),
            )
        )
    log_window_counts(
        member_training_window_counts=member_window_counts,
        validation_window_count=validation_action_windows.shape[0],
        run=run,
    )

    for epoch_index in tqdm(
        range(config.maximum_epoch_count), desc="training ensemble", unit="epoch"
    ):
        still_training = [
            member_training_state
            for member_training_state in member_training_states
            if member_training_state.epoch_count_without_improvement
            < config.patience_epoch_count
        ]
        if not still_training:
            break

        log_epoch(
            epoch_index=epoch_index,
            per_member_metrics=[
                train_member_for_one_epoch(
                    training_state=member_training_state,
                    epoch_index=epoch_index,
                    context=training_context,
                )
                for member_training_state in still_training
            ],
            run=run,
        )

    log_best_checkpoint_summary(member_states=member_training_states, run=run)

    dynamics_model_ensemble = DynamicsModelEnsemble(
        state_space=state_space,
        state_normalizer=normalizers.state,
        dynamics_models=[
            _load_best_member_model(member_training_state=member_training_state)
            for member_training_state in member_training_states
        ],
    )
    dynamics_model_ensemble.eval()

    with torch.no_grad():
        log_ensemble_statistics(
            dynamics_model_ensemble=dynamics_model_ensemble,
            dynamics_plotter=dynamics_plotter,
            validation_transitions_per_episode=validation_transitions_per_episode,
            episodes=episodes,
            run=run,
        )

    return dynamics_model_ensemble


# ---- Private --------------------------------------------


@dataclass
class _Normalizers:
    state: ZScoreNormalizer
    network_input: ZScoreNormalizer
    state_displacement: ZScoreNormalizer


def _create_normalizers(
    *,
    transitions: Transitions,
    state_space: StateSpace,
    network_input_builder: TacilNetworkInputBuilder,
    device: torch.device,
) -> _Normalizers:
    state_batch = torch.from_numpy(transitions.states).to(
        device=device
    )  # (transition_count, state_dimensionality)
    next_state_batch = torch.from_numpy(transitions.next_states).to(
        device=device
    )  # (transition_count, state_dimensionality)
    action_batch = torch.from_numpy(transitions.actions).to(
        device=device
    )  # (transition_count, action_dimensionality)
    return _Normalizers(
        state=ZScoreNormalizer(value_batch=state_batch).to(device=device),
        network_input=ZScoreNormalizer(
            value_batch=network_input_builder(
                state_batch=state_batch, action_batch=action_batch
            )
        ).to(device=device),
        state_displacement=ZScoreNormalizer(
            value_batch=state_space.subtract(
                minuend_batch=next_state_batch, subtrahend_batch=state_batch
            )
        ).to(device=device),
    )


def _load_best_member_model(
    *, member_training_state: MemberTrainingState
) -> DynamicsModel:
    if member_training_state.best_checkpoint is None:
        raise ValueError("No checkpoint was recorded")
    member_training_state.model.load_state_dict(
        state_dict=member_training_state.best_checkpoint.state_dict
    )
    member_training_state.model.eval()
    return member_training_state.model
