import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from wandb import Run

from scripts.data.augmentation.tacil.create_dynamics.config import DynamicsConfig
from scripts.data.augmentation.tacil.create_dynamics.data import (
    build_windows,
    concatenate_transitions,
    extract_transitions,
)
from scripts.data.augmentation.tacil.create_dynamics.dynamics_model_training import (
    TrainingContext,
    create_training_state,
    train_for_one_epoch,
)
from scripts.data.augmentation.tacil.create_dynamics.statistics.wandb_logging import (
    define_epoch_metrics,
    log_best_checkpoint_summary,
    log_ensemble_statistics,
    log_epoch,
    log_window_counts,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
    select_members,
)
from scripts.data.augmentation.tacil.dynamics.normalizers import create_normalizers
from scripts.environments.state_space import StateSpace
from scripts.environments.tacil_dynamics_plotter import TacilDynamicsPlotter
from scripts.environments.tacil_network_input import TacilNetworkInputBuilder
from scripts.shared.episodes import Episodes
from scripts.shared.seeding import set_seed


def train_dynamics_model_ensemble(
    *,
    config: DynamicsConfig,
    state_space: StateSpace,
    network_input_builder: TacilNetworkInputBuilder,
    dynamics_plotter: TacilDynamicsPlotter,
    episodes: Episodes,
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

    (
        training_state_windows,  # (window_count, window_length + 1, state_dimensionality)
        training_action_windows,  # (window_count, window_length, action_dimensionality)
    ) = build_windows(
        transitions_per_episode=training_transitions_per_episode,
        window_length=config.window_length,
    )
    (
        validation_state_windows,  # (window_count, window_length + 1, state_dimensionality)
        validation_action_windows,  # (window_count, window_length, action_dimensionality)
    ) = build_windows(
        transitions_per_episode=validation_transitions_per_episode,
        window_length=config.window_length,
    )

    training_context = TrainingContext(
        training_state_windows=torch.from_numpy(training_state_windows).to(
            device=device
        ),
        training_action_windows=torch.from_numpy(training_action_windows).to(
            device=device
        ),
        validation_state_windows=torch.from_numpy(validation_state_windows).to(
            device=device
        ),
        validation_action_windows=torch.from_numpy(validation_action_windows).to(
            device=device
        ),
    )

    training_state = create_training_state(
        model=DynamicsModelEnsemble(
            member_count=config.ensemble_size,
            state_space=state_space,
            network_input_builder=network_input_builder,
            hidden_layer_count=config.hidden_layer_count,
            hidden_layer_dimensionality=config.hidden_layer_dimensionality,
            lipschitz_limit=config.lipschitz_limit,
            normalizers=create_normalizers(
                transitions=concatenate_transitions(
                    transitions_per_episode=training_transitions_per_episode
                ),
                state_space=state_space,
                network_input_builder=network_input_builder,
                device=device,
            ),
        ).to(device=device),
        seed=seed,
        training_window_count=training_action_windows.shape[0],
    )
    log_window_counts(
        training_window_count=training_action_windows.shape[0],
        validation_window_count=validation_action_windows.shape[0],
        run=run,
    )

    for epoch_index in tqdm(
        range(config.maximum_epoch_count), desc="training ensemble", unit="epoch"
    ):
        if (
            epoch_index - training_state.last_improvement_epoch
            > config.patience_epoch_count
        ):
            break

        log_epoch(
            epoch_index=epoch_index,
            metrics=train_for_one_epoch(
                training_state=training_state,
                epoch_index=epoch_index,
                context=training_context,
            ),
            run=run,
        )

    log_best_checkpoint_summary(
        best_validation_losses=training_state.best_validation_losses,
        best_epochs=training_state.best_epochs,
        run=run,
    )

    converged_member_indices = np.flatnonzero(
        np.isfinite(training_state.best_validation_losses)
    )  # (converged_member_count,)
    if converged_member_indices.size == 0:
        raise ValueError("No ensemble member reached a finite validation loss")

    dynamics_model_ensemble = select_members(
        dynamics_model_ensemble=training_state.best_model,
        member_indices=converged_member_indices.tolist(),
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
