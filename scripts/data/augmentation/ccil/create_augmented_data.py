import numpy as np
import torch
from wandb import Run

from scripts.data.augmentation.ccil.config import CcilConfig
from scripts.data.augmentation.ccil.world_model import WorldModel, fit_world_model
from scripts.shared.distribution_panel import create_distribution_panel
from scripts.shared.distribution_summary import summarize_distribution
from scripts.shared import stage_names
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.log import create_logger
from scripts.shared.seeding import set_seed
from scripts.shared.state_action_pairs import StateActionPairs

logger = create_logger()

STAGE_NAME = stage_names.DATA
DYNAMICS_STAGE_NAME = stage_names.DYNAMICS

EPOCH_METRIC_NAME = f"{DYNAMICS_STAGE_NAME}/0_epoch"

LOSS_SECTION = f"{DYNAMICS_STAGE_NAME}/1_loss"

_BATCH_SIZE = 4_096


def create_ccil_augmented_data(
    *,
    config: CcilConfig,
    expert_episodes: EpisodeStore,
    seed: int,
    device: torch.device,
    run: Run,
) -> StateActionPairs:
    set_seed(seed=seed)

    states, actions, next_states = _extract_transitions(episode_store=expert_episodes)
    state_batch = torch.from_numpy(states).to(device=device)
    action_batch = torch.from_numpy(actions).to(device=device)
    next_state_batch = torch.from_numpy(next_states).to(device=device)

    world_model = WorldModel(
        state_dimensionality=states.shape[1],
        action_dimensionality=actions.shape[1],
        hidden_layer_sizes=config.dynamics.hidden_layer_sizes,
        lipschitz_type=config.dynamics.lipschitz_type,
        lipschitz_constraint=config.dynamics.lipschitz_constraint,
    ).to(device=device)

    epoch_losses = fit_world_model(
        world_model=world_model,
        config=config.dynamics,
        state_batch=state_batch,
        action_batch=action_batch,
        next_state_batch=next_state_batch,
    )
    _define_epoch_metrics(run=run)
    for epoch_index, losses in enumerate(iterable=epoch_losses):
        run.log(
            data={
                f"{LOSS_SECTION}/1_total": losses.total,
                f"{LOSS_SECTION}/2_mse": losses.mse,
                f"{LOSS_SECTION}/3_lipschitz_penalty": losses.lipschitz_penalty,
                EPOCH_METRIC_NAME: epoch_index,
            }
        )

    label_states, label_actions, label_distances = _generate_labels(
        world_model=world_model,
        state_batch=state_batch,
        action_batch=action_batch,
        next_state_batch=next_state_batch,
        config=config,
    )

    distance_threshold = float(
        np.quantile(a=label_distances, q=config.labels.rejection_quantile)
    )
    accepted_mask = label_distances <= distance_threshold
    label_count = label_states.shape[0]
    rejected_count = label_count - int(accepted_mask.sum())

    run.summary[f"{STAGE_NAME}/1_labels/1_expert_transition_count"] = states.shape[0]
    run.summary[f"{STAGE_NAME}/1_labels/2_label_count"] = label_count
    run.summary[f"{STAGE_NAME}/1_labels/3_distance_threshold"] = distance_threshold
    # A rejected label sits further from its expert state than the threshold
    # allows. The denominator is the label_count logged next to it.
    run.summary[f"{STAGE_NAME}/1_labels/4_rejected_label_count"] = rejected_count
    run.summary[f"{STAGE_NAME}/1_labels/5_rejected_label_fraction"] = (
        rejected_count / max(label_count, 1)
    )

    for label, value in summarize_distribution(values=label_distances).items():
        run.summary[f"{STAGE_NAME}/2_label_distances/{label}"] = value
    run.log(
        data={
            f"{STAGE_NAME}/2_label_distances/4_plot": create_distribution_panel(
                values=label_distances, column_name="label distance"
            )
        }
    )

    accepted_states = label_states[accepted_mask]
    run.summary[f"{STAGE_NAME}/3_dataset/1_synthetic_transition_count"] = int(
        accepted_states.shape[0]
    )
    return StateActionPairs(
        states=accepted_states, actions=label_actions[accepted_mask]
    )


def _define_epoch_metrics(*, run: Run) -> None:
    run.define_metric(name=EPOCH_METRIC_NAME)
    run.define_metric(name=f"{LOSS_SECTION}/*", step_metric=EPOCH_METRIC_NAME)


# ---- Private --------------------------------------------


def _extract_transitions(
    *, episode_store: EpisodeStore
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    last_timestep_indices = episode_store.episode_end_indices - 1
    keep_mask = np.ones(episode_store.states.shape[0], dtype=bool)
    keep_mask[last_timestep_indices] = False

    successor_indices = np.nonzero(keep_mask)[0] + 1
    return (
        episode_store.states[keep_mask],
        episode_store.actions[keep_mask],
        episode_store.states[successor_indices],
    )


@torch.no_grad()
def _generate_labels(
    *,
    world_model: WorldModel,
    state_batch: torch.Tensor,  # (transition_count, state_dimensionality)
    action_batch: torch.Tensor,  # (transition_count, action_dimensionality)
    next_state_batch: torch.Tensor,  # (transition_count, state_dimensionality)
    config: CcilConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if config.labels.type == "noisy_action":
        label_count_per_transition = config.labels.label_count_per_transition
    else:
        label_count_per_transition = 1

    label_state_chunks: list[np.ndarray] = []
    label_action_chunks: list[np.ndarray] = []
    distance_chunks: list[np.ndarray] = []

    for batch_start_index in range(0, state_batch.shape[0], _BATCH_SIZE):
        batch_slice = slice(batch_start_index, batch_start_index + _BATCH_SIZE)
        repeated_states = state_batch[batch_slice].repeat_interleave(
            repeats=label_count_per_transition, dim=0
        )
        repeated_actions = action_batch[batch_slice].repeat_interleave(
            repeats=label_count_per_transition, dim=0
        )

        if config.labels.type == "noisy_action":
            repeated_next_states = next_state_batch[batch_slice].repeat_interleave(
                repeats=label_count_per_transition, dim=0
            )
            label_actions = repeated_actions + torch.normal(
                mean=0.0,
                std=config.labels.action_noise_std,
                size=repeated_actions.shape,
                device=repeated_actions.device,
            )
            label_states = repeated_next_states - world_model.predict_residual(
                state_batch=repeated_next_states, action_batch=label_actions
            )
        else:
            label_actions = repeated_actions
            label_states = repeated_states - world_model.predict_residual(
                state_batch=repeated_states, action_batch=label_actions
            )

        distances = torch.linalg.norm(label_states - repeated_states, dim=-1)

        label_state_chunks.append(label_states.cpu().numpy())
        label_action_chunks.append(label_actions.cpu().numpy())
        distance_chunks.append(distances.cpu().numpy())

    return (
        np.concatenate(label_state_chunks).astype(np.float32),
        np.concatenate(label_action_chunks).astype(np.float32),
        np.concatenate(distance_chunks),
    )
