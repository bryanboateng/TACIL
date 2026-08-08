import numpy as np
import torch
from wandb import Run

from scripts.data.augmentation.ccil.config import Config
from scripts.data.augmentation.ccil.f import TransitionBatch
from scripts.data.augmentation.ccil.world_model import WorldModel, fit_world_model
from scripts.data.base.episode_store import EpisodeStore
from scripts.shared.log import create_logger
from scripts.shared.seed import set_seed

logger = create_logger()

STAGE_PREFIX = "2_data-"

_BATCH_SIZE = 4_096


def create_ccil_augmented_data(
    *, config: Config, episode_store: EpisodeStore, device: torch.device, run: Run
) -> TransitionBatch:
    """The accepted labels alone. A label is a single (state, action) pair, so
    the caller appends them to the expert transitions."""
    set_seed(seed=config.seed)

    states, actions, next_states = _transitions(episode_store=episode_store)
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
    for epoch_index, losses in enumerate(iterable=epoch_losses):
        run.log(
            data={
                "1_dynamics-loss/total": losses.total,
                "1_dynamics-loss/mse": losses.mse,
                "1_dynamics-loss/lipschitz_penalty": losses.lipschitz_penalty,
                "1_dynamics-epoch": epoch_index,
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
        np.quantile(a=label_distances, q=config.augmentation.rejection_quantile)
    )
    accepted_mask = label_distances <= distance_threshold
    accepted_count = int(accepted_mask.sum())
    label_count = label_states.shape[0]
    run.summary[f"{STAGE_PREFIX}2_labels/distance_threshold"] = distance_threshold

    run.summary[f"{STAGE_PREFIX}2_labels/transition_count"] = states.shape[0]
    run.summary[f"{STAGE_PREFIX}2_labels/label_count"] = label_count
    run.summary[f"{STAGE_PREFIX}2_labels/accepted_count"] = accepted_count
    run.summary[f"{STAGE_PREFIX}2_labels/accepted_fraction"] = accepted_count / max(
        label_count, 1
    )
    run.summary[f"{STAGE_PREFIX}2_labels/distance_median"] = float(
        np.median(label_distances)
    )

    return TransitionBatch(
        states=label_states[accepted_mask], actions=label_actions[accepted_mask]
    )


# ---- Private --------------------------------------------


def _transitions(
    *, episode_store: EpisodeStore
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All (s, a, s') triples; the last timestep of every episode has no
    successor and is dropped."""
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
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Their generators, batched: labels, their actions, and the distances
    ‖s^G − s‖ used by the rejection filter.

    noisy_action (gen_noisy_action_be): s^G = s' − f(s', ã) with noised ã.
    backward_euler_fast: s^G = s − f(s, a) with the true action (Eq. 4).
    """
    if config.augmentation.type == "noisy_action":
        label_count_per_transition = config.augmentation.label_count_per_transition
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

        if config.augmentation.type == "noisy_action":
            repeated_next_states = next_state_batch[batch_slice].repeat_interleave(
                repeats=label_count_per_transition, dim=0
            )
            label_actions = repeated_actions + torch.normal(
                mean=0.0,
                std=config.augmentation.action_noise_std,
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
