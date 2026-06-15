from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from scripts.data.augmentation.tacil.create_data.backward_rollout import (
    BackwardTrajectories,
)
from scripts.shared.data import EpisodeStore


@dataclass(frozen=True)
class AugmentationResult:
    @dataclass(frozen=True)
    class OutcomeCounts:
        not_converged: int
        rejected: int
        accepted: int

    episode_store: EpisodeStore

    error_bounds: npt.NDArray[
        np.float32
    ]  # per-step normalized bounds over converged steps
    error_bound_threshold: float
    counts: OutcomeCounts


def get_anchor_indices(
    *,
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
    expert_part_length: int,
) -> npt.NDArray[np.int64]:  # (anchor_count,)
    mask = np.ones(
        shape=episode_end_indices[-1], dtype=bool
    )  # (total_transition_count,)
    mask[episode_end_indices - expert_part_length] = False
    return np.flatnonzero(mask).astype(episode_end_indices.dtype, copy=False)


@torch.no_grad()
def create_augmented_episodes(
    *,
    backward_trajectories: BackwardTrajectories,
    episode_store: EpisodeStore,
    anchor_indices: npt.NDArray[np.int64],  # (anchor_count,)
    expert_part_length: int,
    label_error_bound_quantile: float,
) -> AugmentationResult:

    converged_error_bounds = backward_trajectories.error_bounds_batch[
        backward_trajectories.anchor_alive_batch
    ].flatten()  # (alive_count * step,)
    if converged_error_bounds.numel() == 0:
        raise RuntimeError("No backward steps converged. ")

    error_bound_threshold = float(
        torch.quantile(
            input=converged_error_bounds, q=label_error_bound_quantile
        ).item()
    )

    episode_accepted_batch = backward_trajectories.anchor_alive_batch & (
        backward_trajectories.error_bounds_batch <= error_bound_threshold
    ).all(dim=1)  # (anchor_count,)

    augmented_episode_store = _assemble_episodes(
        states=episode_store.states,
        actions=episode_store.actions,
        predecessor_states=backward_trajectories.predecessor_states_batch.cpu().numpy(),
        predecessor_actions=backward_trajectories.predecessor_actions_batch.cpu().numpy(),
        episode_accepted=episode_accepted_batch.cpu().numpy(),
        anchor_positions=np.asarray(anchor_indices, dtype=np.int64),
        expert_part_length=expert_part_length,
    )

    anchor_count = backward_trajectories.predecessor_actions_batch.shape[0]
    accepted_count = int(episode_accepted_batch.sum().item())
    not_converged_count = int((~backward_trajectories.anchor_alive_batch).sum().item())
    rejected_count = anchor_count - accepted_count - not_converged_count

    return AugmentationResult(
        episode_store=augmented_episode_store,
        error_bounds=converged_error_bounds.cpu().numpy(),
        error_bound_threshold=error_bound_threshold,
        counts=AugmentationResult.OutcomeCounts(
            not_converged=not_converged_count,
            rejected=rejected_count,
            accepted=accepted_count,
        ),
    )


def _assemble_episodes(
    *,
    states: npt.NDArray[np.float32],  # (total_transition_count, state_dimensionality)
    actions: npt.NDArray[np.float32],  # (total_transition_count, action_dimensionality)
    predecessor_states: npt.NDArray[
        np.float32
    ],  # (anchor_count, synthetic_part_length, state_dimensionality)
    predecessor_actions: npt.NDArray[
        np.float32
    ],  # (anchor_count, synthetic_part_length, action_dimensionality)
    episode_accepted: npt.NDArray[np.bool_],  # (anchor_count,)
    anchor_positions: npt.NDArray[np.int64],  # (anchor_count,)
    expert_part_length: int,
) -> EpisodeStore:
    synthetic_part_length = predecessor_states.shape[1]

    episode_state_chunks: list[np.ndarray] = []
    episode_action_chunks: list[np.ndarray] = []
    episode_lengths: list[int] = []

    for anchor_index in np.nonzero(episode_accepted)[0]:
        anchor_position: int = anchor_positions[anchor_index]

        # Synthetic prefix in temporal order = predecessors reversed.
        synthetic_states = predecessor_states[anchor_index][
            ::-1
        ]  # (synthetic_part_length, state_dimensionality)
        synthetic_actions = predecessor_actions[anchor_index][
            ::-1
        ]  # (synthetic_part_length, action_dimensionality)

        expert_part_slice = slice(anchor_position, anchor_position + expert_part_length)

        episode_state_chunks.append(
            np.concatenate(
                [
                    synthetic_states,
                    states[
                        expert_part_slice
                    ],  # (expert_part_length, state_dimensionality)
                ],
                axis=0,
            )
        )
        episode_action_chunks.append(
            np.concatenate(
                [
                    synthetic_actions,
                    actions[
                        expert_part_slice
                    ],  # (expert_part_length, action_dimensionality)
                ],
                axis=0,
            )
        )
        episode_lengths.append(synthetic_part_length + expert_part_length)

    return EpisodeStore(
        states=np.concatenate(
            episode_state_chunks, axis=0
        ),  # (total_timestep_count, state_dimensionality)
        actions=np.concatenate(
            episode_action_chunks, axis=0
        ),  # (total_timestep_count, action_dimensionality)
        episode_end_indices=np.cumsum(
            np.asarray(episode_lengths, dtype=np.int64)
        ),  # (episode_count,)
    )


def combine_episode_stores(
    *, expert_episode_store: EpisodeStore, augmentation_episode_store: EpisodeStore
) -> EpisodeStore:
    states = np.concatenate(
        [expert_episode_store.states, augmentation_episode_store.states], axis=0
    )  # (total_timestep_count, state_dimensionality)

    actions = np.concatenate(
        [expert_episode_store.actions, augmentation_episode_store.actions], axis=0
    )  # (total_timestep_count, action_dimensionality)

    original_timestep_count = expert_episode_store.states.shape[0]
    augmented_episode_ends_absolute = (
        original_timestep_count + augmentation_episode_store.episode_end_indices
    )  # (augmented_episode_count,)

    episode_end_indices = np.concatenate(
        [expert_episode_store.episode_end_indices, augmented_episode_ends_absolute],
        axis=0,
    )  # (episode_count,)

    return EpisodeStore(
        states=states, actions=actions, episode_end_indices=episode_end_indices
    )
