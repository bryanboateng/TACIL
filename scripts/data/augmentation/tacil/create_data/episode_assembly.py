from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from scripts.shared.episode_store import EpisodeStore


@dataclass(frozen=True)
class SyntheticTrajectories:
    """Invented chains, stored flat, one slice per anchor.

    Each chain is in temporal order: its first step is the state farthest
    from the anchor, its last action leads (per the model) to the anchor.
    """

    states: torch.Tensor  # (total_step_count, state_dimensionality)
    actions: torch.Tensor  # (total_step_count, action_dimensionality)
    step_count_batch: torch.Tensor  # (anchor_count,)

    @property
    def start_index_batch(self) -> torch.Tensor:  # (anchor_count,)
        return torch.cumsum(input=self.step_count_batch, dim=0) - self.step_count_batch


@dataclass(frozen=True)
class AugmentationResult:
    @dataclass(frozen=True)
    class OutcomeCounts:
        empty: int  # anchors whose first backward step already broke the budget
        accepted: int

    episode_store: EpisodeStore
    synthetic_part_lengths: npt.NDArray[np.int64]  # (accepted_count,)
    counts: OutcomeCounts


@torch.no_grad()
def create_augmented_episodes(
    *,
    synthetic_trajectories: SyntheticTrajectories,
    episode_store: EpisodeStore,
    anchor_indices: npt.NDArray[np.int64],  # (anchor_count,)
    expert_part_length: int,
) -> AugmentationResult:
    synthetic_part_lengths: npt.NDArray[np.int64] = (
        synthetic_trajectories.step_count_batch.cpu().numpy()
    )  # (anchor_count,)
    accepted_anchor_count = int((synthetic_part_lengths > 0).sum())
    empty_anchor_count = int(synthetic_part_lengths.shape[0]) - accepted_anchor_count
    if accepted_anchor_count == 0:
        # A precondition, not a user-facing error: the caller already stops on
        # an empty rollout and reports it. Reaching this means the caller
        # dropped that check.
        raise RuntimeError(
            "create_augmented_episodes needs at least one accepted anchor."
        )

    augmented_episode_store = _assemble_episodes(
        expert_states=episode_store.states,
        expert_actions=episode_store.actions,
        synthetic_states=synthetic_trajectories.states.cpu().numpy(),
        synthetic_actions=synthetic_trajectories.actions.cpu().numpy(),
        synthetic_part_start_index_batch=(
            synthetic_trajectories.start_index_batch.cpu().numpy()
        ),
        synthetic_part_length_batch=synthetic_part_lengths,
        anchor_index_batch=np.asarray(anchor_indices, dtype=np.int64),
        expert_part_length=expert_part_length,
    )

    return AugmentationResult(
        episode_store=augmented_episode_store,
        synthetic_part_lengths=synthetic_part_lengths[synthetic_part_lengths > 0],
        counts=AugmentationResult.OutcomeCounts(
            empty=empty_anchor_count, accepted=accepted_anchor_count
        ),
    )


def _assemble_episodes(
    *,
    expert_states: npt.NDArray[
        np.float32
    ],  # (total_transition_count, state_dimensionality)
    expert_actions: npt.NDArray[
        np.float32
    ],  # (total_transition_count, action_dimensionality)
    synthetic_states: npt.NDArray[
        np.float32
    ],  # (total_step_count, state_dimensionality)
    synthetic_actions: npt.NDArray[
        np.float32
    ],  # (total_step_count, action_dimensionality)
    synthetic_part_start_index_batch: npt.NDArray[np.int64],  # (anchor_count,)
    synthetic_part_length_batch: npt.NDArray[np.int64],  # (anchor_count,)
    anchor_index_batch: npt.NDArray[np.int64],  # (anchor_count,)
    expert_part_length: int,
) -> EpisodeStore:
    accepted_anchor_mask = synthetic_part_length_batch > 0  # (anchor_count,)

    state_trajectories_per_episode: list[np.ndarray] = []
    action_trajectories_per_episode: list[np.ndarray] = []
    episode_lengths: list[int] = []

    for synthetic_part_length, synthetic_part_start_index, anchor_index in zip(
        synthetic_part_length_batch[accepted_anchor_mask],
        synthetic_part_start_index_batch[accepted_anchor_mask],
        anchor_index_batch[accepted_anchor_mask],
    ):
        synthetic_part_slice = slice(
            synthetic_part_start_index,
            synthetic_part_start_index + synthetic_part_length,
        )

        synthetic_state_trajectory = synthetic_states[
            synthetic_part_slice
        ]  # (synthetic_part_length, state_dimensionality)
        synthetic_action_trajectory = synthetic_actions[
            synthetic_part_slice
        ]  # (synthetic_part_length, action_dimensionality)

        expert_part_slice = slice(anchor_index, anchor_index + expert_part_length)

        state_trajectories_per_episode.append(
            np.concatenate(
                [synthetic_state_trajectory, expert_states[expert_part_slice]], axis=0
            )
        )
        action_trajectories_per_episode.append(
            np.concatenate(
                [synthetic_action_trajectory, expert_actions[expert_part_slice]], axis=0
            )
        )
        episode_lengths.append(int(synthetic_part_length) + expert_part_length)

    return EpisodeStore(
        states=np.concatenate(
            state_trajectories_per_episode, axis=0
        ),  # (total_timestep_count, state_dimensionality)
        actions=np.concatenate(
            action_trajectories_per_episode, axis=0
        ),  # (total_timestep_count, action_dimensionality)
        episode_end_indices=np.cumsum(
            np.asarray(episode_lengths, dtype=np.int64)
        ),  # (episode_count,)
    )
