from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from scripts.data.augmentation.assembly import assemble_episodes
from scripts.data.augmentation.tacil.create_data.backward_rollout import (
    SyntheticTrajectories,
)
from scripts.shared.episodes import Episodes


@dataclass(frozen=True)
class AugmentationResult:
    @dataclass(frozen=True)
    class OutcomeCounts:
        empty: int  # anchors whose first backward step already broke the budget
        accepted: int

    episodes: Episodes
    trajectory_lengths: npt.NDArray[np.int64]  # (accepted_count,)
    counts: OutcomeCounts


def create_augmented_episodes(
    *,
    synthetic_trajectories: SyntheticTrajectories,
    expert_episodes: Episodes,
    anchor_indices: npt.NDArray[np.int64],  # (anchor_count,)
    expert_part_length: int,
) -> AugmentationResult:
    trajectory_lengths: npt.NDArray[np.int64] = (
        synthetic_trajectories.trajectory_lengths.cpu().numpy()
    )  # (anchor_count,)

    accepted_anchor_mask = trajectory_lengths > 0  # (anchor_count,)
    accepted_anchor_count = int(accepted_anchor_mask.sum())
    if accepted_anchor_count == 0:
        raise RuntimeError("Can't create augmented episodes. No anchor was accepted.")

    augmented_episodes = assemble_episodes(
        expert_episodes=expert_episodes,
        synthetic_states=synthetic_trajectories.states.cpu().numpy(),
        synthetic_actions=synthetic_trajectories.actions.cpu().numpy(),
        trajectory_lengths=trajectory_lengths,
        anchor_indices=np.asarray(anchor_indices, dtype=np.int64),
        expert_part_length=expert_part_length,
    )

    return AugmentationResult(
        episodes=augmented_episodes,
        trajectory_lengths=trajectory_lengths[accepted_anchor_mask],
        counts=AugmentationResult.OutcomeCounts(
            empty=int(trajectory_lengths.shape[0]) - accepted_anchor_count,
            accepted=accepted_anchor_count,
        ),
    )
