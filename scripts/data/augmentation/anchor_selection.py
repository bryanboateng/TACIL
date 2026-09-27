import numpy as np
import numpy.typing as npt

from scripts.shared.episodes import (
    calculate_episode_lengths,
    calculate_episode_start_indices,
)


def get_anchor_indices(
    *,
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
    expert_part_length: int,
) -> npt.NDArray[np.int64]:  # (anchor_count,)
    episode_start_indices = calculate_episode_start_indices(
        episode_end_indices=episode_end_indices
    )  # (episode_count,)

    episode_lengths = calculate_episode_lengths(
        episode_end_indices=episode_end_indices
    )  # (episode_count,)
    if (episode_lengths < expert_part_length).any():
        raise ValueError(
            f"Found an episode shorter than the expert part length "
            f"({expert_part_length}); it could hold no anchor."
        )

    return np.concatenate(
        [
            np.arange(
                episode_start_index,
                episode_end_index - max(expert_part_length, 1) + 1,
                dtype=episode_end_indices.dtype,
            )
            for episode_start_index, episode_end_index in zip(
                episode_start_indices, episode_end_indices
            )
        ]
    )
