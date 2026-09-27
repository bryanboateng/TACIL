import numpy as np
import numpy.typing as npt

from scripts.shared.episodes import Episodes


def assemble_episodes(
    *,
    expert_episodes: Episodes,
    synthetic_states: npt.NDArray[np.float32],  # (synthetic_step_count, state_dim)
    synthetic_actions: npt.NDArray[np.float32],  # (synthetic_step_count, action_dim)
    trajectory_lengths: npt.NDArray[np.int64],  # (anchor_count,)
    anchor_indices: npt.NDArray[np.int64],  # (anchor_count,)
    expert_part_length: int,
) -> Episodes:
    start_indices = (
        np.cumsum(trajectory_lengths) - trajectory_lengths
    )  # (anchor_count,)
    accepted_anchor_mask = trajectory_lengths > 0  # (anchor_count,)

    state_trajectories_per_episode: list[np.ndarray] = []
    action_trajectories_per_episode: list[np.ndarray] = []
    episode_lengths: list[int] = []

    for trajectory_length, start_index, anchor_index in zip(
        trajectory_lengths[accepted_anchor_mask],
        start_indices[accepted_anchor_mask],
        anchor_indices[accepted_anchor_mask],
    ):
        synthetic_part_slice = slice(start_index, start_index + trajectory_length)

        synthetic_state_trajectory = synthetic_states[
            synthetic_part_slice
        ]  # (trajectory_length, state_dimensionality)
        synthetic_action_trajectory = synthetic_actions[
            synthetic_part_slice
        ]  # (trajectory_length, action_dimensionality)

        expert_part_slice = slice(anchor_index, anchor_index + expert_part_length)

        state_trajectories_per_episode.append(
            np.concatenate(
                [synthetic_state_trajectory, expert_episodes.states[expert_part_slice]],
                axis=0,
            )
        )
        action_trajectories_per_episode.append(
            np.concatenate(
                [
                    synthetic_action_trajectory,
                    expert_episodes.actions[expert_part_slice],
                ],
                axis=0,
            )
        )
        episode_lengths.append(int(trajectory_length) + expert_part_length)

    return Episodes(
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
