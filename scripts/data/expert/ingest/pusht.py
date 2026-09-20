from pathlib import Path

import numpy as np
import zarr
from numpy import typing as npt
from zarr import Array, Group

from scripts.environments.pusht import reward
from scripts.shared.episode_store import EpisodeStore, RewardedEpisodeStore

_STATE_KEY = "data/state"
_ACTION_KEY = "data/action"
_EPISODE_END_KEY = "meta/episode_ends"


def build_episode_store(*, source_path: Path) -> RewardedEpisodeStore:
    store = zarr.open(store=source_path, mode="r")

    if not isinstance(store, Group):
        raise TypeError(
            f"Expected zarr.Group at {source_path}, got {type(store).__name__}"
        )

    states = _read(store=store, key=_STATE_KEY).astype(
        np.float32
    )  # (total_timestep_count, state_dimensionality)
    actions = _read(store=store, key=_ACTION_KEY).astype(
        np.float32
    )  # (total_timestep_count, action_dimensionality)
    episode_end_indices = _read(store=store, key=_EPISODE_END_KEY).astype(
        np.int64
    )  # (episode_count,)

    episodes = EpisodeStore(
        states=states, actions=actions, episode_end_indices=episode_end_indices
    )

    return RewardedEpisodeStore(
        states=episodes.states,
        actions=episodes.actions,
        episode_end_indices=episodes.episode_end_indices,
        rewards=_calculate_rewards(episodes=episodes),
    )


def describe_source(*, source_path: Path) -> str:
    return f"PushT replay buffer at {source_path}"


# ---- Private --------------------------------------------


def _read(*, store: Group, key: str) -> np.ndarray:
    try:
        group_member = store[key]
    except KeyError as exc:
        raise KeyError(f"{key!r} is missing from the PushT dataset") from exc

    if not isinstance(group_member, Array):
        raise TypeError(f"{key!r} is not a zarr.Array")

    return np.asarray(a=group_member[:])


def _calculate_rewards(
    *, episodes: EpisodeStore
) -> npt.NDArray[np.float32]:  # (total_timestep_count,)
    source_indices = episodes.indices_with_successor  # (transition_count,)

    rewards = np.full(
        shape=len(episodes.states), fill_value=np.nan, dtype=np.float32
    )  # (total_timestep_count,)
    rewards[source_indices] = reward.calculate_reward(
        state_batch=episodes.states[source_indices + 1],
        environment=reward.create_environment(),
    ).astype(np.float32)
    return rewards
