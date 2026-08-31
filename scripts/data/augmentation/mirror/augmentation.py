"""Doubles the expert data by mirroring it about the goal's axis of symmetry.

This is the cheapest augmentation in the repository and the only one whose
labels carry no error at all. It trains no model, solves nothing, and has
nothing to tune: see `reflection` for why the transform is exact.

What it buys is state coverage, which is the one thing the distillation
experiment identified as GPI's structural advantage. What it cannot buy is
recovery behaviour -- a mirrored expert episode is still an expert episode, so
it never shows the policy how to come back from a state the expert never
entered.

That is why this is a stage and not a strategy. What comes out is expert data,
so it runs before the augmentation rather than beside it, and whatever follows
sees twice as much: twice the transitions to fit the dynamics model on, twice
the anchors to grow chains from, twice the episodes to stitch between. With no
augmentation behind it, the stage on its own is the mirror strategy.
"""

import numpy as np
from wandb import Run

from scripts.data.augmentation.mirror.reflection import (
    measure_goal_pose_drift,
    reflect_actions,
    reflect_states,
)
from scripts.data.augmentation.mirror.statistics_logging import log_dataset_statistics
from scripts.shared.episode_store import EpisodeStore, concatenate_episode_stores

# The reflection is a valid relabelling only while the goal is a fixed point.
# The derivation says the drift is zero; this is the tolerance for the floating
# point that computes it.
MAXIMUM_GOAL_POSE_DRIFT = 1e-4


def expand_by_reflection(*, expert_episodes: EpisodeStore, run: Run) -> EpisodeStore:
    """The recorded episodes and their mirror images, as one store."""
    goal_pose_drift = measure_goal_pose_drift()
    if goal_pose_drift > MAXIMUM_GOAL_POSE_DRIFT:
        raise RuntimeError(
            f"The reflection moves the goal pose by {goal_pose_drift}, so a "
            f"mirrored episode would solve a different task and its recorded "
            f"actions would be wrong labels. The axis is derived from the goal "
            f"orientation, so this means the environment's goal has changed."
        )

    mirrored_episodes = create_mirrored_episodes(expert_episodes=expert_episodes)

    log_dataset_statistics(
        expert_episodes=expert_episodes,
        mirrored_episodes=mirrored_episodes,
        goal_pose_drift=goal_pose_drift,
        run=run,
    )

    return concatenate_episode_stores(
        episode_stores=[expert_episodes, mirrored_episodes]
    )


def create_mirrored_episodes(*, expert_episodes: EpisodeStore) -> EpisodeStore:
    """The mirror image of every recorded episode, in the same order.

    Episode boundaries are untouched: a reflection acts on each timestep on its
    own and changes no episode's length.
    """
    return EpisodeStore(
        states=reflect_states(state_batch=expert_episodes.states),
        actions=reflect_actions(action_batch=expert_episodes.actions),
        episode_end_indices=np.array(
            object=expert_episodes.episode_end_indices, dtype=np.int64
        ),
    )
