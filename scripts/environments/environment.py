from typing import Protocol

import numpy as np
from gymnasium import Env

from scripts.environments.conditions import Condition
from scripts.environments.gpi_geometry import GpiGeometry
from scripts.environments.state_space import StateSpace
from scripts.evaluation.core.rollout import EpisodeRecord


class Environment(Protocol):
    """One task the pipeline can train and score a policy on.

    Everything that knows how a state is laid out, how a simulator is built,
    and how an episode looks on screen lives behind this interface.
    """

    @property
    def state_space(self) -> StateSpace: ...

    @property
    def gpi_geometry(self) -> GpiGeometry:
        """Raises when GPI cannot drive this environment."""
        ...

    @property
    def video_frames_per_second(self) -> int: ...

    def create_gym_environment(self, *, condition: Condition, render: bool) -> Env: ...

    def render_episode_frames(
        self,
        *,
        episode_record: EpisodeRecord,
        condition: Condition,
        policy_label: str,
        percentile: int,
    ) -> list[np.ndarray]:
        """Turn the raw frames of one episode into the frames of a video."""
        ...
