import numpy as np

from scripts.policies.gpi.dataset import PushTEpisodeDataset


class PlanBuffer:
    """A rolling window of demonstration actions/states, popped one step at a time."""

    def __init__(self, dataset: PushTEpisodeDataset, horizon: int) -> None:
        self.dataset = dataset
        self.horizon_length = horizon
        self.clear()

    def clear(self) -> None:
        self.episode_index: int | None = None
        self.start_timestep: int | None = None
        self.pointer: int = 0
        self.actions: np.ndarray = np.empty((0,))
        self.states: np.ndarray = np.empty((0,))

    def load(self, episode_index: int, timestep: int) -> None:
        episode = self.dataset[episode_index]
        end = timestep + self.horizon_length
        self.actions = np.asarray(episode["action"][timestep:end])
        self.states = np.asarray(episode["obs"][timestep:end])
        self.episode_index = episode_index
        self.start_timestep = timestep
        self.pointer = 0

    def pop(self) -> tuple[np.ndarray, np.ndarray, int]:
        if self.episode_index is None:
            raise RuntimeError("Buffer has not been loaded")
        if self.is_empty():
            raise RuntimeError("Buffer is empty")

        action = self.actions[self.pointer]
        state = self.states[self.pointer]
        timestep = self.start_timestep + self.pointer
        self.pointer += 1
        return action, state, timestep

    def is_empty(self) -> bool:
        return self.pointer >= len(self.actions)
