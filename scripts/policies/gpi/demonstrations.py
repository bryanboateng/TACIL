import numpy as np
import torch

from scripts.policies.gpi.geometry import (
    calculate_observation_distances,
    convert_actions_to_block_frame,
)
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.normalization import MinMaxNormalizer


class Demonstrations:
    def __init__(self, *, episode_store: EpisodeStore, device: torch.device) -> None:
        observations = torch.from_numpy(
            episode_store.states
        ).float()  # (step_count, observation_dimensionality)

        actions = convert_actions_to_block_frame(
            observation_batch=observations,
            action_batch=torch.from_numpy(episode_store.actions).float(),
        )  # (step_count, action_dimensionality)

        self._observation_normalizer = MinMaxNormalizer(
            value_batch=observations
        ).to(device=device)
        self._action_normalizer = MinMaxNormalizer(value_batch=actions).to(
            device=device
        )

        self._normalized_observations = self._observation_normalizer.normalize(
            value_batch=observations.to(device=device)
        )  # (step_count, observation_dimensionality)
        self._normalized_actions = self._action_normalizer.normalize(
            value_batch=actions.to(device=device)
        )  # (step_count, action_dimensionality)

        # Where the demonstration each step belongs to ends.
        self._end_index_per_step = torch.from_numpy(
            np.repeat(
                a=episode_store.episode_end_indices,
                repeats=np.diff(a=episode_store.episode_end_indices, prepend=0),
            )
        ).to(device=device)  # (step_count,)

    @property
    def step_count(self) -> int:
        return len(self._normalized_observations)

    def normalize_observation(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, observation_dimensionality)
    ) -> torch.Tensor:  # (batch_size, observation_dimensionality)
        return self._observation_normalizer.normalize(value_batch=observation_batch)

    def denormalize_actions(
        self,
        *,
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        return self._action_normalizer.denormalize(value_batch=action_batch)

    def find_nearest_step_index(
        self,
        *,
        query_observation: torch.Tensor,  # (observation_dimensionality,)
        excluded_step_indices: set[int],
    ) -> int:
        if len(excluded_step_indices) >= self.step_count:
            raise RuntimeError(
                f"One episode has used up all {self.step_count} recorded steps, "
                f"so no demonstration is left for it to follow. The "
                f"demonstrations hold fewer steps than an episode runs."
            )

        distances = calculate_observation_distances(
            observation_batch=self._normalized_observations,
            query_observation=query_observation,
        )  # (step_count,)

        if len(excluded_step_indices) > 0:
            distances = distances.index_fill(
                dim=0,
                index=torch.tensor(
                    sorted(excluded_step_indices), device=distances.device
                ),  # (excluded_count,)
                value=float("inf"),
            )
        return int(distances.argmin().item())

    def read_step(
        self, *, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:  # observation, action
        return self._normalized_observations[index], self._normalized_actions[index]

    def find_plan_end_index(self, *, start_index: int, maximum_length: int) -> int:
        return min(
            start_index + maximum_length, int(self._end_index_per_step[start_index])
        )
