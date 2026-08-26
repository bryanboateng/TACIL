import numpy as np
import torch

from scripts.policies.gpi.geometry import (
    calculate_observation_distances,
    convert_actions_to_block_frame,
)
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.normalization import MinMaxNormalizer
from scripts.shared.pusht_state import BLOCK_ORIENTATION_INDEX, POSITION_SLICE


class Demonstrations:
    def __init__(self, *, episode_store: EpisodeStore, device: torch.device) -> None:
        observations = torch.from_numpy(
            episode_store.states
        ).float()  # (step_count, observation_dimensionality)

        actions = convert_actions_to_block_frame(
            observation_batch=observations,
            action_batch=torch.from_numpy(episode_store.actions).float(),
        )  # (step_count, action_dimensionality)

        self._observation_normalizer = MinMaxNormalizer(value_batch=observations).to(
            device=device
        )
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

    def find_nearest_step_indices(
        self,
        *,
        query_observation_batch: torch.Tensor,  # (query_count, observation_dimensionality)
    ) -> torch.Tensor:  # (query_count,)
        """The nearest recorded step for every query, with no exclusions -- the
        batched counterpart of `find_nearest_step_index` used to label many
        synthetic states at once."""
        demonstration_positions = self._normalized_observations[
            :, POSITION_SLICE
        ]  # (step_count, 4)
        demonstration_angles = self._normalized_observations[
            :, BLOCK_ORIENTATION_INDEX
        ]  # (step_count,)

        nearest_indices: list[torch.Tensor] = []
        chunk_size = 512
        for start in range(0, query_observation_batch.shape[0], chunk_size):
            query_chunk = query_observation_batch[start : start + chunk_size]  # (c, D)
            position_difference = (
                query_chunk[:, None, POSITION_SLICE] - demonstration_positions[None]
            )  # (c, step_count, 4)
            angle_difference = (
                query_chunk[:, None, BLOCK_ORIENTATION_INDEX]
                - demonstration_angles[None]
            ).abs()  # (c, step_count)
            wrapped_angle_difference = torch.minimum(
                input=angle_difference, other=2.0 - angle_difference
            )
            squared_distance = (position_difference**2).sum(
                dim=2
            ) + wrapped_angle_difference**2  # (c, step_count)
            nearest_indices.append(squared_distance.argmin(dim=1))
        return torch.cat(tensors=nearest_indices)  # (query_count,)

    def read_steps(
        self,
        *,
        indices: torch.Tensor,  # (query_count,)
    ) -> tuple[torch.Tensor, torch.Tensor]:  # observations, actions
        return self._normalized_observations[indices], self._normalized_actions[indices]

    def find_plan_end_index(self, *, start_index: int, maximum_length: int) -> int:
        return min(
            start_index + maximum_length, int(self._end_index_per_step[start_index])
        )
