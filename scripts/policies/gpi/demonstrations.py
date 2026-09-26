import numpy as np
import torch

from scripts.environments.gpi_geometry import GpiGeometry
from scripts.shared.episodes import Episodes, calculate_episode_lengths
from scripts.shared.normalization import MinMaxNormalizer


class Demonstrations:
    def __init__(
        self, *, gpi_geometry: GpiGeometry, episodes: Episodes, device: torch.device
    ) -> None:
        self._gpi_geometry = gpi_geometry

        observations = torch.from_numpy(
            episodes.states
        ).float()  # (step_count, observation_dimensionality)

        self._observation_normalizer = MinMaxNormalizer(value_batch=observations).to(
            device=device
        )

        features = gpi_geometry.project_observations(
            observation_batch=observations
        )  # (step_count, feature_dimensionality)
        self._feature_normalizer = MinMaxNormalizer(value_batch=features).to(
            device=device
        )
        self._features = features.to(
            device=device
        )  # (step_count, feature_dimensionality)

        self._observations = observations.to(
            device=device
        )  # (step_count, observation_dimensionality)
        self._actions = gpi_geometry.convert_actions_to_object_frame(
            observation_batch=observations,
            action_batch=torch.from_numpy(episodes.actions).float(),
        ).to(device=device)  # (step_count, action_dimensionality)

        self._episode_end_index_per_step = torch.from_numpy(
            np.repeat(
                a=episodes.episode_end_indices,
                repeats=calculate_episode_lengths(
                    episode_end_indices=episodes.episode_end_indices
                ),
            )
        ).to(device=device)  # (step_count,)

    @property
    def observation_normalizer(self) -> MinMaxNormalizer:
        return self._observation_normalizer

    def find_nearest_step_indices(
        self,
        *,
        query_observation: torch.Tensor,  # (observation_dimensionality,)
        excluded_step_indices: set[int],
        neighbour_count: int,
        robot_distance_weight: float,
        environment_distance_weight: float,
    ) -> tuple[list[int], torch.Tensor]:  # indices, distances
        available_count = len(self._features) - len(excluded_step_indices)
        if available_count <= 0:
            raise RuntimeError(
                "One episode has used up all recorded steps, "
                "so no demonstration is left for it to follow."
            )

        query_feature = self._gpi_geometry.project_observations(
            observation_batch=query_observation.unsqueeze(dim=0)
        ).squeeze(dim=0)  # (feature_dimensionality,)

        distances = self._gpi_geometry.calculate_observation_distances(
            feature_batch=self._features,
            query_feature=query_feature,
            feature_normalizer=self._feature_normalizer,
            robot_distance_weight=robot_distance_weight,
            environment_distance_weight=environment_distance_weight,
        )  # (step_count,)

        if len(excluded_step_indices) > 0:
            distances = distances.index_fill(
                dim=0,
                index=torch.tensor(
                    sorted(excluded_step_indices), device=distances.device
                ),  # (excluded_count,)
                value=float("inf"),
            )

        selected_distances, selected_indices = torch.topk(
            input=distances, k=min(neighbour_count, available_count), largest=False
        )  # (neighbour_count,), (neighbour_count,)
        return [int(index) for index in selected_indices], selected_distances

    def read_step(
        self, *, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:  # observation, object-frame action
        return self._observations[index], self._actions[index]

    def get_remaining_steps_in_episode(self, *, step_index: int) -> int:
        return int(self._episode_end_index_per_step[step_index]) - step_index
