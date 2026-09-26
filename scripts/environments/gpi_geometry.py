from typing import Protocol

import torch

from scripts.shared.normalization import MinMaxNormalizer


class GpiGeometry(Protocol):
    def project_observations(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, feature_dimensionality)
        ...

    def calculate_observation_distances(
        self,
        *,
        feature_batch: torch.Tensor,  # (step_count, feature_dimensionality)
        query_feature: torch.Tensor,  # (feature_dimensionality,)
        feature_normalizer: MinMaxNormalizer,
        robot_distance_weight: float,
        environment_distance_weight: float,
    ) -> torch.Tensor:  # (step_count,)
        ...

    def calculate_action(
        self,
        *,
        recorded_observation: torch.Tensor,  # (state_dimensionality,)
        recorded_action: torch.Tensor,  # (action_dimensionality,)
        current_observation: torch.Tensor,  # (state_dimensionality,)
        progression_weight: float,
        attraction_weight: float,
    ) -> torch.Tensor:  # (action_dimensionality,)
        ...

    def convert_actions_to_object_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        ...

    def convert_actions_to_world_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        ...
