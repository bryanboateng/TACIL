import torch

from scripts.environments.pusht.observation import (
    AGENT_POSITION_SLICE,
    BLOCK_ORIENTATION_INDEX,
    BLOCK_POSITION_SLICE,
    POSITION_SLICE,
)
from scripts.shared.normalization import MinMaxNormalizer


class PushTGpiGeometry:
    def project_observations(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
    ) -> torch.Tensor:  # (batch_size, STATE_DIMENSIONALITY)
        return observation_batch

    def calculate_observation_distances(
        self,
        *,
        feature_batch: torch.Tensor,  # (step_count, STATE_DIMENSIONALITY)
        query_feature: torch.Tensor,  # (STATE_DIMENSIONALITY,)
        feature_normalizer: MinMaxNormalizer,
        robot_distance_weight: float,
        environment_distance_weight: float,
    ) -> torch.Tensor:  # (step_count,)
        del robot_distance_weight, environment_distance_weight

        normalized_feature_batch = feature_normalizer.normalize(
            value_batch=feature_batch
        )  # (step_count, STATE_DIMENSIONALITY)
        normalized_query_feature = feature_normalizer.normalize(
            value_batch=query_feature.unsqueeze(dim=0)
        ).squeeze(dim=0)  # (STATE_DIMENSIONALITY,)

        position_difference_batch = (
            normalized_feature_batch[:, POSITION_SLICE]
            - normalized_query_feature[POSITION_SLICE]
        )  # (step_count, 4)

        angle_difference_batch = (
            normalized_feature_batch[:, BLOCK_ORIENTATION_INDEX]
            - normalized_query_feature[BLOCK_ORIENTATION_INDEX]
        ).abs()  # (step_count,)
        wrapped_angle_difference_batch = torch.minimum(
            input=angle_difference_batch, other=2.0 - angle_difference_batch
        )  # (step_count,)

        return torch.sqrt(
            input=(position_difference_batch**2).sum(dim=1)
            + wrapped_angle_difference_batch**2
        )  # (step_count,)

    def calculate_action(
        self,
        *,
        recorded_observation: torch.Tensor,  # (STATE_DIMENSIONALITY,)
        recorded_action: torch.Tensor,  # (ACTION_DIMENSIONALITY,)
        current_observation: torch.Tensor,  # (STATE_DIMENSIONALITY,)
        progression_weight: float,
        attraction_weight: float,
    ) -> torch.Tensor:  # (ACTION_DIMENSIONALITY,)
        recorded_position = _convert_agent_positions_to_object_frame(
            observation_batch=recorded_observation.unsqueeze(dim=0)
        ).squeeze(dim=0)  # (ACTION_DIMENSIONALITY,)
        current_position = _convert_agent_positions_to_object_frame(
            observation_batch=current_observation.unsqueeze(dim=0)
        ).squeeze(dim=0)  # (ACTION_DIMENSIONALITY,)

        progression_flow = recorded_action - recorded_position
        attraction_flow = recorded_position - current_position
        return (
            current_position
            + progression_weight * progression_flow
            + attraction_weight * attraction_flow
        )

    def convert_actions_to_object_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
        action_batch: torch.Tensor,  # (batch_size, ACTION_DIMENSIONALITY)
    ) -> torch.Tensor:  # (batch_size, ACTION_DIMENSIONALITY)
        return _rotate(
            vector_batch=action_batch
            - observation_batch[:, BLOCK_POSITION_SLICE],  # (batch_size, 2)
            angle_batch=-observation_batch[:, BLOCK_ORIENTATION_INDEX],
        )

    def convert_actions_to_world_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
        action_batch: torch.Tensor,  # (batch_size, ACTION_DIMENSIONALITY)
    ) -> torch.Tensor:  # (batch_size, ACTION_DIMENSIONALITY)
        return (
            _rotate(
                vector_batch=action_batch,
                angle_batch=observation_batch[:, BLOCK_ORIENTATION_INDEX],
            )
            + observation_batch[:, BLOCK_POSITION_SLICE]
        )


# ---- Private --------------------------------------------


def _convert_agent_positions_to_object_frame(
    *,
    observation_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
) -> torch.Tensor:  # (batch_size, 2)
    return _rotate(
        vector_batch=observation_batch[:, AGENT_POSITION_SLICE]
        - observation_batch[:, BLOCK_POSITION_SLICE],  # (batch_size, 2)
        angle_batch=-observation_batch[:, BLOCK_ORIENTATION_INDEX],
    )


def _rotate(
    *,
    vector_batch: torch.Tensor,  # (batch_size, 2)
    angle_batch: torch.Tensor,  # (batch_size,)
) -> torch.Tensor:  # (batch_size, 2)
    return torch.matmul(
        input=_create_rotation_matrix(angle_batch=angle_batch),
        other=vector_batch.unsqueeze(dim=-1),
    ).squeeze(dim=-1)


def _create_rotation_matrix(
    *,
    angle_batch: torch.Tensor,  # (batch_size,)
) -> torch.Tensor:  # (batch_size, 2, 2)
    cosine_batch = torch.cos(input=angle_batch)  # (batch_size,)
    sine_batch = torch.sin(input=angle_batch)  # (batch_size,)
    return torch.stack(
        tensors=[
            torch.stack(tensors=[cosine_batch, -sine_batch], dim=-1),
            torch.stack(tensors=[sine_batch, cosine_batch], dim=-1),
        ],
        dim=-2,
    )  # (batch_size, 2, 2)
