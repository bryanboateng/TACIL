import torch
from mani_skill.utils.geometry.rotation_conversions import (
    axis_angle_to_matrix,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
)

from scripts.environments.maniskill.panda_action import (
    ACTION_GRIPPER_SLICE,
    ACTION_ROTATION_SLICE,
    ACTION_TRANSLATION_SLICE,
    GRIPPER_RANGE_IN_METRES,
    ROTATION_MAGNITUDE_IN_RADIANS,
    ROTATION_SCALE_IN_RADIANS,
    TRANSLATION_SCALE_IN_METRES,
    calculate_norms,
    calculate_rotation_angles,
    clip_to_unit_norm,
    read_rotations,
)
from scripts.environments.stack_cube.observation import (
    CUBE_A_ORIENTATION_SLICE,
    CUBE_A_POSITION_SLICE,
    CUBE_B_POSITION_SLICE,
    GRIPPER_OPENING_INDEX,
    TCP_ORIENTATION_SLICE,
    TCP_POSITION_SLICE,
)
from scripts.shared.normalization import MinMaxNormalizer

FEATURE_DIMENSIONALITY = 25

_FEATURE_TCP_POSITION_SLICE = slice(0, 3)
_FEATURE_TCP_ROTATION_SLICE = slice(3, 12)
_FEATURE_GRIPPER_SLICE = slice(12, 13)
_FEATURE_CUBE_A_POSITION_SLICE = slice(13, 16)
_FEATURE_CUBE_A_ROTATION_SLICE = slice(16, 25)


class StackCubeGpiGeometry:
    def project_observations(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
    ) -> torch.Tensor:  # (batch_size, FEATURE_DIMENSIONALITY)
        base_cube_position = observation_batch[
            :, CUBE_B_POSITION_SLICE
        ]  # (batch_size, 3)

        return torch.cat(
            tensors=[
                observation_batch[:, TCP_POSITION_SLICE] - base_cube_position,
                read_rotations(
                    observation_batch=observation_batch,
                    orientation_slice=TCP_ORIENTATION_SLICE,
                ).flatten(start_dim=-2),
                observation_batch[:, GRIPPER_OPENING_INDEX : GRIPPER_OPENING_INDEX + 1],
                observation_batch[:, CUBE_A_POSITION_SLICE] - base_cube_position,
                read_rotations(
                    observation_batch=observation_batch,
                    orientation_slice=CUBE_A_ORIENTATION_SLICE,
                ).flatten(start_dim=-2),
            ],
            dim=-1,
        )

    def calculate_observation_distances(
        self,
        *,
        feature_batch: torch.Tensor,  # (step_count, FEATURE_DIMENSIONALITY)
        query_feature: torch.Tensor,  # (FEATURE_DIMENSIONALITY,)
        feature_normalizer: MinMaxNormalizer,
        robot_distance_weight: float,
        environment_distance_weight: float,
    ) -> torch.Tensor:  # (step_count,)
        del feature_normalizer

        difference_batch = (
            feature_batch - query_feature
        )  # (step_count, FEATURE_DIMENSIONALITY)

        robot_distance = (
            calculate_norms(
                difference_batch=difference_batch,
                feature_slice=_FEATURE_TCP_POSITION_SLICE,
            )
            / TRANSLATION_SCALE_IN_METRES
            + calculate_rotation_angles(
                feature_batch=feature_batch,
                query_feature=query_feature,
                rotation_slice=_FEATURE_TCP_ROTATION_SLICE,
            )
            / ROTATION_MAGNITUDE_IN_RADIANS
            + calculate_norms(
                difference_batch=difference_batch, feature_slice=_FEATURE_GRIPPER_SLICE
            )
            / GRIPPER_RANGE_IN_METRES
        )  # (step_count,)

        environment_distance = (
            calculate_norms(
                difference_batch=difference_batch,
                feature_slice=_FEATURE_CUBE_A_POSITION_SLICE,
            )
            / TRANSLATION_SCALE_IN_METRES
            + calculate_rotation_angles(
                feature_batch=feature_batch,
                query_feature=query_feature,
                rotation_slice=_FEATURE_CUBE_A_ROTATION_SLICE,
            )
            / ROTATION_MAGNITUDE_IN_RADIANS
        )  # (step_count,)

        return (
            robot_distance_weight * robot_distance
            + environment_distance_weight * environment_distance
        )

    def calculate_action(
        self,
        *,
        recorded_observation: torch.Tensor,  # (STATE_DIMENSIONALITY,)
        recorded_action: torch.Tensor,  # (ACTION_DIMENSIONALITY,)
        current_observation: torch.Tensor,  # (STATE_DIMENSIONALITY,)
        progression_weight: float,
        attraction_weight: float,
    ) -> torch.Tensor:  # (ACTION_DIMENSIONALITY,)
        recorded_feature = self.project_observations(
            observation_batch=recorded_observation.unsqueeze(dim=0)
        ).squeeze(dim=0)  # (FEATURE_DIMENSIONALITY,)
        current_feature = self.project_observations(
            observation_batch=current_observation.unsqueeze(dim=0)
        ).squeeze(dim=0)  # (FEATURE_DIMENSIONALITY,)

        translation = progression_weight * recorded_action[
            ACTION_TRANSLATION_SLICE
        ] + attraction_weight * (
            recorded_feature[_FEATURE_TCP_POSITION_SLICE]
            - current_feature[_FEATURE_TCP_POSITION_SLICE]
        )  # (3,)

        rotation = progression_weight * recorded_action[
            ACTION_ROTATION_SLICE
        ] + attraction_weight * matrix_to_axis_angle(
            matrix=recorded_feature[_FEATURE_TCP_ROTATION_SLICE].reshape(3, 3)
            @ current_feature[_FEATURE_TCP_ROTATION_SLICE]
            .reshape(3, 3)
            .transpose(dim0=-2, dim1=-1)
        )  # (3,)

        return torch.cat(
            tensors=[translation, rotation, recorded_action[ACTION_GRIPPER_SLICE]],
            dim=-1,
        )

    def convert_actions_to_object_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
        action_batch: torch.Tensor,  # (batch_size, ACTION_DIMENSIONALITY)
    ) -> torch.Tensor:  # (batch_size, ACTION_DIMENSIONALITY)
        del observation_batch

        return torch.cat(
            tensors=[
                action_batch[:, ACTION_TRANSLATION_SLICE].clamp(min=-1.0, max=1.0)
                * TRANSLATION_SCALE_IN_METRES,
                matrix_to_axis_angle(
                    matrix=euler_angles_to_matrix(
                        euler_angles=clip_to_unit_norm(
                            vector_batch=action_batch[:, ACTION_ROTATION_SLICE]
                        )
                        * ROTATION_SCALE_IN_RADIANS,
                        convention="XYZ",
                    )
                ),
                action_batch[:, ACTION_GRIPPER_SLICE],
            ],
            dim=-1,
        )

    def convert_actions_to_world_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
        action_batch: torch.Tensor,  # (batch_size, ACTION_DIMENSIONALITY)
    ) -> torch.Tensor:  # (batch_size, ACTION_DIMENSIONALITY)
        del observation_batch

        return torch.cat(
            tensors=[
                (
                    action_batch[:, ACTION_TRANSLATION_SLICE]
                    / TRANSLATION_SCALE_IN_METRES
                ).clamp(min=-1.0, max=1.0),
                clip_to_unit_norm(
                    vector_batch=matrix_to_euler_angles(
                        matrix=axis_angle_to_matrix(
                            axis_angle=action_batch[:, ACTION_ROTATION_SLICE]
                        ),
                        convention="XYZ",
                    )
                    / ROTATION_SCALE_IN_RADIANS
                ),
                action_batch[:, ACTION_GRIPPER_SLICE],
            ],
            dim=-1,
        )
