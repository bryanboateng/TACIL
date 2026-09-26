import torch
from mani_skill.utils.geometry.rotation_conversions import (
    axis_angle_to_matrix,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
    rotation_6d_to_matrix,
)

from scripts.environments.peg_insertion.observation import (
    ARM_ACTION_COUNT,
    GRIPPER_OPENING_INDEX,
    HOLE_ORIENTATION_SLICE,
    HOLE_POSITION_SLICE,
    HOLE_RADIUS_INDEX,
    PEG_HALF_SIZE_SLICE,
    PEG_ORIENTATION_SLICE,
    PEG_POSITION_SLICE,
    TCP_ORIENTATION_SLICE,
    TCP_POSITION_SLICE,
)
from scripts.shared.normalization import MinMaxNormalizer

_ACTION_TRANSLATION_SLICE = slice(0, 3)
_ACTION_ROTATION_SLICE = slice(3, ARM_ACTION_COUNT)
_ACTION_GRIPPER_SLICE = slice(ARM_ACTION_COUNT, ARM_ACTION_COUNT + 1)

_TRANSLATION_SCALE_IN_METRES = 0.1
_ROTATION_SCALE_IN_RADIANS = -0.1

FEATURE_DIMENSIONALITY = 29

_FEATURE_TCP_POSITION_SLICE = slice(0, 3)
_FEATURE_TCP_ROTATION_SLICE = slice(3, 12)
_FEATURE_GRIPPER_SLICE = slice(12, 13)
_FEATURE_PEG_POSITION_SLICE = slice(13, 16)
_FEATURE_PEG_ROTATION_SLICE = slice(16, 25)
_FEATURE_PEG_HALF_SIZE_SLICE = slice(25, 28)
_FEATURE_HOLE_RADIUS_SLICE = slice(28, 29)

_ROBOT_LENGTH_FEATURE_SLICES = (_FEATURE_TCP_POSITION_SLICE, _FEATURE_GRIPPER_SLICE)
_ENVIRONMENT_LENGTH_FEATURE_SLICES = (
    _FEATURE_PEG_POSITION_SLICE,
    _FEATURE_PEG_HALF_SIZE_SLICE,
    _FEATURE_HOLE_RADIUS_SLICE,
)


class PegInsertionGpiGeometry:
    def project_observations(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
    ) -> torch.Tensor:  # (batch_size, FEATURE_DIMENSIONALITY)
        hole_rotation = _read_rotations(
            observation_batch=observation_batch,
            orientation_slice=HOLE_ORIENTATION_SLICE,
        )  # (batch_size, 3, 3)
        hole_position = observation_batch[:, HOLE_POSITION_SLICE]  # (batch_size, 3)

        return torch.cat(
            tensors=[
                _rotate_into_frame(
                    vector_batch=observation_batch[:, TCP_POSITION_SLICE]
                    - hole_position,
                    frame_rotation=hole_rotation,
                ),
                _flatten_rotations_in_frame(
                    rotation_batch=_read_rotations(
                        observation_batch=observation_batch,
                        orientation_slice=TCP_ORIENTATION_SLICE,
                    ),
                    frame_rotation=hole_rotation,
                ),
                observation_batch[:, GRIPPER_OPENING_INDEX : GRIPPER_OPENING_INDEX + 1],
                _rotate_into_frame(
                    vector_batch=observation_batch[:, PEG_POSITION_SLICE]
                    - hole_position,
                    frame_rotation=hole_rotation,
                ),
                _flatten_rotations_in_frame(
                    rotation_batch=_read_rotations(
                        observation_batch=observation_batch,
                        orientation_slice=PEG_ORIENTATION_SLICE,
                    ),
                    frame_rotation=hole_rotation,
                ),
                observation_batch[:, PEG_HALF_SIZE_SLICE],
                observation_batch[:, HOLE_RADIUS_INDEX : HOLE_RADIUS_INDEX + 1],
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
        difference_batch = (feature_batch - query_feature) * _scale_to_unit_interval(
            feature_normalizer=feature_normalizer
        )  # (step_count, FEATURE_DIMENSIONALITY)

        robot_distance = _sum_group_norms(
            difference_batch=difference_batch, group_slices=_ROBOT_LENGTH_FEATURE_SLICES
        ) + _calculate_rotation_distances(
            feature_batch=feature_batch,
            query_feature=query_feature,
            rotation_slice=_FEATURE_TCP_ROTATION_SLICE,
        )  # (step_count,)

        environment_distance = _sum_group_norms(
            difference_batch=difference_batch,
            group_slices=_ENVIRONMENT_LENGTH_FEATURE_SLICES,
        ) + _calculate_rotation_distances(
            feature_batch=feature_batch,
            query_feature=query_feature,
            rotation_slice=_FEATURE_PEG_ROTATION_SLICE,
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
            _ACTION_TRANSLATION_SLICE
        ] + attraction_weight * (
            recorded_feature[_FEATURE_TCP_POSITION_SLICE]
            - current_feature[_FEATURE_TCP_POSITION_SLICE]
        )  # (3,)

        rotation = progression_weight * recorded_action[
            _ACTION_ROTATION_SLICE
        ] + attraction_weight * matrix_to_axis_angle(
            matrix=recorded_feature[_FEATURE_TCP_ROTATION_SLICE].reshape(3, 3)
            @ current_feature[_FEATURE_TCP_ROTATION_SLICE]
            .reshape(3, 3)
            .transpose(dim0=-2, dim1=-1)
        )  # (3,)

        return torch.cat(
            tensors=[translation, rotation, recorded_action[_ACTION_GRIPPER_SLICE]],
            dim=-1,
        )

    def convert_actions_to_object_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
        action_batch: torch.Tensor,  # (batch_size, ACTION_DIMENSIONALITY)
    ) -> torch.Tensor:  # (batch_size, ACTION_DIMENSIONALITY)
        hole_rotation = _read_rotations(
            observation_batch=observation_batch,
            orientation_slice=HOLE_ORIENTATION_SLICE,
        )  # (batch_size, 3, 3)

        return torch.cat(
            tensors=[
                _rotate_into_frame(
                    vector_batch=action_batch[:, _ACTION_TRANSLATION_SLICE].clamp(
                        min=-1.0, max=1.0
                    )
                    * _TRANSLATION_SCALE_IN_METRES,
                    frame_rotation=hole_rotation,
                ),
                matrix_to_axis_angle(
                    matrix=hole_rotation.transpose(dim0=-2, dim1=-1)
                    @ euler_angles_to_matrix(
                        euler_angles=_clip_to_unit_norm(
                            vector_batch=action_batch[:, _ACTION_ROTATION_SLICE]
                        )
                        * _ROTATION_SCALE_IN_RADIANS,
                        convention="XYZ",
                    )
                    @ hole_rotation
                ),
                action_batch[:, _ACTION_GRIPPER_SLICE],
            ],
            dim=-1,
        )

    def convert_actions_to_world_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
        action_batch: torch.Tensor,  # (batch_size, ACTION_DIMENSIONALITY)
    ) -> torch.Tensor:  # (batch_size, ACTION_DIMENSIONALITY)
        hole_rotation = _read_rotations(
            observation_batch=observation_batch,
            orientation_slice=HOLE_ORIENTATION_SLICE,
        )  # (batch_size, 3, 3)

        world_translation = torch.matmul(
            input=hole_rotation,
            other=action_batch[:, _ACTION_TRANSLATION_SLICE].unsqueeze(dim=-1),
        ).squeeze(dim=-1)  # (batch_size, 3)
        world_rotation = (
            hole_rotation
            @ axis_angle_to_matrix(axis_angle=action_batch[:, _ACTION_ROTATION_SLICE])
            @ hole_rotation.transpose(dim0=-2, dim1=-1)
        )  # (batch_size, 3, 3)

        return torch.cat(
            tensors=[
                (world_translation / _TRANSLATION_SCALE_IN_METRES).clamp(
                    min=-1.0, max=1.0
                ),
                _clip_to_unit_norm(
                    vector_batch=matrix_to_euler_angles(
                        matrix=world_rotation, convention="XYZ"
                    )
                    / _ROTATION_SCALE_IN_RADIANS
                ),
                action_batch[:, _ACTION_GRIPPER_SLICE],
            ],
            dim=-1,
        )


# ---- Private --------------------------------------------


def _read_rotations(
    *,
    observation_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
    orientation_slice: slice,
) -> torch.Tensor:  # (batch_size, 3, 3)
    return rotation_6d_to_matrix(d6=observation_batch[:, orientation_slice]).transpose(
        dim0=-2, dim1=-1
    )


def _rotate_into_frame(
    *,
    vector_batch: torch.Tensor,  # (batch_size, 3)
    frame_rotation: torch.Tensor,  # (batch_size, 3, 3)
) -> torch.Tensor:  # (batch_size, 3)
    return torch.matmul(
        input=frame_rotation.transpose(dim0=-2, dim1=-1),
        other=vector_batch.unsqueeze(dim=-1),
    ).squeeze(dim=-1)


def _flatten_rotations_in_frame(
    *,
    rotation_batch: torch.Tensor,  # (batch_size, 3, 3)
    frame_rotation: torch.Tensor,  # (batch_size, 3, 3)
) -> torch.Tensor:  # (batch_size, 9)
    return (frame_rotation.transpose(dim0=-2, dim1=-1) @ rotation_batch).flatten(
        start_dim=-2
    )


def _clip_to_unit_norm(
    *,
    vector_batch: torch.Tensor,  # (batch_size, 3)
) -> torch.Tensor:  # (batch_size, 3)
    norm = torch.linalg.norm(vector_batch, dim=-1, keepdim=True)  # (batch_size, 1)
    return torch.where(
        condition=norm > 1.0, input=vector_batch / norm, other=vector_batch
    )


def _scale_to_unit_interval(
    *, feature_normalizer: MinMaxNormalizer
) -> torch.Tensor:  # (FEATURE_DIMENSIONALITY,)
    return feature_normalizer.scale / 2.0


def _sum_group_norms(
    *,
    difference_batch: torch.Tensor,  # (step_count, FEATURE_DIMENSIONALITY)
    group_slices: tuple[slice, ...],
) -> torch.Tensor:  # (step_count,)
    return sum(
        torch.linalg.norm(difference_batch[:, group_slice], dim=-1)
        for group_slice in group_slices
    )


def _calculate_rotation_distances(
    *,
    feature_batch: torch.Tensor,  # (step_count, FEATURE_DIMENSIONALITY)
    query_feature: torch.Tensor,  # (FEATURE_DIMENSIONALITY,)
    rotation_slice: slice,
) -> torch.Tensor:  # (step_count,)
    trace_batch = (
        feature_batch[:, rotation_slice] * query_feature[rotation_slice]
    ).sum(dim=-1)  # (step_count,)
    return (
        torch.arccos(input=((trace_batch - 1.0) / 2.0).clamp(min=-1.0, max=1.0))
        / torch.pi
    )
