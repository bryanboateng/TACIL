import torch

from scripts.environments import angle_arithmetic
from scripts.environments.peg_insertion.observation import (
    ACTION_DIMENSIONALITY,
    ANGLE_INDICES,
    ARM_JOINT_COUNT,
    GRIPPER_POSITION_SLICE,
    HOLE_ORIENTATION_INDEX,
    HOLE_POSITION_SLICE,
    HOLE_RADIUS_INDEX,
    JOINT_ANGLE_SLICE,
    JOINT_VELOCITY_SLICE,
    PEG_HALF_SIZE_SLICE,
    PEG_ORIENTATION_INDEX,
    PEG_POSITION_SLICE,
    STATE_DIMENSIONALITY,
)
from scripts.environments.state_picture import (
    OrientationGroup,
    PositionGroup,
    StatePicture,
)

_ARM_JOINT_SLICE = slice(0, ARM_JOINT_COUNT)
_ARM_ACTION_SLICE = slice(0, ARM_JOINT_COUNT)
_GRIPPER_ACTION_SLICE = slice(ARM_JOINT_COUNT, ACTION_DIMENSIONALITY)

_NETWORK_INPUT_DIMENSIONALITY = (
    (JOINT_ANGLE_SLICE.stop - JOINT_ANGLE_SLICE.start)
    + (JOINT_VELOCITY_SLICE.stop - JOINT_VELOCITY_SLICE.start)
    + 3
    + 3
    + 3
    + 2
    + 2
    + (PEG_HALF_SIZE_SLICE.stop - PEG_HALF_SIZE_SLICE.start)
    + 1
    + ARM_JOINT_COUNT
    + (ACTION_DIMENSIONALITY - ARM_JOINT_COUNT)
)


def _horizontal_plane(*, entries: slice) -> slice:
    return slice(entries.start, entries.start + 2)


_PICTURE = StatePicture(
    position_groups=(
        PositionGroup(
            label="Gripper Position",
            entries=_horizontal_plane(entries=GRIPPER_POSITION_SLICE),
        ),
        PositionGroup(
            label="Peg Position", entries=_horizontal_plane(entries=PEG_POSITION_SLICE)
        ),
        PositionGroup(
            label="Hole Position",
            entries=_horizontal_plane(entries=HOLE_POSITION_SLICE),
        ),
    ),
    orientation_groups=(
        OrientationGroup(label="Peg Orientation", index=PEG_ORIENTATION_INDEX),
        OrientationGroup(label="Hole Orientation", index=HOLE_ORIENTATION_INDEX),
    ),
    horizontal_limits=(-0.25, 0.25),
    vertical_limits=(-0.45, 0.45),
    position_unit="m",
)


class PegInsertionStateSpace:
    @property
    def dimensionality(self) -> int:
        return STATE_DIMENSIONALITY

    @property
    def action_dimensionality(self) -> int:
        return ACTION_DIMENSIONALITY

    @property
    def network_input_dimensionality(self) -> int:
        return _NETWORK_INPUT_DIMENSIONALITY

    @property
    def picture(self) -> StatePicture:
        return _PICTURE

    def add(
        self,
        *,
        first_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        second_addend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return angle_arithmetic.add(
            first_addend_batch=first_addend_batch,
            second_addend_batch=second_addend_batch,
            angle_mask=self._angle_mask(like=first_addend_batch),
        )

    def subtract(
        self,
        *,
        minuend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        subtrahend_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return angle_arithmetic.subtract(
            minuend_batch=minuend_batch,
            subtrahend_batch=subtrahend_batch,
            angle_mask=self._angle_mask(like=minuend_batch),
        )

    def average(
        self,
        *,
        state_batch: torch.Tensor,  # (member_count, batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, state_dimensionality)
        return angle_arithmetic.average(
            state_batch=state_batch, angle_mask=self._angle_mask(like=state_batch)
        )

    def create_network_input(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, network_input_dimensionality)
        gripper_position_batch = state_batch[
            ..., GRIPPER_POSITION_SLICE
        ]  # (batch_size, 3)
        peg_position_batch = state_batch[..., PEG_POSITION_SLICE]  # (batch_size, 3)
        peg_orientation_batch = state_batch[
            ..., PEG_ORIENTATION_INDEX : PEG_ORIENTATION_INDEX + 1
        ]  # (batch_size, 1)
        hole_orientation_batch = state_batch[
            ..., HOLE_ORIENTATION_INDEX : HOLE_ORIENTATION_INDEX + 1
        ]  # (batch_size, 1)

        return torch.cat(
            tensors=[
                state_batch[..., JOINT_ANGLE_SLICE],
                state_batch[..., JOINT_VELOCITY_SLICE],
                gripper_position_batch,
                peg_position_batch - gripper_position_batch,
                state_batch[..., HOLE_POSITION_SLICE] - peg_position_batch,
                torch.sin(peg_orientation_batch),
                torch.cos(peg_orientation_batch),
                torch.sin(hole_orientation_batch),
                torch.cos(hole_orientation_batch),
                state_batch[..., PEG_HALF_SIZE_SLICE],
                state_batch[..., HOLE_RADIUS_INDEX : HOLE_RADIUS_INDEX + 1],
                action_batch[..., _ARM_ACTION_SLICE]
                - state_batch[..., _ARM_JOINT_SLICE],
                action_batch[..., _GRIPPER_ACTION_SLICE],
            ],
            dim=-1,
        )

    # ---- Private --------------------------------------------

    def _angle_mask(
        self, *, like: torch.Tensor
    ) -> torch.Tensor:  # (state_dimensionality,)
        return angle_arithmetic.create_mask(
            dimensionality=STATE_DIMENSIONALITY, angle_indices=ANGLE_INDICES, like=like
        )
