from scripts.environments.peg_insertion.observation import (
    ANGLE_INDICES,
    GRIPPER_POSITION_SLICE,
    HOLE_POSITION_SLICE,
    HOLE_RADIUS_INDEX,
    PEG_HALF_SIZE_SLICE,
    PEG_POSITION_SLICE,
    STATE_DIMENSIONALITY,
)
from scripts.environments.state_space import Bound, StateSpace

_POSITION_LIMIT = 2.0


def create_state_space() -> StateSpace:
    return StateSpace(
        dimensionality=STATE_DIMENSIONALITY,
        angle_indices=ANGLE_INDICES,
        bounds=(
            *(
                Bound(
                    entries=entries, minimum=-_POSITION_LIMIT, maximum=_POSITION_LIMIT
                )
                for entries in (
                    GRIPPER_POSITION_SLICE,
                    PEG_POSITION_SLICE,
                    HOLE_POSITION_SLICE,
                )
            ),
            Bound(
                entries=slice(HOLE_RADIUS_INDEX, HOLE_RADIUS_INDEX + 1),
                minimum=0.0,
                maximum=float("inf"),
            ),
            Bound(
                entries=slice(PEG_HALF_SIZE_SLICE.start, PEG_HALF_SIZE_SLICE.start + 1),
                minimum=0.0,
                maximum=float("inf"),
            ),
        ),
    )
