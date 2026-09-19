from scripts.environments.pusht.observation import (
    ANGLE_INDICES,
    ARENA_SIZE,
    POSITION_SLICE,
    STATE_DIMENSIONALITY,
)
from scripts.environments.state_space import Bound, StateSpace


def create_state_space() -> StateSpace:
    return StateSpace(
        dimensionality=STATE_DIMENSIONALITY,
        angle_indices=ANGLE_INDICES,
        bounds=(
            Bound(
                entries=POSITION_SLICE,
                minimum=-float(ARENA_SIZE),
                maximum=2.0 * ARENA_SIZE,
            ),
        ),
    )
