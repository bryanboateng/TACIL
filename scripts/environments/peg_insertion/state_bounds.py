from scripts.environments.peg_insertion.observation import (
    HOLE_POSITION_SLICE,
    HOLE_RADIUS_INDEX,
    PEG_HALF_SIZE_SLICE,
    PEG_POSITION_SLICE,
    TCP_POSITION_SLICE,
)
from scripts.environments.state_bounds import Bound

_POSITION_LIMIT = 2.0


def create_bounds() -> tuple[Bound, ...]:
    return (
        *(
            Bound(entries=entries, minimum=-_POSITION_LIMIT, maximum=_POSITION_LIMIT)
            for entries in (TCP_POSITION_SLICE, PEG_POSITION_SLICE, HOLE_POSITION_SLICE)
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
    )
