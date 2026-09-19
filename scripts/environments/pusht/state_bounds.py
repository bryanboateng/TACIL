from scripts.environments.pusht.observation import ARENA_SIZE, POSITION_SLICE
from scripts.environments.state_bounds import Bound


def create_bounds() -> tuple[Bound, ...]:
    return (
        Bound(
            entries=POSITION_SLICE, minimum=-float(ARENA_SIZE), maximum=2.0 * ARENA_SIZE
        ),
    )
