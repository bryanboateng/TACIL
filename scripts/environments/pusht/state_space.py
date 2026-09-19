from scripts.environments.pusht.observation import ANGLE_INDICES, STATE_DIMENSIONALITY
from scripts.environments.state_space import StateSpace


def create_state_space() -> StateSpace:
    return StateSpace(dimensionality=STATE_DIMENSIONALITY, angle_indices=ANGLE_INDICES)
