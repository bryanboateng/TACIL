from scripts.environments.pusht.state_space import PushTStateSpace
from scripts.environments.vector_state_space import VectorStateSpace

# How one environment's state vector behaves.
#
# Most of the pipeline treats a state as a plain vector. This breaks down
# where a state carries a rotation, because rotations do not add and subtract
# like numbers. Every place that adds, subtracts or averages states takes this
# type, which lets the rest of the pipeline stay free of environment
# knowledge.
type StateSpace = PushTStateSpace | VectorStateSpace
