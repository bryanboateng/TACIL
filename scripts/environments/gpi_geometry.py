from scripts.environments.maniskill.gpi_geometry import ManiSkillGpiGeometry
from scripts.environments.pusht.gpi_geometry import PushTGpiGeometry

# What GPI needs to know about one environment.
#
# GPI reads a state, finds the nearest recorded state, and steers the agent
# along that recording. Three things about that are environment specific:
# where the agent sits inside a state, how near two states are, and which
# frame an action is written in. The reference implementation factors exactly
# these out as well, into its `EpisodeDataset` protocol (GPI/gpi/types.py).
type GpiGeometry = PushTGpiGeometry | ManiSkillGpiGeometry
