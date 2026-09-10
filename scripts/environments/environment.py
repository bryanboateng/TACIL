from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.environments.maniskill.environment import ManiSkillEnvironment
    from scripts.environments.pusht.environment import PushTEnvironment

# One task the pipeline can train and score a policy on. Everything that knows
# how a state is laid out, how a simulator is built, and how an episode looks
# on screen lives in one of these.
#
# The imports above run only for the type checker, and the alias only names
# them when something asks for its value. A PushT run therefore does not load
# ManiSkill, which needs a graphics driver that is not on every machine.
type Environment = "PushTEnvironment | ManiSkillEnvironment"
