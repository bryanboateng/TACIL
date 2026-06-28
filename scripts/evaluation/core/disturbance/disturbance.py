from gym_pusht.envs import PushTEnv

from scripts.evaluation.core.disturbance.action_noise import (
    ActionNoise,
    setup_action_noise,
)
from scripts.evaluation.core.disturbance.block_push import BlockPush, setup_block_push

Disturbance = BlockPush | ActionNoise


def setup_disturber(*, environment: PushTEnv, disturbance: Disturbance) -> None:
    match disturbance:
        case BlockPush():
            setup_block_push(environment=environment, config=disturbance)
        case ActionNoise():
            setup_action_noise(environment=environment, config=disturbance)
