import torch

from scripts.environments.peg_insertion.observation import (
    ARM_ACTION_COUNT,
    TCP_POSITION_SLICE,
)

# TODO: GPI is not correct for pd_ee_delta_pose yet. Four things are wrong:
#
# 1. policies/gpi/policy.py computes `progression_flow = recorded_action -
#    recorded_position`, which assumes the action is an absolute target (PushT).
#    Our action IS the displacement, so the subtraction has to go and the
#    returned value has to be a delta instead of an absolute position. The GPI
#    paper works in displacements throughout (appendix C.1 rewrites PushT's
#    absolute action as `u = [x_target - x_a, y_target - y_a]`), so this is the
#    paper's own form, not a deviation.
#
# 2. The slices below do not line up: the agent position is 3 entries, the
#    action slice is 6. The actuated subspace of a 6-DoF pose action is the
#    whole TCP pose, so attraction needs a rotation term as well - the relative
#    rotation `recorded o current^-1` as an axis-angle vector, which is 3
#    entries and matches action[3:6]. Plain subtraction does not work there, so
#    the flow computation has to move out of policy.py and into this protocol.
#
# 3. policies/gpi/demonstrations.py fits a separate MinMaxNormalizer on the
#    actions. Progression then comes out of the action scale and attraction out
#    of the state scale, and the two mean different physical amounts, so adding
#    them is wrong. Measured on PushT the two scales differ by 3% (state span
#    482.7/478.0 vs action span 499.0/486.0), which is why the original gets
#    away with the same construction; here the action is a ~2.5 cm step while
#    the TCP position spans the workspace, so the mismatch is roughly 10x.
#    Fix: scale the translation part of the action by the TCP position scale
#    (recorded delta * 0.1 -> metres -> state units) instead of its own range.
#
# 4. The rotation part needs the controller bound (0.1 rad) as its scale, and
#    the gripper entry has no state counterpart and passes through untouched.
_AGENT_POSITION_SLICE = TCP_POSITION_SLICE
_ACTION_POSITION_SLICE = slice(0, ARM_ACTION_COUNT)


class PegInsertionGpiGeometry:
    @property
    def agent_position_slice(self) -> slice:
        return _AGENT_POSITION_SLICE

    @property
    def action_position_slice(self) -> slice:
        return _ACTION_POSITION_SLICE

    def convert_actions_to_object_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
        action_batch: torch.Tensor,  # (batch_size, ACTION_DIMENSIONALITY)
    ) -> torch.Tensor:  # (batch_size, ACTION_DIMENSIONALITY)
        del observation_batch
        return action_batch

    def convert_actions_to_world_frame(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, STATE_DIMENSIONALITY)
        action_batch: torch.Tensor,  # (batch_size, ACTION_DIMENSIONALITY)
    ) -> torch.Tensor:  # (batch_size, ACTION_DIMENSIONALITY)
        del observation_batch
        return action_batch

    def calculate_observation_distances(
        self,
        *,
        observation_batch: torch.Tensor,  # (step_count, STATE_DIMENSIONALITY)
        query_observation: torch.Tensor,  # (STATE_DIMENSIONALITY,)
    ) -> torch.Tensor:  # (step_count,)
        return torch.linalg.norm(observation_batch - query_observation, dim=1)
