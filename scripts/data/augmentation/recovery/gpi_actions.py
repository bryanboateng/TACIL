"""GPI's action rule, evaluated on arbitrary states.

GPI needs no training: for any state it looks up the nearest recorded step and
builds an action from what was recorded there. That makes it a policy defined
everywhere, and a strong prior for what a recovery action should look like --
built out of expert actions, so it moves the way the demonstrations move.

Here it is used as a source of actions for invented states, not as the policy
under evaluation. The chains it produces are rolled forward through the
dynamics model, so state and action stay consistent.
"""

import torch

from scripts.policies.gpi.demonstrations import Demonstrations
from scripts.policies.gpi.geometry import convert_actions_to_world_frame
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.pusht_state import AGENT_POSITION_SLICE


class GpiActionSource:
    def __init__(
        self,
        *,
        expert_episode_store: EpisodeStore,
        progression_weight: float,
        attraction_weight: float,
        device: torch.device,
    ) -> None:
        # Built once: it normalises and holds every recorded step.
        self._demonstrations = Demonstrations(
            episode_store=expert_episode_store, device=device
        )
        self._progression_weight = progression_weight
        self._attraction_weight = attraction_weight

    @torch.no_grad()
    def __call__(
        self,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality), world frame
        normalized_state_batch = self._demonstrations.normalize_observation(
            observation_batch=state_batch
        )
        nearest_step_index_batch = self._demonstrations.find_nearest_step_indices(
            query_observation_batch=normalized_state_batch
        )  # (batch_size,)
        recorded_observation_batch, recorded_action_batch = (
            self._demonstrations.read_steps(indices=nearest_step_index_batch)
        )

        current_position_batch = normalized_state_batch[:, AGENT_POSITION_SLICE]
        recorded_position_batch = recorded_observation_batch[:, AGENT_POSITION_SLICE]

        # The same combination GPI applies at inference (see policy._calculate
        # _agent_target): the frame mix of a block-frame action with a
        # world-frame position is deliberate, matching the reference.
        agent_target_batch = (
            current_position_batch
            + self._progression_weight
            * (recorded_action_batch - recorded_position_batch)
            + self._attraction_weight
            * (recorded_position_batch - current_position_batch)
        )  # (batch_size, 2)

        return convert_actions_to_world_frame(
            observation_batch=state_batch,
            action_batch=self._demonstrations.denormalize_actions(
                action_batch=agent_target_batch
            ),
        )
