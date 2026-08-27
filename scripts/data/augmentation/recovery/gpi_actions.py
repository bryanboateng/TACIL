"""GPI's policy, run on invented states so its behaviour can be copied.

GPI needs no training: for any state it looks up a recorded step and builds an
action from what was recorded there. That makes it a policy defined
everywhere, and a strong prior for what a recovery action should look like --
built out of expert actions, so it moves the way the demonstrations move.

It is not, however, a function of the state alone. Each episode carries a plan
and a record of which steps it has already spent, and those are what push it
forward through a demonstration instead of letting it return to the step it
just used. Reading its formula without that memory gives a different, weaker
rule, so this keeps the same per-episode state as the policy does, one set per
chain.

The observation noise the policy adds is left out. It is there to shake the
policy loose at evaluation time; as a teacher it would only blur the labels.
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
        plan_length: int,
        chain_count: int,
        device: torch.device,
    ) -> None:
        # Built once: it normalises and holds every recorded step.
        self._demonstrations = Demonstrations(
            episode_store=expert_episode_store, device=device
        )
        self._progression_weight = progression_weight
        self._attraction_weight = attraction_weight
        self._plan_length = plan_length

        self._used_step_mask_batch = torch.zeros(
            chain_count,
            self._demonstrations.step_count,
            dtype=torch.bool,
            device=device,
        )
        self._next_step_index_batch = torch.zeros(
            chain_count, dtype=torch.long, device=device
        )
        self._plan_end_index_batch = torch.zeros(
            chain_count, dtype=torch.long, device=device
        )

    @torch.no_grad()
    def __call__(
        self,
        state_batch: torch.Tensor,  # (active_count, state_dimensionality)
        chain_index_batch: torch.Tensor,  # (active_count,)
    ) -> torch.Tensor:  # (active_count, action_dimensionality), world frame
        # Clamped like the policy does. The clamp sits inside the reference's
        # observation-noise step, so it applies at every inference even when
        # no noise is added, and an invented state off the recorded range
        # would otherwise be read differently here than there.
        normalized_state_batch = self._demonstrations.normalize_observation(
            observation_batch=state_batch
        ).clamp(min=-1.0, max=1.0)
        step_index_batch = self._take_next_step_indices(
            normalized_state_batch=normalized_state_batch,
            chain_index_batch=chain_index_batch,
        )
        recorded_observation_batch, recorded_action_batch = (
            self._demonstrations.read_steps(indices=step_index_batch)
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
        )  # (active_count, 2)

        return convert_actions_to_world_frame(
            observation_batch=state_batch,
            action_batch=self._demonstrations.denormalize_actions(
                action_batch=agent_target_batch
            ),
        )

    # ---- Private --------------------------------------------

    def _take_next_step_indices(
        self,
        *,
        normalized_state_batch: torch.Tensor,  # (active_count, state_dimensionality)
        chain_index_batch: torch.Tensor,  # (active_count,)
    ) -> torch.Tensor:  # (active_count,)
        next_step_index_batch = self._next_step_index_batch[chain_index_batch]
        plan_end_index_batch = self._plan_end_index_batch[chain_index_batch]

        # A chain only looks for a new step once its plan has run out; until
        # then it walks the demonstration it committed to.
        replanning_mask_batch = next_step_index_batch >= plan_end_index_batch
        replanning_positions = replanning_mask_batch.nonzero().squeeze(dim=1)
        if replanning_positions.numel() > 0:
            replanning_chain_indices = chain_index_batch[replanning_positions]
            used_step_mask_batch = self._used_step_mask_batch[replanning_chain_indices]

            # A chain that has spent every recorded step has nowhere left to
            # go; it starts over rather than failing the whole run.
            exhausted_mask_batch = used_step_mask_batch.all(dim=1)
            used_step_mask_batch[exhausted_mask_batch] = False
            self._used_step_mask_batch[replanning_chain_indices] = used_step_mask_batch

            start_index_batch = self._demonstrations.find_nearest_step_indices(
                query_observation_batch=normalized_state_batch[replanning_positions],
                excluded_step_mask_batch=used_step_mask_batch,
            )
            self._next_step_index_batch[replanning_chain_indices] = start_index_batch
            self._plan_end_index_batch[replanning_chain_indices] = (
                self._demonstrations.find_plan_end_indices(
                    start_index_batch=start_index_batch,
                    maximum_length=self._plan_length,
                )
            )
            next_step_index_batch = next_step_index_batch.clone()
            next_step_index_batch[replanning_positions] = start_index_batch

        self._used_step_mask_batch[chain_index_batch, next_step_index_batch] = True
        self._next_step_index_batch[chain_index_batch] = next_step_index_batch + 1
        return next_step_index_batch
