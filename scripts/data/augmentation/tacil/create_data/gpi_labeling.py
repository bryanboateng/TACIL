"""Labels synthetic states with GPI's geometric action instead of the anchor
action.

TACIL builds off-manifold states by rolling backward from anchors, and normally
labels each with the (noisy) anchor action -- a constant, crude recovery signal.
This instead labels each synthetic state the way GPI acts: pull toward the
nearest demonstration step and follow its action. The state coverage stays
TACIL's; only the target action changes.
"""

import torch

from scripts.policies.gpi.demonstrations import Demonstrations
from scripts.policies.gpi.geometry import convert_actions_to_world_frame
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.pusht_state import AGENT_POSITION_SLICE


@torch.no_grad()
def relabel_synthetic_actions(
    *,
    synthetic_state_batch: torch.Tensor,  # (step_count, state_dimensionality)
    expert_episode_store: EpisodeStore,
    progression_weight: float,
    attraction_weight: float,
    device: torch.device,
) -> torch.Tensor:  # (step_count, action_dimensionality), world frame
    demonstrations = Demonstrations(episode_store=expert_episode_store, device=device)

    normalized_state_batch = demonstrations.normalize_observation(
        observation_batch=synthetic_state_batch
    )  # (step_count, state_dimensionality)

    nearest_step_index_batch = demonstrations.find_nearest_step_indices(
        query_observation_batch=normalized_state_batch
    )  # (step_count,)
    recorded_observation_batch, recorded_action_batch = demonstrations.read_steps(
        indices=nearest_step_index_batch
    )  # (step_count, state_dimensionality), (step_count, action_dimensionality)

    current_position_batch = normalized_state_batch[
        :, AGENT_POSITION_SLICE
    ]  # (step_count, 2)
    recorded_position_batch = recorded_observation_batch[
        :, AGENT_POSITION_SLICE
    ]  # (step_count, 2)

    # The same combination GPI applies at inference (see policy._calculate_agent
    # _target): the frame mix of a block-frame action with a world-frame position
    # is deliberate, matching the reference.
    agent_target_batch = (
        current_position_batch
        + progression_weight * (recorded_action_batch - recorded_position_batch)
        + attraction_weight * (recorded_position_batch - current_position_batch)
    )  # (step_count, 2)

    return convert_actions_to_world_frame(
        observation_batch=synthetic_state_batch,
        action_batch=demonstrations.denormalize_actions(
            action_batch=agent_target_batch
        ),
    )  # (step_count, action_dimensionality)
