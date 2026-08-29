"""Follows GPI from disturbed states, so its behaviour becomes training data.

GPI is a policy, not a dataset: it decides what to do at any state by looking
through the demonstrations. To hand that behaviour to a policy that cannot
look anything up, it has to be written down as (state, action) pairs.

Each chain starts at a disturbed copy of an expert state, and from there GPI
acts while the dynamics model carries the state forward. Because every state
follows from the action recorded before it, the chain is a trajectory a policy
can be trained on rather than a set of loose labels. A chain stops when it is
back among the demonstrations, or when its length runs out.

Nothing is filtered. GPI's actions are what is being copied, not a proposal to
be checked, and demanding that a chain arrive somewhere would throw away
exactly the states where GPI does something worth learning.
"""

from dataclasses import dataclass

import torch
from tqdm import tqdm

from scripts.data.augmentation.distillation.gpi_actions import GpiActionSource
from scripts.data.augmentation.tacil.create_data.episode_assembly import (
    SyntheticTrajectories,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.pusht_state import subtract_states


@dataclass(frozen=True)
class RolloutDiagnostics:
    chain_count: int
    already_on_manifold_count: int  # disturbance too weak to leave the data
    returned_count: int  # chains that came back among the demonstrations
    capped_count: int  # chains still away when their length ran out


@dataclass(frozen=True)
class DistilledChains:
    trajectories: SyntheticTrajectories
    # The recorded state each chain ended up nearest to, as a row of the
    # demonstration batch. Its continuation is what the episode carries on with.
    nearest_demonstration_index_batch: torch.Tensor  # (chain_count,)


@torch.no_grad()
def follow_gpi(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    start_state_batch: torch.Tensor,  # (chain_count, state_dimensionality)
    demonstration_state_batch: torch.Tensor,  # (demonstration_count, state_dimensionality)  # noqa: E501
    gpi_action_source: GpiActionSource,
    max_chain_length: int,
    on_manifold_tolerance: float,
) -> tuple[DistilledChains, RolloutDiagnostics]:
    chain_count, state_dimensionality = start_state_batch.shape
    device = start_state_batch.device

    state_batch = start_state_batch.clone()
    distance_batch, nearest_index_batch = _find_nearest_demonstration(
        dynamics_model_ensemble=dynamics_model_ensemble,
        state_batch=state_batch,
        demonstration_state_batch=demonstration_state_batch,
    )

    # A start that already sits on the demonstrations has nothing to recover
    # from, and its chain would only teach the policy to stand still.
    active_mask_batch = distance_batch > on_manifold_tolerance
    already_on_manifold_count = int((~active_mask_batch).sum())
    returned_count = 0

    step_count_batch = torch.zeros(chain_count, dtype=torch.long, device=device)
    state_batch_per_depth: list[torch.Tensor] = []
    action_batch_per_depth: list[torch.Tensor] = []

    progress_bar = tqdm(total=max_chain_length, desc="gpi step", unit="step")
    for depth in range(max_chain_length):
        active_indices = active_mask_batch.nonzero().squeeze(dim=1)
        if active_indices.numel() == 0:
            break
        progress_bar.set_postfix(active_chain_count=int(active_indices.numel()))

        active_state_batch = state_batch[active_indices]
        active_action_batch = gpi_action_source(active_state_batch, active_indices)

        # Recorded before stepping: this action is what the policy should
        # produce when it sees this state.
        depth_state_batch = torch.zeros(
            chain_count, state_dimensionality, device=device
        )
        depth_state_batch[active_indices] = active_state_batch
        state_batch_per_depth.append(depth_state_batch)

        depth_action_batch = torch.zeros(
            chain_count, active_action_batch.shape[1], device=device
        )
        depth_action_batch[active_indices] = active_action_batch
        action_batch_per_depth.append(depth_action_batch)

        step_count_batch[active_indices] = depth + 1
        state_batch[active_indices] = dynamics_model_ensemble.predict_next_state(
            state_batch=active_state_batch, action_batch=active_action_batch
        )

        distance_batch, index_batch = _find_nearest_demonstration(
            dynamics_model_ensemble=dynamics_model_ensemble,
            state_batch=state_batch[active_indices],
            demonstration_state_batch=demonstration_state_batch,
        )
        nearest_index_batch[active_indices] = index_batch

        returned_mask_batch = distance_batch <= on_manifold_tolerance
        returned_count += int(returned_mask_batch.sum())
        active_mask_batch[active_indices[returned_mask_batch]] = False
        progress_bar.update(n=1)
    progress_bar.close()

    return DistilledChains(
        trajectories=_assemble_trajectories(
            state_batch_per_depth=state_batch_per_depth,
            action_batch_per_depth=action_batch_per_depth,
            step_count_batch=step_count_batch,
        ),
        nearest_demonstration_index_batch=nearest_index_batch,
    ), RolloutDiagnostics(
        chain_count=chain_count,
        already_on_manifold_count=already_on_manifold_count,
        returned_count=returned_count,
        capped_count=int(active_mask_batch.sum()),
    )


# ---- Private --------------------------------------------


def _find_nearest_demonstration(
    *,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    state_batch: torch.Tensor,  # (query_count, state_dimensionality)
    demonstration_state_batch: torch.Tensor,  # (demonstration_count, state_dimensionality)  # noqa: E501
) -> tuple[torch.Tensor, torch.Tensor]:  # distances, indices
    distances: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    chunk_size = 256
    for start in range(0, state_batch.shape[0], chunk_size):
        distance_batch = torch.linalg.norm(
            dynamics_model_ensemble.state_normalizer.normalize_difference(
                difference_batch=subtract_states(
                    minuend_batch=state_batch[start : start + chunk_size].unsqueeze(
                        dim=1
                    ),
                    subtrahend_batch=demonstration_state_batch.unsqueeze(dim=0),
                )
            ),
            dim=-1,
        )  # (chunk_size, demonstration_count)
        chunk_distance_batch, chunk_index_batch = distance_batch.min(dim=1)
        distances.append(chunk_distance_batch)
        indices.append(chunk_index_batch)
    return torch.cat(tensors=distances), torch.cat(tensors=indices)


def _assemble_trajectories(
    *,
    state_batch_per_depth: list[torch.Tensor],
    action_batch_per_depth: list[torch.Tensor],
    step_count_batch: torch.Tensor,  # (chain_count,)
) -> SyntheticTrajectories:
    device = step_count_batch.device
    total_step_count = int(step_count_batch.sum().item())

    states = torch.empty(
        total_step_count, state_batch_per_depth[0].shape[1], device=device
    )
    actions = torch.empty(
        total_step_count, action_batch_per_depth[0].shape[1], device=device
    )
    start_index_batch = torch.cumsum(input=step_count_batch, dim=0) - step_count_batch

    for depth, (depth_state_batch, depth_action_batch) in enumerate(
        iterable=zip(state_batch_per_depth, action_batch_per_depth)
    ):
        chain_indices = (step_count_batch > depth).nonzero().squeeze(dim=1)
        if chain_indices.numel() == 0:
            continue
        write_index_batch = start_index_batch[chain_indices] + depth
        states[write_index_batch] = depth_state_batch[chain_indices]
        actions[write_index_batch] = depth_action_batch[chain_indices]

    # The chains were followed forward, so they are already in temporal order.
    return SyntheticTrajectories(
        states=states, actions=actions, step_count_batch=step_count_batch
    )
