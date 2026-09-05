"""The actions a chain starts from, one per branch.

A chain follows one action backwards, with fresh noise at every step. Which
action that is decides where the chain goes, so growing several chains from the
same anchor under different actions turns a single thread back through the data
into a cone around it. That cone is the thing a policy needs when it has drifted
off the demonstration, and one thread is not.

Two ways to fill the branches:

    anchor_noise        every branch starts from the anchor's own recorded
                        action, which is what TACIL has always done. The
                        branches still differ, because the noise is drawn again
                        for each of them, but they all vary around one action.
    neighbour_actions   each branch starts from the recorded action of a
                        nearby state in a DIFFERENT episode -- what another
                        demonstration did when it stood roughly here.

The anchor's own episode is left out, and that exclusion is the whole point.
Measured on the recorded set, 85% of a state's sixteen nearest neighbours lie
in its own episode, a median of four steps away, and their actions sit closer
to the anchor's action than the step noise does. Without the exclusion this
source would be a more expensive way of doing nothing.

Retrieving a neighbour's action is what the geometric baseline does at every
inference step. Doing it here, across demonstrations, writes the same move into
the training data instead.
"""

from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import torch

from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.pusht_state import subtract_states

ActionSource: TypeAlias = Literal["anchor_noise", "neighbour_actions"]

# States compared per chunk. The distance to every recorded state is measured
# at once, so the widest tensor holds chunk x recorded x state, and this keeps
# that inside a few hundred megabytes on the largest budget.
NEIGHBOUR_CHUNK_SIZE = 256


def build_branch_actions(
    *,
    action_source: ActionSource,
    branch_count: int,
    episode_store: EpisodeStore,
    anchor_indices: npt.NDArray[np.int64],  # (anchor_count,)
    dynamics_model_ensemble: DynamicsModelEnsemble,
    device: torch.device,
) -> torch.Tensor:  # (anchor_count, branch_count, action_dimensionality)
    anchor_action_batch = torch.as_tensor(
        data=episode_store.actions[anchor_indices]
    ).to(device=device)  # (anchor_count, action_dimensionality)

    if action_source == "anchor_noise" or branch_count == 1:
        return anchor_action_batch.unsqueeze(dim=1).expand(
            -1, branch_count, -1
        )  # (anchor_count, branch_count, action_dimensionality)

    recorded_state_batch = torch.as_tensor(data=episode_store.states).to(device=device)
    recorded_action_batch = torch.as_tensor(data=episode_store.actions).to(
        device=device
    )
    episode_index_per_step = torch.as_tensor(
        data=_build_episode_index_per_step(
            episode_end_indices=episode_store.episode_end_indices
        )
    ).to(device=device)  # (recorded_count,)

    # How many states are open to an anchor at worst: its own episode is out of
    # reach, and the longest episode is the most that can cost.
    episode_lengths = np.diff(a=episode_store.episode_end_indices, prepend=0)
    reachable_count = int(recorded_state_batch.shape[0] - episode_lengths.max())
    if reachable_count < 1:
        raise ValueError(
            "Every recorded state lies in the anchor's own episode, so there is "
            "no other demonstration to retrieve an action from. The neighbour "
            "source needs at least two episodes."
        )

    neighbour_index_batch = _find_nearest_recorded_states(
        query_state_batch=torch.as_tensor(data=episode_store.states[anchor_indices]).to(
            device=device
        ),
        recorded_state_batch=recorded_state_batch,
        query_episode_index_batch=episode_index_per_step[
            torch.as_tensor(data=anchor_indices).to(device=device)
        ],
        recorded_episode_index_batch=episode_index_per_step,
        neighbour_count=min(branch_count, reachable_count),
        dynamics_model_ensemble=dynamics_model_ensemble,
    )  # (anchor_count, neighbour_count)

    branch_action_batch = recorded_action_batch[neighbour_index_batch]
    # Fewer recorded states than branches is only possible on a tiny dataset;
    # the missing branches repeat the last neighbour rather than failing.
    if branch_action_batch.shape[1] < branch_count:
        branch_action_batch = torch.cat(
            tensors=[
                branch_action_batch,
                branch_action_batch[:, -1:].expand(
                    -1, branch_count - branch_action_batch.shape[1], -1
                ),
            ],
            dim=1,
        )
    return branch_action_batch


@torch.no_grad()
def _find_nearest_recorded_states(
    *,
    query_state_batch: torch.Tensor,  # (query_count, state_dimensionality)
    recorded_state_batch: torch.Tensor,  # (recorded_count, state_dimensionality)
    query_episode_index_batch: torch.Tensor,  # (query_count,)
    recorded_episode_index_batch: torch.Tensor,  # (recorded_count,)
    neighbour_count: int,
    dynamics_model_ensemble: DynamicsModelEnsemble,
) -> torch.Tensor:  # (query_count, neighbour_count)
    """The nearest recorded states outside the query's own episode, nearest
    first.

    Distances are measured in the normalized state space, which is what makes
    a displacement of the block comparable to a displacement of the agent.
    """
    neighbour_index_chunks: list[torch.Tensor] = []

    for chunk_start in range(0, query_state_batch.shape[0], NEIGHBOUR_CHUNK_SIZE):
        chunk_state_batch = query_state_batch[
            chunk_start : chunk_start + NEIGHBOUR_CHUNK_SIZE
        ]  # (chunk_size, state_dimensionality)

        distance_batch = torch.linalg.norm(
            dynamics_model_ensemble.state_normalizer.normalize_difference(
                difference_batch=subtract_states(
                    minuend_batch=chunk_state_batch.unsqueeze(dim=1),
                    subtrahend_batch=recorded_state_batch.unsqueeze(dim=0),
                )
            ),
            dim=-1,
        )  # (chunk_size, recorded_count)

        # The anchor's own episode is put out of reach rather than dropped, so
        # the row keeps its shape and `topk` can still take the nearest ones.
        distance_batch = distance_batch.masked_fill(
            mask=recorded_episode_index_batch.unsqueeze(dim=0)
            == query_episode_index_batch[
                chunk_start : chunk_start + NEIGHBOUR_CHUNK_SIZE
            ].unsqueeze(dim=1),
            value=torch.inf,
        )

        neighbour_index_chunks.append(
            torch.topk(
                input=distance_batch, k=neighbour_count, dim=1, largest=False
            ).indices
        )

    return torch.cat(tensors=neighbour_index_chunks, dim=0)


def _build_episode_index_per_step(
    *,
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
) -> npt.NDArray[np.int64]:  # (total_timestep_count,)
    """Which episode each recorded step belongs to."""
    return np.repeat(
        a=np.arange(len(episode_end_indices)),
        repeats=np.diff(a=episode_end_indices, prepend=0),
    )
