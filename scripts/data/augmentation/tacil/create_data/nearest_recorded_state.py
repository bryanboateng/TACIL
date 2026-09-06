from dataclasses import dataclass

import torch

from scripts.data.augmentation.tacil.create_data.state_distance import (
    iterate_distances_to_recorded_states,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)


@dataclass(frozen=True)
class NearestRecordedStateDistances:
    state_distances: torch.Tensor  # (query_count,)
    action_distances: torch.Tensor  # (query_count,)


@torch.no_grad()
def measure_distances_to_nearest_recorded_state(
    *,
    query_state_batch: torch.Tensor,  # (query_count, state_dimensionality)
    query_action_batch: torch.Tensor,  # (query_count, action_dimensionality)
    recorded_state_batch: torch.Tensor,  # (recorded_count, state_dimensionality)
    recorded_action_batch: torch.Tensor,  # (recorded_count, action_dimensionality)
    dynamics_model_ensemble: DynamicsModelEnsemble,
    queries_are_the_recorded_states: bool,
) -> NearestRecordedStateDistances:
    """The nearest neighbour is found among the states alone. Its action is then
    looked up, so the two distances answer 'how far from the data is this, and
    does its label agree with the label the data carries there'.

    Searching states and actions together would pick the neighbour whose action
    already matches, which is the very thing being measured.
    """
    state_distance_batches: list[torch.Tensor] = []
    action_distance_batches: list[torch.Tensor] = []

    for chunk_start, distance_batch in iterate_distances_to_recorded_states(
        query_state_batch=query_state_batch,
        recorded_state_batch=recorded_state_batch,
        dynamics_model_ensemble=dynamics_model_ensemble,
    ):  # (chunk_size, recorded_count)
        if queries_are_the_recorded_states:
            chunk_indices = torch.arange(
                distance_batch.shape[0], device=distance_batch.device
            )
            distance_batch[chunk_indices, chunk_start + chunk_indices] = float("inf")

        nearest = distance_batch.min(dim=1)
        state_distance_batches.append(nearest.values)
        action_distance_batches.append(
            torch.linalg.norm(
                query_action_batch[chunk_start : chunk_start + distance_batch.shape[0]]
                - recorded_action_batch[nearest.indices],
                dim=-1,
            )
        )

    return NearestRecordedStateDistances(
        state_distances=torch.cat(tensors=state_distance_batches),
        action_distances=torch.cat(tensors=action_distance_batches),
    )
