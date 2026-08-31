"""Finds places where one recorded episode can be left for another.

A junction is a pair of recorded steps from different episodes: a source step
whose state the run is at, and a target step whose action the run borrows. It
is worth taking when the model says the borrowed action lands where the target
episode goes next -- then the run can carry on inside the target episode, and
the result is a trajectory no single demonstration contains.

Only the junction is invented. Every state and every action in a stitched
episode is recorded expert data.
"""

from dataclasses import dataclass

import numpy as np
import torch
from numpy import typing as npt
from tqdm import tqdm

from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.pusht_state import subtract_states

SOURCE_CHUNK_SIZE = 512


@dataclass(frozen=True)
class Junctions:
    # The step the run is at when it switches, as a flat index.
    source_indices: npt.NDArray[np.int64]  # (junction_count,)
    # The step whose action is borrowed, as a flat index. The run continues
    # from this step's successor.
    target_indices: npt.NDArray[np.int64]  # (junction_count,)
    # How far the borrowed action lands from the state it should reach.
    errors: npt.NDArray[np.float32]  # (junction_count,)


@dataclass(frozen=True)
class JunctionDiagnostics:
    searchable_source_count: int
    candidate_count: int  # pairs close enough to be worth testing
    accepted_count: int  # candidates whose error stayed inside the limit


@torch.no_grad()
def find_junctions(
    *,
    episode_store: EpisodeStore,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    junction_radius: float,
    junction_error_limit: float,
) -> tuple[Junctions, JunctionDiagnostics]:
    device = next(dynamics_model_ensemble.parameters()).device

    episode_index_per_step = build_episode_index_per_step(
        episode_end_indices=episode_store.episode_end_indices
    )  # (total_timestep_count,)

    state_batch = torch.as_tensor(data=episode_store.states).to(device=device)
    action_batch = torch.as_tensor(data=episode_store.actions).to(device=device)
    episode_index_batch = torch.as_tensor(data=episode_index_per_step).to(device=device)

    # A target hands over its successor, so the last step of an episode cannot
    # be one: there is nothing after it to continue with.
    target_mask = torch.ones(len(episode_index_per_step), dtype=torch.bool)
    target_mask[episode_store.episode_end_indices - 1] = False
    target_indices = target_mask.nonzero().squeeze(dim=1).to(device=device)

    nearest_target_index_batch, nearest_distance_batch = _find_nearest_foreign_step(
        state_batch=state_batch,
        episode_index_batch=episode_index_batch,
        target_indices=target_indices,
        dynamics_model_ensemble=dynamics_model_ensemble,
    )

    candidate_mask = nearest_distance_batch <= junction_radius  # (source_count,)
    candidate_source_indices = candidate_mask.nonzero().squeeze(dim=1)
    candidate_target_indices = nearest_target_index_batch[candidate_source_indices]

    if candidate_source_indices.numel() == 0:
        return Junctions(
            source_indices=np.zeros(shape=0, dtype=np.int64),
            target_indices=np.zeros(shape=0, dtype=np.int64),
            errors=np.zeros(shape=0, dtype=np.float32),
        ), JunctionDiagnostics(
            searchable_source_count=int(state_batch.shape[0]),
            candidate_count=0,
            accepted_count=0,
        )

    # The test that matters: from the source state, does the borrowed action
    # reach what the target episode reaches next?
    error_batch = dynamics_model_ensemble.measure_replay_error(
        start_state_batch=state_batch[candidate_source_indices],
        action_batch_list=[action_batch[candidate_target_indices]],
        end_state_batch=state_batch[candidate_target_indices + 1],
    )  # (candidate_count,)

    accepted_mask = error_batch <= junction_error_limit  # (candidate_count,)

    return Junctions(
        source_indices=candidate_source_indices[accepted_mask].cpu().numpy(),
        target_indices=candidate_target_indices[accepted_mask].cpu().numpy(),
        errors=error_batch[accepted_mask].cpu().numpy().astype(np.float32),
    ), JunctionDiagnostics(
        searchable_source_count=int(state_batch.shape[0]),
        candidate_count=int(candidate_source_indices.numel()),
        accepted_count=int(accepted_mask.sum()),
    )


def build_episode_index_per_step(
    *,
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
) -> npt.NDArray[np.int64]:  # (total_timestep_count,)
    episode_lengths = np.diff(a=episode_end_indices, prepend=0)  # (episode_count,)
    return np.repeat(
        a=np.arange(len(episode_end_indices), dtype=np.int64), repeats=episode_lengths
    )


# ---- Private --------------------------------------------


def _find_nearest_foreign_step(
    *,
    state_batch: torch.Tensor,  # (total_timestep_count, state_dimensionality)
    episode_index_batch: torch.Tensor,  # (total_timestep_count,)
    target_indices: torch.Tensor,  # (target_count,)
    dynamics_model_ensemble: DynamicsModelEnsemble,
) -> tuple[torch.Tensor, torch.Tensor]:  # both (total_timestep_count,)
    """For every step, the closest step that belongs to a different episode.

    Distances are taken in the normalized state space, the same currency the
    chain error and the step distance use, and the orientation is wrapped
    rather than subtracted outright.
    """
    target_state_batch = state_batch[target_indices]
    target_episode_index_batch = episode_index_batch[target_indices]

    nearest_index_chunks: list[torch.Tensor] = []
    nearest_distance_chunks: list[torch.Tensor] = []

    for chunk_start in tqdm(
        iterable=range(0, state_batch.shape[0], SOURCE_CHUNK_SIZE),
        desc="junction search",
        unit="chunk",
    ):
        chunk_slice = slice(chunk_start, chunk_start + SOURCE_CHUNK_SIZE)
        chunk_state_batch = state_batch[chunk_slice]  # (chunk_size, state_dim)

        distance_batch = torch.linalg.norm(
            dynamics_model_ensemble.state_normalizer.normalize_difference(
                difference_batch=subtract_states(
                    minuend_batch=chunk_state_batch.unsqueeze(dim=1),
                    subtrahend_batch=target_state_batch.unsqueeze(dim=0),
                )
            ),
            dim=-1,
        )  # (chunk_size, target_count)

        # A step of the same episode is not a junction; it is just the episode.
        distance_batch = distance_batch.masked_fill(
            mask=(
                episode_index_batch[chunk_slice].unsqueeze(dim=1)
                == target_episode_index_batch.unsqueeze(dim=0)
            ),
            value=float("inf"),
        )

        nearest = distance_batch.min(dim=1)
        nearest_index_chunks.append(target_indices[nearest.indices])
        nearest_distance_chunks.append(nearest.values)

    return torch.cat(tensors=nearest_index_chunks), torch.cat(
        tensors=nearest_distance_chunks
    )
