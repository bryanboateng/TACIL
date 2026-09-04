"""Which recorded steps a backward chain is grown from.

Two rules. `AllAnchors` takes every step that has a long enough run-out, which
is what TACIL has always done. `PolicyVisitedAnchors` keeps the share of those
that lie closest to where a probe policy drifts, so the corrective tube is
centred on the region a policy occupies rather than on the expert's path.
"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import torch

from scripts.data.augmentation.tacil.create_data.policy_probe import (
    ProbeDiagnostics,
    roll_out_probe_policy,
    train_probe_policy,
)
from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.pusht_state import AGENT_POSITION_SLICE, add_states, subtract_states
from scripts.shared.transition_batch import TransitionBatch

ANCHOR_CHUNK_SIZE = 512


@dataclass(frozen=True)
class AllAnchors:
    """Every recorded step with a long enough run-out."""


@dataclass(frozen=True)
class PolicyVisitedAnchors:
    """Only the steps nearest to where a probe policy ends up."""

    # Share of the eligible steps to keep. At one this is `AllAnchors` with a
    # probe trained for nothing.
    keep_share: float = 0.5

    # How long the probe is followed through the dynamics model. Longer means
    # more drift and more model error in the estimate of where it goes.
    rollout_length: int = 32

    # Std of the block displacement that pushes each start off the recorded
    # path, as a fraction of each dimension's range in the expert data. Only
    # the block moves, as in the evaluation disturbances.
    disturbance_range_fraction: float = 0.15

    # Epochs for the stand-in policy. It only has to drift like a policy, not
    # to be a good one.
    probe_epoch_count: int = 50

    def __post_init__(self) -> None:
        if not 0.0 <= self.keep_share <= 1.0:
            raise ValueError(
                f"keep_share must be between 0 and 1, but it is {self.keep_share}."
            )
        for name, value in (
            ("keep_share", self.keep_share),
            ("rollout_length", self.rollout_length),
            ("disturbance_range_fraction", self.disturbance_range_fraction),
            ("probe_epoch_count", self.probe_epoch_count),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, but it is {value}.")


AnchorSelection: TypeAlias = AllAnchors | PolicyVisitedAnchors


@dataclass(frozen=True)
class AnchorSelectionDiagnostics:
    eligible_count: int
    selected_count: int
    probe: ProbeDiagnostics | None


def describe_selection(*, selection: AnchorSelection) -> str:
    if isinstance(selection, AllAnchors):
        return "all"
    return "policy_visited"


def get_anchor_indices(
    *,
    episode_end_indices: npt.NDArray[np.int64],  # (episode_count,)
    expert_part_length: int,
) -> npt.NDArray[np.int64]:  # (anchor_count,)
    episode_start_indices = np.concatenate(
        (np.zeros(shape=1, dtype=episode_end_indices.dtype), episode_end_indices[:-1])
    )  # (episode_count,)

    episode_lengths = episode_end_indices - episode_start_indices  # (episode_count,)
    if (episode_lengths < expert_part_length).any():
        raise ValueError(
            f"Found an episode shorter than the expert part length "
            f"({expert_part_length}); it could hold no anchor."
        )

    return np.concatenate(
        [
            np.arange(
                episode_start_index,
                episode_end_index - max(expert_part_length, 1) + 1,
                dtype=episode_end_indices.dtype,
            )
            for episode_start_index, episode_end_index in zip(
                episode_start_indices, episode_end_indices
            )
        ]
    )


def select_anchor_indices(
    *,
    selection: AnchorSelection,
    episode_store: EpisodeStore,
    dynamics_model_ensemble: DynamicsModelEnsemble,
    expert_part_length: int,
    seed: int,
) -> tuple[npt.NDArray[np.int64], AnchorSelectionDiagnostics]:
    eligible_anchor_indices = get_anchor_indices(
        episode_end_indices=episode_store.episode_end_indices,
        expert_part_length=expert_part_length,
    )  # (eligible_count,)

    if isinstance(selection, AllAnchors):
        return eligible_anchor_indices, AnchorSelectionDiagnostics(
            eligible_count=len(eligible_anchor_indices),
            selected_count=len(eligible_anchor_indices),
            probe=None,
        )

    device = next(dynamics_model_ensemble.parameters()).device

    probe_policy, final_training_loss = train_probe_policy(
        transitions=TransitionBatch(
            states=episode_store.states, actions=episode_store.actions
        ),
        epoch_count=selection.probe_epoch_count,
        seed=seed,
        device=device,
    )

    start_state_batch = _disturb_states(
        state_batch=torch.as_tensor(
            data=episode_store.states[eligible_anchor_indices]
        ).to(device=device),
        samples=episode_store.states,
        range_fraction=selection.disturbance_range_fraction,
    )  # (eligible_count, state_dimensionality)

    visited_state_batch = roll_out_probe_policy(
        probe_policy=probe_policy,
        dynamics_model_ensemble=dynamics_model_ensemble,
        start_state_batch=start_state_batch,
        rollout_length=selection.rollout_length,
    )  # (eligible_count * rollout_length, state_dimensionality)

    anchor_state_batch = torch.as_tensor(
        data=episode_store.states[eligible_anchor_indices]
    ).to(device=device)

    distance_to_probe_batch = _measure_distance_to_nearest(
        state_batch=anchor_state_batch,
        reference_state_batch=visited_state_batch,
        dynamics_model_ensemble=dynamics_model_ensemble,
    )  # (eligible_count,)

    # At least one anchor, or the rollout would have nothing to start from and
    # the run would stop with no labels.
    keep_count = max(
        1, int(round(number=selection.keep_share * len(eligible_anchor_indices)))
    )
    kept_positions = torch.argsort(input=distance_to_probe_batch)[:keep_count]

    # Sorted, so the anchors stay in recorded order and the assembled episodes
    # come out in the same order as under `AllAnchors`.
    selected_anchor_indices = np.sort(
        a=eligible_anchor_indices[kept_positions.cpu().numpy()]
    )

    return selected_anchor_indices, AnchorSelectionDiagnostics(
        eligible_count=len(eligible_anchor_indices),
        selected_count=len(selected_anchor_indices),
        probe=ProbeDiagnostics(
            start_count=int(start_state_batch.shape[0]),
            step_count=selection.rollout_length,
            final_training_loss=final_training_loss,
            median_drift=float(
                _measure_normalized_distances(
                    first_state_batch=visited_state_batch[
                        -start_state_batch.shape[0] :
                    ],
                    second_state_batch=start_state_batch,
                    dynamics_model_ensemble=dynamics_model_ensemble,
                )
                .median()
                .item()
            ),
        ),
    )


# ---- Private --------------------------------------------


def _disturb_states(
    *,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    samples: npt.NDArray[np.float32],  # (N, state_dimensionality)
    range_fraction: float,
) -> torch.Tensor:  # (batch_size, state_dimensionality)
    sample_batch = torch.as_tensor(data=samples).to(device=state_batch.device)
    disturbance_std = range_fraction * (
        sample_batch.max(dim=0).values - sample_batch.min(dim=0).values
    )  # (state_dimensionality,)
    # Only the block moves: the evaluation disturbances hit the block, and a
    # displaced agent needs no teacher to find its way back.
    disturbance_std[AGENT_POSITION_SLICE] = 0.0

    return add_states(
        first_addend_batch=state_batch,
        second_addend_batch=torch.randn_like(input=state_batch) * disturbance_std,
    )


def _measure_normalized_distances(
    *,
    first_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    second_state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    dynamics_model_ensemble: DynamicsModelEnsemble,
) -> torch.Tensor:  # (batch_size,)
    return torch.linalg.norm(
        dynamics_model_ensemble.state_normalizer.normalize_difference(
            difference_batch=subtract_states(
                minuend_batch=first_state_batch, subtrahend_batch=second_state_batch
            )
        ),
        dim=-1,
    )


@torch.no_grad()
def _measure_distance_to_nearest(
    *,
    state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    reference_state_batch: torch.Tensor,  # (reference_count, state_dimensionality)
    dynamics_model_ensemble: DynamicsModelEnsemble,
) -> torch.Tensor:  # (batch_size,)
    nearest_distance_chunks: list[torch.Tensor] = []

    for chunk_start in range(0, state_batch.shape[0], ANCHOR_CHUNK_SIZE):
        chunk_state_batch = state_batch[
            chunk_start : chunk_start + ANCHOR_CHUNK_SIZE
        ]  # (chunk_size, state_dimensionality)

        distance_batch = torch.linalg.norm(
            dynamics_model_ensemble.state_normalizer.normalize_difference(
                difference_batch=subtract_states(
                    minuend_batch=chunk_state_batch.unsqueeze(dim=1),
                    subtrahend_batch=reference_state_batch.unsqueeze(dim=0),
                )
            ),
            dim=-1,
        )  # (chunk_size, reference_count)

        nearest_distance_chunks.append(distance_batch.min(dim=1).values)

    return torch.cat(tensors=nearest_distance_chunks)
