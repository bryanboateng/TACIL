"""The rules that decide how long a backward chain may grow.

A chain is only worth keeping while the model still knows what it is talking
about, and there is more than one way to say when that stops being true. All
of them come down to a number per chain, compared against one limit, so they
can be swapped and compared.

Four of the five share a shape. Each accepted step contributes an error term,
and a term contributed at depth d has to survive d more transitions before it
reaches the anchor, so it is stretched by whatever those transitions do to a
perturbation:

    tested = spent + amplification * term
    spent := tested,  amplification *= transition gain      (once accepted)

The rules differ in what the term measures and whether anything is carried
over at all:

    per_step_uncertainty            term = ensemble disagreement, nothing kept
    accumulated_uncertainty         term = ensemble disagreement, gain fixed at 1
    lipschitz_amplified_uncertainty term = ensemble disagreement, gain measured
    local_lipschitz_displacement    term = local steepness x step length, gain 1

The fifth, `replay`, does not accumulate anything. It replays the whole
candidate chain forward with each member and measures how far their endpoints
land from the anchor -- it asks the model directly what the finished episode
claims, instead of adding up what each step cost.
"""

from typing import Literal

import torch

from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.pusht_state import subtract_states

ChainErrorCriterion = Literal[
    "replay",
    "per_step_uncertainty",
    "accumulated_uncertainty",
    "lipschitz_amplified_uncertainty",
    "local_lipschitz_displacement",
]

_UNCERTAINTY_CRITERIA = frozenset(
    {
        "per_step_uncertainty",
        "accumulated_uncertainty",
        "lipschitz_amplified_uncertainty",
    }
)


class ChainErrorTracker:
    """Measures a candidate step, and remembers what earlier steps cost.

    `measure` never changes anything: the caller decides which steps are
    accepted and reports them back with `commit`, because a rejected step ends
    its chain and must not be paid for.
    """

    def __init__(
        self,
        *,
        criterion: ChainErrorCriterion,
        dynamics_model_ensemble: DynamicsModelEnsemble,
        anchor_count: int,
        device: torch.device,
    ) -> None:
        self._criterion = criterion
        self._dynamics_model_ensemble = dynamics_model_ensemble
        self._spent_batch = torch.zeros(anchor_count, device=device)
        self._amplification_batch = torch.ones(anchor_count, device=device)

    @torch.no_grad()
    def measure(
        self,
        *,
        anchor_indices: torch.Tensor,  # (active_count,)
        predecessor_state_batch: torch.Tensor,  # (active_count, state_dimensionality)
        predecessor_action_batch: torch.Tensor,  # (active_count, action_dimensionality)
        successor_state_batch: torch.Tensor,  # (active_count, state_dimensionality)
        anchor_state_batch: torch.Tensor,  # (active_count, state_dimensionality)
        earlier_action_batches: list[torch.Tensor],  # newest first
    ) -> torch.Tensor:  # (active_count,)
        if self._criterion == "replay":
            return self._dynamics_model_ensemble.replay_error(
                start_state_batch=predecessor_state_batch,
                action_batch_list=[predecessor_action_batch, *earlier_action_batches],
                end_state_batch=anchor_state_batch,
            )

        term_batch = self._measure_term(
            predecessor_state_batch=predecessor_state_batch,
            predecessor_action_batch=predecessor_action_batch,
            successor_state_batch=successor_state_batch,
        )
        if self._criterion == "per_step_uncertainty":
            return term_batch

        return (
            self._spent_batch[anchor_indices]
            + self._amplification_batch[anchor_indices] * term_batch
        )

    @torch.no_grad()
    def commit(
        self,
        *,
        anchor_indices: torch.Tensor,  # (accepted_count,)
        measured_error_batch: torch.Tensor,  # (accepted_count,)
        predecessor_state_batch: torch.Tensor,  # (accepted_count, state_dimensionality)
        predecessor_action_batch: torch.Tensor,  # (accepted_count, action_dimensionality)  # noqa: E501
    ) -> None:
        if self._criterion in ("replay", "per_step_uncertainty"):
            return

        self._spent_batch[anchor_indices] = measured_error_batch
        if self._criterion == "lipschitz_amplified_uncertainty":
            self._amplification_batch[anchor_indices] *= (
                self._dynamics_model_ensemble.local_transition_lipschitz_constants(
                    state_batch=predecessor_state_batch,
                    action_batch=predecessor_action_batch,
                )
            )

    # ---- Private --------------------------------------------

    def _measure_term(
        self,
        *,
        predecessor_state_batch: torch.Tensor,
        predecessor_action_batch: torch.Tensor,
        successor_state_batch: torch.Tensor,
    ) -> torch.Tensor:  # (active_count,)
        if self._criterion in _UNCERTAINTY_CRITERIA:
            return self._dynamics_model_ensemble.prediction_disagreement(
                state_batch=predecessor_state_batch,
                action_batch=predecessor_action_batch,
            )

        # local_lipschitz_displacement: how steep the model is here, times how
        # far this step moves. A first-order bound on what the step costs,
        # measured without ever consulting the members' spread.
        step_length_batch = torch.linalg.norm(
            self._dynamics_model_ensemble.state_normalizer.normalize_difference(
                difference_batch=subtract_states(
                    minuend_batch=successor_state_batch,
                    subtrahend_batch=predecessor_state_batch,
                )
            ),
            dim=-1,
        )  # (active_count,)
        return (
            self._dynamics_model_ensemble.local_lipschitz_constants(
                state_batch=predecessor_state_batch,
                action_batch=predecessor_action_batch,
            )
            * step_length_batch
        )
