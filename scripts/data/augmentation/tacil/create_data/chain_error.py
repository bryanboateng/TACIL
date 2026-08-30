"""The rules that decide how long a backward chain may grow.

A chain is only worth keeping while the model still knows what it is talking
about, and there is more than one way to say when that stops being true. Every
rule here comes down to one number per chain, compared against one limit, so
they can be swapped and compared.

Four of the five are a choice of two things.

**What a single step costs.** Two ways to put a number on how wrong one
invented step might be:

    uncertainty         measured -- how far the ensemble members' predictions
                        of this step lie apart. Where they agree the model has
                        been told what happens here; where they scatter it is
                        extrapolating.
    label_error_bound   proved -- CCIL's Theorem 4.2 bounds the error of a
                        generated label by the local Lipschitz constant times
                        the distance the label sits from the expert state, so
                        this is the model's local steepness times the length
                        of the step. Simplified: the theorem's bound uses the
                        constants of both the true and the learned dynamics and
                        adds the model's training error, while this uses the
                        learned model's local constant alone.

**How steps add up.** A cost paid at depth d still has to survive d more
transitions before the chain reaches its anchor:

    summed              costs are added up as they are
    compounded          costs are added up, and each older cost is stretched by
                        what the transitions since have done to it -- the
                        transition gain, measured on the model

A third way, charging only the newest step and forgetting the rest, was
tried and dropped: both of its variants came last of seven in the 2026-08-30
comparison. That both lost, whichever cost they charged, is the evidence that
keeping an account over the chain is what separates this from a one-step gate.

The fifth rule, `replay`, keeps no account at all. It replays the whole candidate
chain forward with each member and measures how far their endpoints land from
the anchor: it asks the model directly whether the finished episode holds up,
instead of adding up what its steps cost.
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

from scripts.data.augmentation.tacil.dynamics.dynamics_model_ensemble import (
    DynamicsModelEnsemble,
)
from scripts.shared.pusht_state import subtract_states

ChainErrorCost = Literal["uncertainty", "label_error_bound"]
ChainErrorAccumulation = Literal["summed", "compounded"]


@dataclass(frozen=True)
class ReplayCriterion:
    """Judge the finished chain, not its steps: replay it with every member
    and see how far their endpoints land from the anchor."""


@dataclass(frozen=True)
class AccountCriterion:
    """Charge every step and keep the running total."""

    cost: ChainErrorCost
    accumulation: ChainErrorAccumulation


ChainErrorCriterion: TypeAlias = ReplayCriterion | AccountCriterion


def describe_criterion(*, criterion: ChainErrorCriterion) -> str:
    """A short name for a rule, for logging and for naming a sweep."""
    if isinstance(criterion, ReplayCriterion):
        return "replay"
    return f"{criterion.accumulation}_{criterion.cost}"


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
        self._transition_gain_batch = torch.ones(anchor_count, device=device)

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
        if isinstance(self._criterion, ReplayCriterion):
            return self._dynamics_model_ensemble.measure_replay_error(
                start_state_batch=predecessor_state_batch,
                action_batch_list=[predecessor_action_batch, *earlier_action_batches],
                end_state_batch=anchor_state_batch,
            )

        step_cost_batch = self._measure_step_cost(
            predecessor_state_batch=predecessor_state_batch,
            predecessor_action_batch=predecessor_action_batch,
            successor_state_batch=successor_state_batch,
        )
        return (
            self._spent_batch[anchor_indices]
            + self._transition_gain_batch[anchor_indices] * step_cost_batch
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
        if isinstance(self._criterion, ReplayCriterion):
            return

        self._spent_batch[anchor_indices] = measured_error_batch
        if self._criterion.accumulation == "compounded":
            self._transition_gain_batch[anchor_indices] *= (
                self._dynamics_model_ensemble.measure_local_transition_lipschitz_constants(
                    state_batch=predecessor_state_batch,
                    action_batch=predecessor_action_batch,
                )
            )

    # ---- Private --------------------------------------------

    def _measure_step_cost(
        self,
        *,
        predecessor_state_batch: torch.Tensor,
        predecessor_action_batch: torch.Tensor,
        successor_state_batch: torch.Tensor,
    ) -> torch.Tensor:  # (active_count,)
        if self._criterion.cost == "uncertainty":
            return self._dynamics_model_ensemble.measure_prediction_disagreement(
                state_batch=predecessor_state_batch,
                action_batch=predecessor_action_batch,
            )

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
            self._dynamics_model_ensemble.measure_local_lipschitz_constants(
                state_batch=predecessor_state_batch,
                action_batch=predecessor_action_batch,
            )
            * step_length_batch
        )
