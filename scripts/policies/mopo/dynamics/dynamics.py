from collections.abc import Callable
from dataclasses import dataclass

import torch

from scripts.policies.mopo.dynamics.dynamics_ensemble import DynamicsModelEnsemble

# Decides, from a state alone, whether the episode is over.
type TerminationFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ImaginedStep:
    next_state_batch: torch.Tensor  # (batch_size, state_dimensionality)
    reward_batch: torch.Tensor  # (batch_size, 1)
    termination_batch: torch.Tensor  # (batch_size, 1)
    penalty_sum: torch.Tensor  # ()


class InputNormalizer:
    def __init__(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> None:
        self.mean = value_batch.mean(dim=0, keepdim=True)
        standard_deviation = value_batch.std(dim=0, unbiased=False, keepdim=True)
        standard_deviation[standard_deviation < 1e-12] = 1.0
        self.standard_deviation = standard_deviation

    def normalize(
        self,
        *,
        value_batch: torch.Tensor,  # (batch_size, dimensionality)
    ) -> torch.Tensor:
        return (
            value_batch - self.mean
        ) / self.standard_deviation  # (batch_size, dimensionality)


class Dynamics:
    def __init__(
        self,
        *,
        model_ensemble: DynamicsModelEnsemble,
        input_normalizer: InputNormalizer,
        termination_function: TerminationFunction,
        penalty_coefficient: float,
        device: torch.device,
    ) -> None:
        self.model_ensemble = model_ensemble
        self.input_normalizer = input_normalizer
        self.termination_function = termination_function
        self.penalty_coefficient = penalty_coefficient
        self.device = device

    def imagine_step(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> ImaginedStep:
        with torch.no_grad():
            input_batch = self.input_normalizer.normalize(
                value_batch=torch.cat(tensors=[state_batch, action_batch], dim=-1)
            )

            (
                mean_batch_per_member,  # (ensemble_size, batch_size, state_dimensionality + 1)
                log_variance_batch_per_member,  # (ensemble_size, batch_size, state_dimensionality + 1)
            ) = self.model_ensemble(input_batch)
            standard_deviation_batch_per_member = (
                0.5 * log_variance_batch_per_member
            ).exp()  # (ensemble_size, batch_size, state_dimensionality + 1)

            state_change_and_reward_batch_per_member = (
                mean_batch_per_member
                + torch.randn_like(input=mean_batch_per_member)
                * standard_deviation_batch_per_member
            )  # (ensemble_size, batch_size, state_dimensionality + 1)

            state_change_and_reward_batch = self._pick_from_random_elites(
                value_batch_per_member=state_change_and_reward_batch_per_member
            )  # (batch_size, state_dimensionality + 1)

            next_state_batch = state_batch + state_change_and_reward_batch[..., :-1]
            reward_batch = state_change_and_reward_batch[..., -1:]

            penalty_batch = (
                standard_deviation_batch_per_member.norm(dim=2)
                .amax(dim=0)
                .unsqueeze(dim=1)
            )  # (batch_size, 1)

            return ImaginedStep(
                next_state_batch=next_state_batch,
                reward_batch=reward_batch - self.penalty_coefficient * penalty_batch,
                termination_batch=self.termination_function(next_state_batch)
                .reshape(-1, 1)
                .float(),
                penalty_sum=penalty_batch.sum(),
            )

    # ---- Private --------------------------------------------

    def _pick_from_random_elites(
        self,
        *,
        value_batch_per_member: torch.Tensor,  # (ensemble_size, batch_size, value_dimensionality)
    ) -> torch.Tensor:  # (batch_size, value_dimensionality)
        batch_size = value_batch_per_member.shape[1]
        member_index_batch = self.model_ensemble.draw_elite_indices(count=batch_size)
        return value_batch_per_member[
            member_index_batch, torch.arange(end=batch_size, device=self.device)
        ]
