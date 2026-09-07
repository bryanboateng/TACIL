import copy
from dataclasses import dataclass

import numpy as np
import torch
from numpy import typing as npt

from scripts.policies.mopo.buffer import Batch
from scripts.policies.mopo.dynamics import Dynamics
from scripts.policies.mopo.transitions import Transitions
from scripts.shared.normalization import MinMaxNormalizer, ZScoreNormalizer
from scripts.shared.policy_types import Policy, PolicyFactory


@dataclass(frozen=True)
class LearningStatistics:
    actor_loss: torch.Tensor
    critic_loss: torch.Tensor
    temperature_loss: torch.Tensor
    temperature: torch.Tensor


@dataclass(frozen=True)
class RolloutStatistics:
    transition_count: int
    reward_mean: float
    penalty_mean: float


class GaussianActor(torch.nn.Module):
    def __init__(
        self,
        *,
        state_dimensionality: int,
        action_dimensionality: int,
        hidden_layer_dimensionalities: tuple[int, ...],
    ) -> None:
        super().__init__()

        self.backbone = _create_backbone(
            input_dimensionality=state_dimensionality,
            hidden_layer_dimensionalities=hidden_layer_dimensionalities,
        )
        self.mean_layer = torch.nn.Linear(
            in_features=hidden_layer_dimensionalities[-1],
            out_features=action_dimensionality,
        )
        self.log_standard_deviation_layer = torch.nn.Linear(
            in_features=hidden_layer_dimensionalities[-1],
            out_features=action_dimensionality,
        )

    def forward(
        self,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
    ) -> torch.distributions.Normal:
        features = self.backbone(state_batch)
        return torch.distributions.Normal(
            loc=self.mean_layer(features),
            scale=(
                self.log_standard_deviation_layer(features)
                .clamp(min=-5.0, max=2.0)
                .exp()
            ),
            # Saves time. The clamp above already keeps it positive.
            validate_args=False,
        )


class Critic(torch.nn.Module):
    def __init__(
        self,
        *,
        state_dimensionality: int,
        action_dimensionality: int,
        hidden_layer_dimensionalities: tuple[int, ...],
    ) -> None:
        super().__init__()

        self.network = torch.nn.Sequential(
            _create_backbone(
                input_dimensionality=state_dimensionality + action_dimensionality,
                hidden_layer_dimensionalities=hidden_layer_dimensionalities,
            ),
            torch.nn.Linear(
                in_features=hidden_layer_dimensionalities[-1], out_features=1
            ),
        )

    def forward(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, 1)
        return self.network(torch.cat(tensors=[state_batch, action_batch], dim=-1))


class MopoPolicy:
    def __init__(
        self,
        *,
        actor: GaussianActor,
        first_critic: Critic,
        second_critic: Critic,
        actor_optimizer: torch.optim.Optimizer,
        first_critic_optimizer: torch.optim.Optimizer,
        second_critic_optimizer: torch.optim.Optimizer,
        log_temperature: torch.Tensor,
        temperature_optimizer: torch.optim.Optimizer,
        target_entropy: float,
        dynamics: Dynamics,
        discount_factor: float,
        target_update_rate: float,
        device: torch.device,
    ) -> None:
        self.actor = actor
        self.first_critic = first_critic
        self.second_critic = second_critic
        self.first_target_critic = copy.deepcopy(first_critic).eval()
        self.second_target_critic = copy.deepcopy(second_critic).eval()

        self.actor_optimizer = actor_optimizer
        self.first_critic_optimizer = first_critic_optimizer
        self.second_critic_optimizer = second_critic_optimizer

        self.log_temperature = log_temperature
        self.temperature_optimizer = temperature_optimizer
        self.target_entropy = target_entropy
        self.temperature = log_temperature.detach().exp()

        self.dynamics = dynamics
        self.discount_factor = discount_factor
        self.target_update_rate = target_update_rate
        self.device = device

    def train(self) -> None:
        self.actor.train()
        self.first_critic.train()
        self.second_critic.train()

    def eval(self) -> None:
        self.actor.eval()
        self.first_critic.eval()
        self.second_critic.eval()

    def sample_action(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, action_dimensionality), (batch_size, 1)
        action_distribution = self.actor(state_batch)

        raw_action_batch = (
            action_distribution.mean if deterministic else action_distribution.rsample()
        )
        action_batch = torch.tanh(input=raw_action_batch)

        log_probability_batch = action_distribution.log_prob(
            value=raw_action_batch
        ).sum(dim=-1, keepdim=True) - torch.log(
            input=1.0 - action_batch.square() + 1e-6
        ).sum(dim=-1, keepdim=True)

        return action_batch, log_probability_batch

    def imagine_rollout(
        self,
        *,
        start_state_batch: npt.NDArray[np.float32],  # (start_count, state_dim)
        length: int,
        generator: np.random.Generator,
    ) -> tuple[Transitions, RolloutStatistics]:
        with torch.no_grad():
            state_batches: list[npt.NDArray[np.float32]] = []
            action_batches: list[npt.NDArray[np.float32]] = []
            next_state_batches: list[npt.NDArray[np.float32]] = []
            reward_batches: list[npt.NDArray[np.float32]] = []
            termination_batches: list[npt.NDArray[np.float32]] = []
            penalty_means: list[float] = []

            state_batch = start_state_batch
            for _ in range(length):
                action_batch = (
                    self.sample_action(
                        state_batch=torch.as_tensor(
                            data=state_batch, device=self.device
                        ),
                        deterministic=False,
                    )[0]
                    .cpu()
                    .numpy()
                )  # (active_count, action_dimensionality)

                imagined_step = self.dynamics.imagine_step(
                    state_batch=state_batch,
                    action_batch=action_batch,
                    generator=generator,
                )

                state_batches.append(state_batch)
                action_batches.append(action_batch)
                next_state_batches.append(imagined_step.next_state_batch)
                reward_batches.append(imagined_step.reward_batch)
                termination_batches.append(imagined_step.termination_batch)
                penalty_means.append(imagined_step.penalty_mean)

                continuation_mask = imagined_step.termination_batch.flatten() == 0.0
                if not continuation_mask.any():
                    break

                state_batch = imagined_step.next_state_batch[continuation_mask]

            transitions = Transitions(
                states=np.concatenate(state_batches, axis=0),
                actions=np.concatenate(action_batches, axis=0),
                next_states=np.concatenate(next_state_batches, axis=0),
                rewards=np.concatenate(reward_batches, axis=0),
                terminations=np.concatenate(termination_batches, axis=0),
            )

            return transitions, RolloutStatistics(
                transition_count=len(transitions.states),
                reward_mean=float(transitions.rewards.mean()),
                penalty_mean=float(np.mean(penalty_means)),
            )

    def learn(self, *, batch: Batch) -> LearningStatistics:
        target_value_batch = self._calculate_target_value(batch)

        first_value_batch = self.first_critic(
            state_batch=batch.states, action_batch=batch.actions
        )
        second_value_batch = self.second_critic(
            state_batch=batch.states, action_batch=batch.actions
        )

        first_critic_loss = (first_value_batch - target_value_batch).square().mean()
        self.first_critic_optimizer.zero_grad()
        first_critic_loss.backward()
        self.first_critic_optimizer.step()

        second_critic_loss = (second_value_batch - target_value_batch).square().mean()
        self.second_critic_optimizer.zero_grad()
        second_critic_loss.backward()
        self.second_critic_optimizer.step()

        action_batch, log_probability_batch = self.sample_action(
            state_batch=batch.states, deterministic=False
        )
        actor_loss = (
            -torch.min(
                input=self.first_critic(
                    state_batch=batch.states, action_batch=action_batch
                ),
                other=self.second_critic(
                    state_batch=batch.states, action_batch=action_batch
                ),
            ).mean()
            + self.temperature * log_probability_batch.mean()
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        temperature_loss = -(
            self.log_temperature
            * (log_probability_batch.detach() + self.target_entropy)
        ).mean()
        self.temperature_optimizer.zero_grad()
        temperature_loss.backward()
        self.temperature_optimizer.step()
        self.temperature = torch.clamp(
            input=self.log_temperature.detach().exp(), min=0.0, max=1.0
        )

        self._update_target_critics()

        return LearningStatistics(
            actor_loss=actor_loss.detach(),
            critic_loss=(first_critic_loss.detach() + second_critic_loss.detach())
            / 2.0,
            temperature_loss=temperature_loss.detach(),
            temperature=self.temperature,
        )

    # ---- Private --------------------------------------------

    def _calculate_target_value(self, batch: Batch):
        with torch.no_grad():
            next_action_batch, next_log_probability_batch = self.sample_action(
                state_batch=batch.next_states, deterministic=False
            )
            next_value_batch = (
                torch.min(
                    input=self.first_target_critic(
                        state_batch=batch.next_states, action_batch=next_action_batch
                    ),
                    other=self.second_target_critic(
                        state_batch=batch.next_states, action_batch=next_action_batch
                    ),
                )
                - self.temperature * next_log_probability_batch
            )
            target_value_batch = (
                batch.rewards
                + self.discount_factor * (1.0 - batch.terminations) * next_value_batch
            )
            return target_value_batch

    def _update_target_critics(self) -> None:
        for critic, target_critic in (
            (self.first_critic, self.first_target_critic),
            (self.second_critic, self.second_target_critic),
        ):
            for parameter, target_parameter in zip(
                critic.parameters(), target_critic.parameters(), strict=True
            ):
                target_parameter.data.copy_(
                    other=target_parameter.data * (1.0 - self.target_update_rate)
                    + parameter.data * self.target_update_rate
                )


def create_mopo_policy_factory(
    *,
    policy: MopoPolicy,
    state_normalizer: ZScoreNormalizer,
    action_normalizer: MinMaxNormalizer,
    device: torch.device,
) -> PolicyFactory:
    policy.eval()

    def create_policy(*, episode_seeds: list[int]) -> Policy:
        del episode_seeds  # seeds unused: the evaluated policy acts deterministically

        def create_actions(
            *, observations_by_episode_index: dict[int, np.ndarray]
        ) -> dict[int, np.ndarray]:
            state_batch = state_normalizer.normalize(
                value_batch=torch.from_numpy(
                    np.stack(arrays=list(observations_by_episode_index.values()))
                ).float()
            ).to(device=device)

            with torch.no_grad():
                action_batch = action_normalizer.denormalize(
                    value_batch=policy.sample_action(
                        state_batch=state_batch, deterministic=True
                    )[0].cpu()
                ).numpy()

            return dict(
                zip(observations_by_episode_index.keys(), action_batch, strict=True)
            )

        return create_actions

    return create_policy


# ---- Private --------------------------------------------


def _create_backbone(
    *, input_dimensionality: int, hidden_layer_dimensionalities: tuple[int, ...]
) -> torch.nn.Module:
    layers: list[torch.nn.Module] = []
    dimensionalities = [input_dimensionality, *hidden_layer_dimensionalities]
    for in_features, out_features in zip(
        dimensionalities[:-1], dimensionalities[1:], strict=True
    ):
        layers.append(
            torch.nn.Linear(in_features=in_features, out_features=out_features)
        )
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)
