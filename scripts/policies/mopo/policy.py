from dataclasses import dataclass

import numpy as np
import torch

from scripts.policies.mopo.buffer import Batch
from scripts.policies.mopo.dynamics import Dynamics
from scripts.policies.mopo.transitions import Transitions
from scripts.shared.normalization import MinMaxNormalizer, ZScoreNormalizer
from scripts.shared.policy_types import Policy, PolicyFactory

_STATISTIC_COUNT = 4


@dataclass(frozen=True)
class LearningStatistics:
    actor_loss: float
    critic_loss: float
    temperature_loss: float
    temperature: float


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


class CriticEnsemble(torch.nn.Module):
    def __init__(
        self,
        *,
        state_dimensionality: int,
        action_dimensionality: int,
        hidden_layer_dimensionalities: tuple[int, ...],
        critic_count: int,
    ) -> None:
        super().__init__()

        self.critic_count = critic_count
        dimensionalities = [
            state_dimensionality + action_dimensionality,
            *hidden_layer_dimensionalities,
            1,
        ]
        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()
        for input_dimensionality, output_dimensionality in zip(
            dimensionalities[:-1], dimensionalities[1:], strict=True
        ):
            weight, bias = _stack_linear_layers(
                input_dimensionality=input_dimensionality,
                output_dimensionality=output_dimensionality,
                critic_count=critic_count,
            )
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(
        self,
        input_batch: torch.Tensor,  # (batch_size, input_dimensionality)
    ) -> torch.Tensor:  # (critic_count, batch_size, 1)
        # Every critic sees the same input, so one copy is spread across the
        # critic axis without being duplicated in memory.
        value_batch = input_batch.unsqueeze(dim=0).expand(self.critic_count, -1, -1)

        # baddbmm is the batched product with the bias folded in.
        # This is faster than having two steps because it costs an extra intermediate on every layer.
        *hidden_layers, output_layer = zip(self.weights, self.biases, strict=True)
        for weight, bias in hidden_layers:
            value_batch = torch.relu(
                input=torch.baddbmm(input=bias, batch1=value_batch, batch2=weight)
            )

        output_weight, output_bias = output_layer
        return torch.baddbmm(
            input=output_bias, batch1=value_batch, batch2=output_weight
        )


class MopoPolicy:
    def __init__(
        self,
        *,
        actor: GaussianActor,
        actor_optimizer: torch.optim.Optimizer,
        critic: CriticEnsemble,
        critic_optimizer: torch.optim.Optimizer,
        target_critic: CriticEnsemble,
        target_critic_update_rate: float,
        log_temperature: torch.Tensor,
        temperature_optimizer: torch.optim.Optimizer,
        target_entropy: float,
        dynamics: Dynamics,
        discount_factor: float,
        compile_learning_step: bool,
        device: torch.device,
    ) -> None:
        self.actor = actor
        self.actor_optimizer = actor_optimizer

        self.critic = critic
        self.critic_optimizer = critic_optimizer

        self.target_critic = target_critic
        self.target_critic_update_rate = target_critic_update_rate

        self.log_temperature = log_temperature
        self.temperature_optimizer = temperature_optimizer
        self.target_entropy = target_entropy

        self.dynamics = dynamics
        self.discount_factor = discount_factor
        self.device = device

        self._critic_parameters: list[torch.Tensor] = list(critic.parameters())
        self._target_critic_parameters: list[torch.Tensor] = list(
            target_critic.parameters()
        )

        self._statistic_sums = torch.zeros(_STATISTIC_COUNT, device=device)
        self._learn_step = (
            torch.compile(model=self._run_learn_step, mode="reduce-overhead")
            if compile_learning_step
            else self._run_learn_step
        )

    def train(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic.eval()

    def sample_action(
        self,
        *,
        state_batch: torch.Tensor,  # (batch_size, state_dimensionality)
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_distribution = self.actor(state_batch)

        raw_action_batch = (
            action_distribution.mean if deterministic else action_distribution.rsample()
        )

        # The probability of an action is the multiplication of each action features' probability.
        # Because we want the log probability we multiply sum each action features' log probability.
        raw_log_probability_batch = action_distribution.log_prob(
            value=raw_action_batch
        ).sum(dim=-1, keepdim=True)  # (batch_size, 1)

        action_batch = torch.tanh(input=raw_action_batch)

        squash_correction_batch = torch.log(
            input=1.0 - action_batch.square() + 1e-6
        ).sum(dim=-1, keepdim=True)  # (batch_size, 1)

        return (
            action_batch,  # (batch_size, action_dimensionality)
            raw_log_probability_batch - squash_correction_batch,  # (batch_size, 1)
        )

    def imagine_rollout(
        self,
        *,
        start_state_batch: torch.Tensor,  # (start_count, state_dimensionality)
        length: int,
    ) -> tuple[Transitions, RolloutStatistics]:
        with torch.no_grad():
            state_batches: list[torch.Tensor] = []
            action_batches: list[torch.Tensor] = []
            next_state_batches: list[torch.Tensor] = []
            reward_batches: list[torch.Tensor] = []
            termination_batches: list[torch.Tensor] = []
            penalty_sums: list[torch.Tensor] = []

            state_batch = start_state_batch
            for _ in range(length):
                action_batch = self.sample_action(
                    state_batch=state_batch, deterministic=False
                )[0]

                imagined_step = self.dynamics.imagine_step(
                    state_batch=state_batch, action_batch=action_batch
                )

                state_batches.append(state_batch)
                action_batches.append(action_batch)
                next_state_batches.append(imagined_step.next_state_batch)
                reward_batches.append(imagined_step.reward_batch)
                termination_batches.append(imagined_step.termination_batch)
                penalty_sums.append(imagined_step.penalty_sum)

                continuation_mask = imagined_step.termination_batch[:, 0] == 0.0
                if not bool(continuation_mask.any()):
                    break

                state_batch = imagined_step.next_state_batch[continuation_mask]

            rollout = Transitions(
                states=torch.cat(tensors=state_batches, dim=0),
                actions=torch.cat(tensors=action_batches, dim=0),
                next_states=torch.cat(tensors=next_state_batches, dim=0),
                rewards=torch.cat(tensors=reward_batches, dim=0),
                terminations=torch.cat(tensors=termination_batches, dim=0),
            )

            transition_count = len(rollout.states)
            return rollout, RolloutStatistics(
                transition_count=transition_count,
                reward_mean=float(rollout.rewards.mean()),
                penalty_mean=float(torch.stack(tensors=penalty_sums).sum())
                / transition_count,
            )

    def learn(self, *, batch: Batch) -> None:
        # A replayable graph starts a fresh iteration on its own only when no
        # backward pass is still outstanding. This step carries its own
        # backward, so the boundary has to be named.
        torch.compiler.cudagraph_mark_step_begin()
        self._statistic_sums += self._learn_step(batch)

    def take_average_statistics(self, *, step_count: int) -> LearningStatistics:
        actor_loss, critic_loss, temperature_loss, temperature = (
            self._statistic_sums / step_count
        ).tolist()
        self._statistic_sums.zero_()
        return LearningStatistics(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            temperature_loss=temperature_loss,
            temperature=temperature,
        )

    # ---- Private --------------------------------------------

    def _run_learn_step(self, batch: Batch) -> torch.Tensor:  # (_STATISTIC_COUNT,)
        temperature = _read_temperature(log_temperature=self.log_temperature)

        target_value_batch = self._calculate_target_value(
            batch=batch, temperature=temperature
        )
        critic_loss = self._update_critic(
            batch=batch, target_value_batch=target_value_batch
        )
        actor_loss, temperature_loss = self._update_actor_and_temperature(
            batch=batch, temperature=temperature
        )

        self._update_target_critic()

        # The temperature carries a length of one, so every entry is reshaped
        # to a plain number before they are stacked.
        return torch.stack(
            tensors=[
                actor_loss.detach().reshape(()),
                critic_loss.reshape(()),
                temperature_loss.detach().reshape(()),
                _read_temperature(log_temperature=self.log_temperature).reshape(()),
            ]
        )

    def _calculate_target_value(
        self,
        *,
        batch: Batch,
        temperature: torch.Tensor,  # (1,)
    ) -> torch.Tensor:  # (batch_size, 1)
        with torch.no_grad():
            next_action_batch, next_log_probability_batch = self.sample_action(
                state_batch=batch.next_states, deterministic=False
            )
            next_value_batch = (
                self.target_critic(
                    torch.cat(tensors=[batch.next_states, next_action_batch], dim=-1)
                )
                .min(dim=0)
                .values
                - temperature * next_log_probability_batch
            )
            target_value_batch = (
                batch.rewards
                + self.discount_factor * (1.0 - batch.terminations) * next_value_batch
            )
        return target_value_batch

    def _update_critic(
        self,
        *,
        batch: Batch,
        target_value_batch: torch.Tensor,  # (batch_size, 1)
    ) -> torch.Tensor:  # ()
        """Moves the critic towards the mark and reports how far off it was."""
        value_batch = self.critic(
            torch.cat(tensors=[batch.states, batch.actions], dim=-1)
        )  # (critic_count, batch_size, 1)
        critic_loss_batch = (
            (value_batch - target_value_batch.unsqueeze(0)).square().mean(dim=(1, 2))
        )  # (critic_count,)

        # Summed for the step, which gives each critic the gradient it would
        # get on its own.
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss_batch.sum().backward()
        self.critic_optimizer.step()

        # Reported as the average, which is what one critic was off by.
        return critic_loss_batch.detach().mean()

    def _update_actor_and_temperature(
        self,
        *,
        batch: Batch,
        temperature: torch.Tensor,  # (1,)
    ) -> tuple[torch.Tensor, torch.Tensor]:  # each ()
        """Moves the actor and the temperature, and reports both losses.

        The actor is scored on what it would do now rather than on what was
        recorded, which is what lets it learn from data it did not produce.
        """
        sampled_action_batch, log_probability_batch = self.sample_action(
            state_batch=batch.states, deterministic=False
        )
        actor_loss = (
            -self.critic(
                torch.cat(tensors=[batch.states, sampled_action_batch], dim=-1)
            )
            .min(dim=0)
            .values.mean()
            + temperature * log_probability_batch.mean()
        )

        temperature_loss = -(
            self.log_temperature
            * (log_probability_batch.detach() + self.target_entropy)
        ).mean()
        # The actor and the temperature share no parameter, so one backward
        # pass over their sum gives both the gradient they would get apart.
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.temperature_optimizer.zero_grad(set_to_none=True)
        (actor_loss + temperature_loss).backward()
        self.actor_optimizer.step()
        self.temperature_optimizer.step()
        return actor_loss, temperature_loss

    def _update_target_critic(self) -> None:
        with torch.no_grad():
            torch._foreach_mul_(
                self._target_critic_parameters, 1.0 - self.target_critic_update_rate
            )
            torch._foreach_add_(
                self._target_critic_parameters,
                self._critic_parameters,
                alpha=self.target_critic_update_rate,
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


def _stack_linear_layers(
    *, input_dimensionality: int, output_dimensionality: int, critic_count: int
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    layers = [
        torch.nn.Linear(
            in_features=input_dimensionality, out_features=output_dimensionality
        )
        for _ in range(critic_count)
    ]
    return (
        torch.nn.Parameter(
            data=torch.stack(tensors=[layer.weight.T for layer in layers])
        ),  # (critic_count, input_dimensionality, output_dimensionality)
        torch.nn.Parameter(
            data=torch.stack(tensors=[layer.bias for layer in layers]).unsqueeze(dim=1)
        ),  # (critic_count, 1, output_dimensionality)
    )


def _read_temperature(*, log_temperature: torch.Tensor) -> torch.Tensor:  # (1,)
    return log_temperature.detach().exp().clamp(min=0.0, max=1.0)


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
