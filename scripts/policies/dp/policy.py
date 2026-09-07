import collections
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from scripts.policies.dp.config import HorizonLengthConfig
from scripts.policies.dp.network import ConditionalUnet1d
from scripts.policies.dp.noise_scheduler import create_noise_scheduler
from scripts.shared.policy_types import Policy, PolicyFactory


def create_diffusion_policy_factory(
    *,
    noise_prediction_network: ConditionalUnet1d,
    horizon_lengths: HorizonLengthConfig,
    noise_scheduler_name: Literal["ddim", "ddpm"],
    device: torch.device,
) -> PolicyFactory:
    noise_scheduler = create_noise_scheduler(name=noise_scheduler_name)

    noise_prediction_network.eval()

    def create_policy(*, episode_seeds: list[int]) -> Policy:
        episode_states = [
            _EpisodeState(
                noise_generator=torch.Generator(device=device).manual_seed(
                    episode_seed
                ),
                observation_deque=collections.deque(maxlen=horizon_lengths.observation),
                action_queue=collections.deque(),
            )
            for episode_seed in episode_seeds
        ]

        def plan_actions(*, episode_indices: list[int]) -> None:
            observation_history_batch = torch.from_numpy(
                np.stack(
                    arrays=[
                        np.stack(arrays=episode_states[episode_index].observation_deque)
                        for episode_index in episode_indices
                    ]
                )
            ).to(
                device=device, dtype=torch.float32
            )  # (planned_episode_count, observation_horizon_length, observation_dimensionality)

            condition_batch = noise_prediction_network.observation_normalizer.normalize(
                value_batch=observation_history_batch
            ).flatten(
                start_dim=1
            )  # (planned_episode_count, observation_horizon_length * observation_dimensionality)

            noise_generators = [
                episode_states[episode_index].noise_generator
                for episode_index in episode_indices
            ]
            noisy_action_sequence_batch = torch.cat(
                tensors=[
                    torch.randn(
                        size=(
                            1,
                            horizon_lengths.prediction,
                            noise_prediction_network.action_dimensionality,
                        ),
                        generator=noise_generator,
                        device=device,
                    )
                    for noise_generator in noise_generators
                ]
            )  # (planned_episode_count, prediction_horizon_length, action_dimensionality)

            for diffusion_timestep in noise_scheduler.timesteps:
                # The network predicts the noise at the scale it was drawn, not
                # the smaller amount that is present at this timestep.
                predicted_noise_batch = noise_prediction_network(
                    noisy_sample_batch=noisy_action_sequence_batch,
                    diffusion_timestep=diffusion_timestep,
                    condition_batch=condition_batch,
                )  # (planned_episode_count, prediction_horizon_length, action_dimensionality)

                # The scheduler recovers the clean sequence implied by the
                # prediction, then puts back the noise that belongs to the next
                # lower timestep.
                #
                # One noise_generator per episode, so the whole batch is stepped at
                # once and each episode still draws its own noise.
                # `step`'s signature allows only a single noise_generator but the `randn_tensor`
                # behind it takes the list.
                # noinspection PyTypeChecker
                step_output = noise_scheduler.step(
                    model_output=predicted_noise_batch,
                    timestep=diffusion_timestep,
                    sample=noisy_action_sequence_batch,
                    generator=noise_generators,  # ty:ignore[invalid-argument-type]
                )

                # noinspection PyUnresolvedReferences
                noisy_action_sequence_batch = step_output.prev_sample  # (planned_episode_count, prediction_horizon_length, action_dimensionality) # ty:ignore[unresolved-attribute]

            predicted_action_sequence_batch = (
                noise_prediction_network.action_normalizer.denormalize(
                    value_batch=noisy_action_sequence_batch
                )
                .cpu()
                .numpy()
            )  # (planned_episode_count, prediction_horizon_length, action_dimensionality)

            # step         0 1 2 3 4 5 6 7 8 9 A B C D E F
            # observation  o o
            # prediction   a a a a a a a a a a a a a a a a
            # execution      x x x x x x x x
            #                ^ execution_start_index
            execution_start_index = horizon_lengths.observation - 1
            for episode_index, predicted_action_sequence in zip(
                episode_indices, predicted_action_sequence_batch, strict=True
            ):
                episode_states[episode_index].action_queue.extend(
                    predicted_action_sequence[
                        execution_start_index : execution_start_index
                        + horizon_lengths.execution
                    ]
                )

        def create_actions(
            *, observations_by_episode_index: dict[int, np.ndarray]
        ) -> dict[int, np.ndarray]:
            for episode_index, observation in observations_by_episode_index.items():
                observation_deque = episode_states[episode_index].observation_deque
                observation_deque.append(observation)
                # Pad on first call so we always condition on
                # `horizon_length.observation` frames.
                while len(observation_deque) < horizon_lengths.observation:
                    observation_deque.appendleft(observation)

            # Re-plan only for the episodes whose previous action horizon is exhausted.
            exhausted_episode_indices = [
                episode_index
                for episode_index in observations_by_episode_index.keys()
                if len(episode_states[episode_index].action_queue) == 0
            ]
            if len(exhausted_episode_indices) > 0:
                with torch.no_grad():
                    plan_actions(episode_indices=exhausted_episode_indices)

            return {
                episode_index: episode_states[episode_index].action_queue.popleft()
                for episode_index in observations_by_episode_index.keys()
            }

        return create_actions

    return create_policy


# ---- Private --------------------------------------------


@dataclass(frozen=True)
class _EpisodeState:
    noise_generator: torch.Generator
    observation_deque: collections.deque
    action_queue: collections.deque
