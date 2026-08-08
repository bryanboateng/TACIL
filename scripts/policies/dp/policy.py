import collections
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from scripts.policies.dp.network import ConditionalUnet1d
from scripts.policies.dp.noise_scheduler import get_ddim_scheduler, get_ddpm_scheduler
from scripts.shared.policy_types import Policy, PolicyFactory


@dataclass(frozen=True)
class _EpisodeState:
    noise_generator: torch.Generator
    observation_deque: collections.deque
    action_queue: collections.deque


def build_diffusion_policy_factory(
    *,
    noise_prediction_network: ConditionalUnet1d,
    action_dimensionality: int,
    observation_horizon_length: int,
    prediction_horizon_length: int,
    action_horizon_length: int,
    sampler: Literal["ddim", "ddpm"],
    device: torch.device,
) -> PolicyFactory:
    """Wrap a trained network so the evaluation can drive it.

    Takes the network as it is, holding whichever weights the caller put there.
    Nothing is written to disk and read back: a round trip through a file would
    only be a slower way of copying weights that are already in this process.
    """
    match sampler:
        case "ddim":
            noise_scheduler = get_ddim_scheduler()
        case "ddpm":
            noise_scheduler = get_ddpm_scheduler()

    noise_prediction_network.eval()

    def make_policy(*, seeds: list[int]) -> Policy:
        episode_states = [
            _EpisodeState(
                noise_generator=torch.Generator(device=device).manual_seed(seed),
                observation_deque=collections.deque(maxlen=observation_horizon_length),
                action_queue=collections.deque(),
            )
            for seed in seeds
        ]

        def plan(*, episode_indices: list[int]) -> None:
            observation_history_batch = torch.from_numpy(
                np.stack(
                    arrays=[
                        np.stack(arrays=episode_states[episode_index].observation_deque)
                        for episode_index in episode_indices
                    ]
                )
            ).to(
                device=device, dtype=torch.float32
            )  # (planning_count, observation_horizon_length, observation_dimensionality)

            condition_batch = noise_prediction_network.observation_normalizer.normalize(
                value_batch=observation_history_batch
            ).flatten(
                start_dim=1
            )  # (planning_count, observation_horizon_length * observation_dimensionality)

            noise_generators = [
                episode_states[episode_index].noise_generator
                for episode_index in episode_indices
            ]
            denoising_state = torch.cat(
                tensors=[
                    torch.randn(
                        size=(1, prediction_horizon_length, action_dimensionality),
                        device=device,
                        generator=generator,
                    )
                    for generator in noise_generators
                ]
            )  # (planning_count, prediction_horizon_length, action_dimensionality)

            for diffusion_timestep in noise_scheduler.timesteps:
                predicted_noise = noise_prediction_network(
                    noisy_sample_batch=denoising_state,
                    diffusion_timestep=diffusion_timestep,
                    condition_batch=condition_batch,
                )  # (planning_count, prediction_horizon_length, action_dimensionality)

                # One generator per episode, so the whole batch is stepped at
                # once and each episode still draws its own noise.
                # `step`'s signature allows only a single generator but the `randn_tensor`
                # behind it takes the list.
                # noinspection PyTypeChecker
                step_output = noise_scheduler.step(
                    model_output=predicted_noise,
                    timestep=diffusion_timestep,
                    sample=denoising_state,
                    generator=noise_generators,  # ty:ignore[invalid-argument-type]
                )

                # noinspection PyUnresolvedReferences
                denoising_state = step_output.prev_sample  # (planning_count, prediction_horizon_length, action_dimensionality) # ty:ignore[unresolved-attribute]

            # Loop has finished: denoising_state is now the predicted normalized action sequences.
            predicted_action_sequence_batch = (
                noise_prediction_network.action_normalizer.denormalize(
                    value_batch=denoising_state
                )
                .cpu()
                .numpy()
            )  # (planning_count, prediction_horizon_length, action_dimensionality)

            # Queue the action-horizon slice that begins at the present control
            # step (skipping outputs aligned with past observations).
            action_horizon_start_index = observation_horizon_length - 1
            for episode_index, predicted_action_sequence in zip(
                episode_indices, predicted_action_sequence_batch, strict=True
            ):
                episode_states[episode_index].action_queue.extend(
                    predicted_action_sequence[
                        action_horizon_start_index : action_horizon_start_index
                        + action_horizon_length
                    ]
                )

        def policy(*, observations: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
            for episode_index, observation in observations.items():
                observation_deque = episode_states[episode_index].observation_deque
                observation_deque.append(observation)
                # Pad on first call so we always condition on
                # `observation_horizon_length` frames.
                while len(observation_deque) < observation_horizon_length:
                    observation_deque.appendleft(observation)

            # Re-plan only for the episodes whose previous action horizon is exhausted.
            planning_episode_indices = [
                episode_index
                for episode_index in observations.keys()
                if len(episode_states[episode_index].action_queue) == 0
            ]
            if len(planning_episode_indices) > 0:
                with torch.no_grad():
                    plan(episode_indices=planning_episode_indices)

            return {
                episode_index: episode_states[episode_index].action_queue.popleft()
                for episode_index in observations.keys()
            }

        return policy

    return make_policy
