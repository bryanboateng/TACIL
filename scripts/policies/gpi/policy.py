from dataclasses import dataclass, field

import numpy as np
import torch
from numpy import typing as npt

from scripts.policies.gpi.config import Config, ObservationNoiseConfig
from scripts.policies.gpi.demonstrations import Demonstrations
from scripts.policies.gpi.geometry import convert_position_to_world_frame
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.policy_types import Policy, PolicyFactory
from scripts.shared.pusht_state import (
    AGENT_POSITION_SLICE,
    BLOCK_ORIENTATION_INDEX,
    BLOCK_POSITION_SLICE,
)


def create_geometric_policy_factory(
    *, episode_store: EpisodeStore, config: Config, device: torch.device
) -> PolicyFactory:
    demonstrations = Demonstrations(episode_store=episode_store, device=device)

    def create_policy(*, episode_seeds: list[int]) -> Policy:
        episode_states = [
            _EpisodeState(
                noise_generator=torch.Generator(device=device).manual_seed(
                    episode_seed
                ),
                noise_std=config.observation_noise.initial_std,
            )
            for episode_seed in episode_seeds
        ]

        def create_actions(
            *, observations_by_episode_index: dict[int, npt.NDArray[np.float32]]
        ) -> dict[int, npt.NDArray[np.float32]]:
            return {
                episode_index: _calculate_action(
                    observation=torch.from_numpy(observation).to(
                        device=device, dtype=torch.float32
                    ),
                    demonstrations=demonstrations,
                    episode_state=episode_states[episode_index],
                    observation_noise_config=config.observation_noise,
                    steps_per_search=config.steps_per_search,
                    progression_weight=config.progression_weight,
                    attraction_weight=config.attraction_weight,
                )
                for episode_index, observation in observations_by_episode_index.items()
            }

        return create_actions

    return create_policy


# ---- Private --------------------------------------------


@dataclass
class _EpisodeState:
    noise_generator: torch.Generator
    noise_std: float
    next_step_index: int = 0
    remaining_step_count: int = 0
    used_step_indices: set[int] = field(default_factory=set)


def _calculate_action(
    *,
    observation: torch.Tensor,
    demonstrations: Demonstrations,
    episode_state: _EpisodeState,
    observation_noise_config: ObservationNoiseConfig,
    steps_per_search: int,
    progression_weight: float,
    attraction_weight: float,
) -> npt.NDArray[np.float32]:  # (action_dimensionality,)
    normalized_observation = demonstrations.normalize_observation(
        observation_batch=observation.unsqueeze(dim=0)
    ).squeeze(dim=0)  # (observation_dimensionality,)

    noisy_normalized_observation = _add_observation_noise(
        observation=normalized_observation,
        noise_std=episode_state.noise_std,
        noise_generator=episode_state.noise_generator,
    )  # (observation_dimensionality,)

    episode_state.noise_std = max(
        episode_state.noise_std * observation_noise_config.decay_rate,
        observation_noise_config.minimum_std,
    )

    step_index = _consume_next_step_index(
        episode_state=episode_state,
        observation=noisy_normalized_observation,
        demonstrations=demonstrations,
        steps_per_search=steps_per_search,
    )

    recorded_observation, recorded_action = demonstrations.read_step(index=step_index)

    normalized_agent_target_position = _calculate_agent_target_position(
        recorded_position=recorded_observation[AGENT_POSITION_SLICE],  # (2,)
        recorded_action=recorded_action,  # (2,)
        current_position=noisy_normalized_observation[AGENT_POSITION_SLICE],  # (2,)
        progression_weight=progression_weight,
        attraction_weight=attraction_weight,
    )  # (2,)

    return (
        convert_position_to_world_frame(
            block_position_batch=observation[BLOCK_POSITION_SLICE].unsqueeze(dim=0),
            block_angle_batch=observation[BLOCK_ORIENTATION_INDEX].unsqueeze(dim=0),
            position_batch=demonstrations.denormalize_actions(
                action_batch=normalized_agent_target_position.unsqueeze(dim=0)
            ),
        )
        .squeeze(dim=0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def _add_observation_noise(
    *, observation: torch.Tensor, noise_std: float, noise_generator: torch.Generator
) -> torch.Tensor:
    if noise_std <= 0.0:
        return observation

    noise = (
        torch.randn(
            size=observation.shape, generator=noise_generator, device=observation.device
        )
        * noise_std
    )  # (observation_dimensionality,)
    return (observation + noise).clamp(min=-1.0, max=1.0)


def _consume_next_step_index(
    *,
    episode_state: _EpisodeState,
    observation: torch.Tensor,  # (observation_dimensionality,)
    demonstrations: Demonstrations,
    steps_per_search: int,
) -> int:
    if episode_state.remaining_step_count == 0:
        start_index = demonstrations.find_nearest_step_index(
            query_observation=observation,
            excluded_step_indices=episode_state.used_step_indices,
        )
        episode_state.next_step_index = start_index
        episode_state.remaining_step_count = demonstrations.count_steps_to_walk(
            start_index=start_index, steps_per_search=steps_per_search
        )

    step_index = episode_state.next_step_index
    episode_state.next_step_index += 1
    episode_state.remaining_step_count -= 1
    episode_state.used_step_indices.add(step_index)
    return step_index


def _calculate_agent_target_position(
    *,
    recorded_position: torch.Tensor,  # (2,)
    recorded_action: torch.Tensor,  # (2,)
    current_position: torch.Tensor,  # (2,)
    progression_weight: float,
    attraction_weight: float,
) -> torch.Tensor:  # (2,), normalized
    progression_flow = recorded_action - recorded_position  # (2,)
    attraction_flow = recorded_position - current_position  # (2,)
    return (
        current_position
        + progression_weight * progression_flow
        + attraction_weight * attraction_flow
    )  # (2,)
