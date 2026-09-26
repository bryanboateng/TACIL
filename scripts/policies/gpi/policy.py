from dataclasses import dataclass, field

import numpy as np
import torch
from numpy import typing as npt

from scripts.environments.gpi_geometry import GpiGeometry
from scripts.policies.gpi.config import Config, ObservationNoiseConfig
from scripts.policies.gpi.demonstrations import Demonstrations
from scripts.shared.episodes import Episodes
from scripts.shared.policy_types import Policy, PolicyFactory


def create_geometric_policy_factory(
    *,
    gpi_geometry: GpiGeometry,
    episodes: Episodes,
    config: Config,
    device: torch.device,
) -> PolicyFactory:
    demonstrations = Demonstrations(
        gpi_geometry=gpi_geometry, episodes=episodes, device=device
    )

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
                    gpi_geometry=gpi_geometry,
                    demonstrations=demonstrations,
                    episode_state=episode_states[episode_index],
                    observation_noise_config=config.observation_noise,
                    steps_per_search=config.steps_per_search,
                    neighbour_count=config.neighbour_count,
                    action_smoothing=config.action_smoothing,
                    robot_distance_weight=config.robot_distance_weight,
                    environment_distance_weight=config.environment_distance_weight,
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
    previous_action: npt.NDArray[np.float32] | None = None


def _calculate_action(
    *,
    observation: torch.Tensor,
    gpi_geometry: GpiGeometry,
    demonstrations: Demonstrations,
    episode_state: _EpisodeState,
    observation_noise_config: ObservationNoiseConfig,
    steps_per_search: int,
    neighbour_count: int,
    action_smoothing: float,
    robot_distance_weight: float,
    environment_distance_weight: float,
    progression_weight: float,
    attraction_weight: float,
) -> npt.NDArray[np.float32]:  # (action_dimensionality,)
    normalized_observation = demonstrations.observation_normalizer.normalize(
        value_batch=observation.unsqueeze(dim=0)
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

    noisy_observation = demonstrations.observation_normalizer.denormalize(
        value_batch=noisy_normalized_observation.unsqueeze(dim=0)
    ).squeeze(dim=0)  # (observation_dimensionality,)

    step_indices, step_weights = _consume_next_step_indices(
        episode_state=episode_state,
        observation=noisy_observation,
        demonstrations=demonstrations,
        steps_per_search=steps_per_search,
        neighbour_count=neighbour_count,
        robot_distance_weight=robot_distance_weight,
        environment_distance_weight=environment_distance_weight,
    )

    object_frame_action_batch = torch.stack(
        tensors=[
            gpi_geometry.calculate_action(
                recorded_observation=recorded_observation,
                recorded_action=recorded_action,
                current_observation=noisy_observation,
                progression_weight=progression_weight,
                attraction_weight=attraction_weight,
            )
            for recorded_observation, recorded_action in (
                demonstrations.read_step(index=index) for index in step_indices
            )
        ]
    )  # (neighbour_count, action_dimensionality)

    object_frame_action = (
        object_frame_action_batch
        * torch.tensor(
            data=step_weights,
            device=object_frame_action_batch.device,
            dtype=object_frame_action_batch.dtype,
        ).unsqueeze(dim=-1)  # (neighbour_count, 1)
    ).sum(dim=0)  # (action_dimensionality,)

    action = (
        gpi_geometry.convert_actions_to_world_frame(
            observation_batch=observation.unsqueeze(dim=0),
            action_batch=object_frame_action.unsqueeze(dim=0),
        )
        .squeeze(dim=0)
        .cpu()
        .numpy()
    )  # (action_dimensionality,)

    smoothed_action = _smooth_action(
        action=action,
        previous_action=episode_state.previous_action,
        action_smoothing=action_smoothing,
    )  # (action_dimensionality,)
    episode_state.previous_action = smoothed_action
    return smoothed_action


def _smooth_action(
    *,
    action: npt.NDArray[np.float32],  # (action_dimensionality,)
    previous_action: npt.NDArray[np.float32] | None,  # (action_dimensionality,)
    action_smoothing: float,
) -> npt.NDArray[np.float32]:  # (action_dimensionality,)
    if action_smoothing <= 0.0 or previous_action is None:
        return action
    return (
        (action + action_smoothing * previous_action) / (1.0 + action_smoothing)
    ).astype(dtype=np.float32)


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


def _consume_next_step_indices(
    *,
    episode_state: _EpisodeState,
    observation: torch.Tensor,  # (observation_dimensionality,)
    demonstrations: Demonstrations,
    steps_per_search: int,
    neighbour_count: int,
    robot_distance_weight: float,
    environment_distance_weight: float,
) -> tuple[list[int], list[float]]:  # indices, blending weights
    if steps_per_search == 1:
        step_indices, distances = demonstrations.find_nearest_step_indices(
            query_observation=observation,
            excluded_step_indices=episode_state.used_step_indices,
            neighbour_count=neighbour_count,
            robot_distance_weight=robot_distance_weight,
            environment_distance_weight=environment_distance_weight,
        )
        episode_state.used_step_indices.add(step_indices[0])
        inverse_distances = 1.0 / (distances + 1e-8)  # (neighbour_count,)
        return step_indices, (inverse_distances / inverse_distances.sum()).tolist()

    if episode_state.remaining_step_count == 0:
        (start_index,), _ = demonstrations.find_nearest_step_indices(
            query_observation=observation,
            excluded_step_indices=episode_state.used_step_indices,
            neighbour_count=1,
            robot_distance_weight=robot_distance_weight,
            environment_distance_weight=environment_distance_weight,
        )
        episode_state.next_step_index = start_index
        episode_state.remaining_step_count = min(
            steps_per_search,
            demonstrations.get_remaining_steps_in_episode(step_index=start_index),
        )

    step_index = episode_state.next_step_index
    episode_state.next_step_index += 1
    episode_state.remaining_step_count -= 1
    episode_state.used_step_indices.add(step_index)
    return [step_index], [1.0]
