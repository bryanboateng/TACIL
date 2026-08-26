from dataclasses import dataclass, field

import numpy as np
import torch
from numpy import typing as npt

from scripts.environments.gpi_geometry import GpiGeometry
from scripts.policies.gpi.config import Config
from scripts.policies.gpi.demonstrations import Demonstrations
from scripts.shared.episode_store import EpisodeStore
from scripts.shared.policy_types import Policy, PolicyFactory


@dataclass
class _EpisodeState:
    noise_generator: torch.Generator
    noise_std: float
    next_step_index: int = 0
    plan_end_index: int = 0
    used_step_indices: set[int] = field(default_factory=set)


def create_geometric_policy_factory(
    *,
    episode_store: EpisodeStore,
    geometry: GpiGeometry,
    config: Config,
    device: torch.device,
) -> PolicyFactory:
    demonstrations = Demonstrations(
        episode_store=episode_store, geometry=geometry, device=device
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

        def policy(
            *, observations: dict[int, npt.NDArray[np.float32]]
        ) -> dict[int, npt.NDArray[np.float32]]:
            return {
                episode_index: _calculate_action(
                    observation=torch.from_numpy(observation).to(
                        device=device, dtype=torch.float32
                    ),
                    demonstrations=demonstrations,
                    geometry=geometry,
                    episode_state=episode_states[episode_index],
                    config=config,
                )
                for episode_index, observation in observations.items()
            }

        return policy

    return create_policy


# ---- Private --------------------------------------------


def _calculate_action(
    *,
    observation: torch.Tensor,
    demonstrations: Demonstrations,
    geometry: GpiGeometry,
    episode_state: _EpisodeState,
    config: Config,
) -> npt.NDArray[np.float32]:  # (action_dimensionality,)
    normalized_observation = demonstrations.normalize_observation(
        observation_batch=observation.unsqueeze(dim=0)
    ).squeeze(dim=0)  # (observation_dimensionality,)

    noisy_observation = _add_observation_noise(
        observation=normalized_observation,
        noise_std=episode_state.noise_std,
        noise_generator=episode_state.noise_generator,
    )  # (observation_dimensionality,)

    episode_state.noise_std = max(
        episode_state.noise_std * config.observation_noise.decay_rate,
        config.observation_noise.minimum_std,
    )

    step_index = _take_next_step_index(
        episode_state=episode_state,
        observation=noisy_observation,
        demonstrations=demonstrations,
        maximum_plan_length=config.plan_length,
    )

    recorded_observation, recorded_action = demonstrations.read_step(index=step_index)

    state_slice = geometry.agent_position_slice
    action_slice = geometry.action_position_slice

    # Only the entries that name a position are steered. Any others, such as a
    # gripper opening, are taken from the recorded step as they are. That is
    # what the reference does too, at GPI/gpi/policies/state.py:99, where the
    # whole recorded action is copied and only its first two entries are
    # overwritten.
    action = recorded_action.clone()  # (action_dimensionality,), normalized
    action[action_slice] = _calculate_agent_target(
        recorded_position=recorded_observation[state_slice],
        recorded_action=recorded_action[action_slice],
        current_position=noisy_observation[state_slice],
        progression_weight=config.progression_weight,
        attraction_weight=config.attraction_weight,
    )

    return _convert_to_world_action(
        action=action,
        observation=observation,
        demonstrations=demonstrations,
        geometry=geometry,
    )


def _add_observation_noise(
    *, observation: torch.Tensor, noise_std: float, noise_generator: torch.Generator
) -> torch.Tensor:
    noise = (
        torch.randn(
            size=observation.shape, generator=noise_generator, device=observation.device
        )
        * noise_std
    )  # (observation_dimensionality,)
    noisy_observation = (observation + noise).clamp(min=-1.0, max=1.0)
    return noisy_observation


def _take_next_step_index(
    *,
    episode_state: _EpisodeState,
    observation: torch.Tensor,  # (observation_dimensionality,)
    demonstrations: Demonstrations,
    maximum_plan_length: int,
) -> int:
    if episode_state.next_step_index >= episode_state.plan_end_index:
        start_index = demonstrations.find_nearest_step_index(
            query_observation=observation,
            excluded_step_indices=episode_state.used_step_indices,
        )
        episode_state.next_step_index = start_index
        episode_state.plan_end_index = demonstrations.find_plan_end_index(
            start_index=start_index, maximum_length=maximum_plan_length
        )

    step_index = episode_state.next_step_index
    episode_state.next_step_index += 1
    episode_state.used_step_indices.add(step_index)
    return step_index


def _calculate_agent_target(
    *,
    recorded_position: torch.Tensor,  # (agent_position_dimensionality,)
    recorded_action: torch.Tensor,  # (agent_position_dimensionality,)
    current_position: torch.Tensor,  # (agent_position_dimensionality,)
    progression_weight: float,
    attraction_weight: float,
) -> torch.Tensor:  # (agent_position_dimensionality,), normalized
    # The action is read in the block frame, the two positions in the world
    # frame. Do not "correct" this. It is what the reference does, at
    # GPI/gpi/policies/state.py:99, because its dataset converts only the
    # action (GPI/pusht/datasets.py:68). The paper converts the agent position
    # as well (equations 16 and 17), so the reference diverges from it. The
    # baseline is ported from the code, not from the paper.
    progression_flow = recorded_action - recorded_position
    attraction_flow = recorded_position - current_position
    return (
        current_position
        + progression_weight * progression_flow
        + attraction_weight * attraction_flow
    )


def _convert_to_world_action(
    *,
    action: torch.Tensor,  # (action_dimensionality,)
    observation: torch.Tensor,  # (state_dimensionality,)
    demonstrations: Demonstrations,
    geometry: GpiGeometry,
) -> npt.NDArray[np.float32]:  # (action_dimensionality,)
    return (
        geometry.convert_actions_to_world_frame(
            observation_batch=observation.unsqueeze(dim=0),
            action_batch=demonstrations.denormalize_actions(
                action_batch=action.unsqueeze(dim=0)
            ),
        )
        .squeeze(dim=0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )  # (action_dimensionality,)
