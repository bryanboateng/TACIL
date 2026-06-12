import collections
from pathlib import Path

import numpy as np
import torch

from scripts.policies.dp.network import ConditionalUnet1d
from scripts.policies.dp.noise_scheduler import get_noise_scheduler
from scripts.shared.normalization_range import NormalizationRange
from scripts.shared.paths import PROJECT_ROOT_DIRECTORY_PATH
from scripts.shared.policy_types import Policy, PolicyFactory


def save_diffusion_policy(
    *,
    path: Path,
    noise_prediction_network: torch.nn.Module,
    observation_normalization_range: NormalizationRange,
    action_normalization_range: NormalizationRange,
    action_dimensionality: int,
    condition_dimensionality: int,
    hidden_channel_counts: tuple[int, ...],
    diffusion_timestep_count: int,
    observation_horizon_length: int,
    prediction_horizon_length: int,
    action_horizon_length: int,
) -> None:
    torch.save(
        obj={
            "noise_prediction_network_state_dict": noise_prediction_network.state_dict(),
            "observation_normalization_range": {
                "minimum": observation_normalization_range.minimum,
                "maximum": observation_normalization_range.maximum,
            },
            "action_normalization_range": {
                "minimum": action_normalization_range.minimum,
                "maximum": action_normalization_range.maximum,
            },
            "action_dimensionality": action_dimensionality,
            "condition_dimensionality": condition_dimensionality,
            "hidden_channel_counts": hidden_channel_counts,
            "diffusion_timestep_count": diffusion_timestep_count,
            "observation_horizon_length": observation_horizon_length,
            "prediction_horizon_length": prediction_horizon_length,
            "action_horizon_length": action_horizon_length,
        },
        f=path,
    )


def load_diffusion_policy(
    *, checkpoint_path: Path, device: torch.device
) -> PolicyFactory:
    checkpoint = torch.load(
        f=PROJECT_ROOT_DIRECTORY_PATH / checkpoint_path, map_location=device
    )

    observation_horizon_length = checkpoint["observation_horizon_length"]
    prediction_horizon_length = checkpoint["prediction_horizon_length"]
    action_horizon_length = checkpoint["action_horizon_length"]
    action_dimensionality = checkpoint["action_dimensionality"]
    diffusion_timestep_count = checkpoint["diffusion_timestep_count"]

    noise_scheduler = get_noise_scheduler(
        diffusion_timestep_count=diffusion_timestep_count
    )

    observation_normalization_range = NormalizationRange(
        **checkpoint["observation_normalization_range"]
    )
    action_normalization_range = NormalizationRange(
        **checkpoint["action_normalization_range"]
    )

    noise_prediction_network = ConditionalUnet1d(
        input_channel_count=action_dimensionality,
        condition_dimensionality=checkpoint["condition_dimensionality"],
        hidden_channel_counts=checkpoint["hidden_channel_counts"],
    ).to(device)
    noise_prediction_network.load_state_dict(
        state_dict=checkpoint["noise_prediction_network_state_dict"]
    )
    noise_prediction_network.eval()

    def make_policy(*, seed: int) -> Policy:
        diffusion_noise_generator = torch.Generator(device=device).manual_seed(seed)
        observation_deque: collections.deque = collections.deque(
            maxlen=observation_horizon_length
        )
        action_queue: collections.deque = collections.deque()

        def policy(*, observation: np.ndarray) -> np.ndarray:
            observation_deque.append(observation)
            # Pad on first call so we always condition on
            # `observation_horizon_length` frames.
            while len(observation_deque) < observation_horizon_length:
                observation_deque.appendleft(observation)

            # Re-plan only when the previous action horizon is exhausted.
            if action_queue:
                return action_queue.popleft()

            observation_history = torch.from_numpy(np.stack(observation_deque)).to(
                device=device, dtype=torch.float32
            )  # (observation_horizon_length, observation_dimensionality)
            normalized_observation_history = observation_normalization_range.normalize(
                values=observation_history
            )  # (observation_horizon_length, observation_dimensionality)
            condition_batch = normalized_observation_history.unsqueeze(dim=0).flatten(
                1
            )  # (1, observation_horizon_length * observation_dimensionality)

            with torch.no_grad():
                denoising_state = torch.randn(
                    size=(1, prediction_horizon_length, action_dimensionality),
                    device=device,
                    generator=diffusion_noise_generator,
                )  # (1, prediction_horizon_length, action_dimensionality)
                for diffusion_timestep in noise_scheduler.timesteps:
                    predicted_noise = noise_prediction_network(
                        noisy_sample_batch=denoising_state,
                        diffusion_timestep=diffusion_timestep,
                        condition_batch=condition_batch,
                    )  # (1, prediction_horizon_length, action_dimensionality)
                    step_output = noise_scheduler.step(
                        model_output=predicted_noise,
                        timestep=diffusion_timestep,
                        sample=denoising_state,
                        generator=diffusion_noise_generator,
                    )

                    # noinspection PyUnresolvedReferences
                    denoising_state = step_output.prev_sample  # (1, prediction_horizon_length, action_dimensionality) # ty:ignore[unresolved-attribute]

                # Loop has finished: denoising_state is now the predicted normalized action sequence.
                predicted_action_sequence = (
                    action_normalization_range.denormalize(values=denoising_state[0])
                    .cpu()
                    .numpy()
                )  # (prediction_horizon_length, action_dimensionality)

            # Queue the action-horizon slice that begins at the present control
            # step (skipping outputs aligned with past observations).
            action_horizon_start_index = observation_horizon_length - 1
            action_queue.extend(
                predicted_action_sequence[
                    action_horizon_start_index : action_horizon_start_index
                    + action_horizon_length
                ]
            )
            return action_queue.popleft()

        return policy

    return make_policy
