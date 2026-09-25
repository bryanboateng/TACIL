import numpy as np
import torch

from scripts.policies.shared.observation_noise import disturb_half_of_the_batch
from scripts.shared.normalization import MinMaxNormalizer, ZScoreNormalizer
from scripts.shared.policy_types import Policy, PolicyFactory


class BehaviorCloningPolicyModel(torch.nn.Module):
    def __init__(
        self,
        *,
        observation_dimensionality: int,
        action_dimensionality: int,
        hidden_layer_dimensionality: int,
        observation_normalizer: ZScoreNormalizer,
        action_normalizer: MinMaxNormalizer,
    ) -> None:
        super().__init__()

        self.network = _create_network(
            observation_dimensionality=observation_dimensionality,
            action_dimensionality=action_dimensionality,
            hidden_layer_dimensionality=hidden_layer_dimensionality,
        )

        self.observation_normalizer = observation_normalizer
        self.action_normalizer = action_normalizer

    def forward(
        # *, keep key-word parameters optional because the model is exported and used in other places
        self,
        observation_batch: torch.Tensor,  # (batch_size, observation_dimensionality)
        observation_noise: float | None = None,
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        normalized_observation_batch = self.observation_normalizer.normalize(
            value_batch=observation_batch
        )  # (batch_size, observation_dimensionality)

        if observation_noise is not None:
            normalized_observation_batch = disturb_half_of_the_batch(
                value_batch=normalized_observation_batch, noise=observation_noise
            )

        normalized_action_batch = self.network(
            normalized_observation_batch
        )  # (batch_size, action_dimensionality)

        return self.action_normalizer.denormalize(
            value_batch=normalized_action_batch
        )  # (batch_size, action_dimensionality)


def create_behavior_cloning_policy_factory(
    *, policy_model: BehaviorCloningPolicyModel, device: torch.device
) -> PolicyFactory:
    policy_model.eval()

    def create_policy(*, episode_seeds: list[int]) -> Policy:
        del episode_seeds  # seeds unused: Behavior Cloning is deterministic

        def create_actions(
            *, observations_by_episode_index: dict[int, np.ndarray]
        ) -> dict[int, np.ndarray]:
            observation_batch = torch.from_numpy(
                np.stack(arrays=list(observations_by_episode_index.values()))
            ).to(
                device=device, dtype=torch.float32
            )  # (episode_count, observation_dimensionality)

            with torch.no_grad():
                action_batch = (
                    policy_model(observation_batch).cpu().numpy()
                )  # (episode_count, action_dimensionality)

            return dict(
                zip(observations_by_episode_index.keys(), action_batch, strict=True)
            )

        return create_actions

    return create_policy


# ---- Private --------------------------------------------


def _create_network(
    *,
    observation_dimensionality: int,
    action_dimensionality: int,
    hidden_layer_dimensionality: int,
) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(
            in_features=observation_dimensionality,
            out_features=hidden_layer_dimensionality,
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(
            in_features=hidden_layer_dimensionality,
            out_features=hidden_layer_dimensionality,
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(
            in_features=hidden_layer_dimensionality, out_features=action_dimensionality
        ),
        torch.nn.Tanh(),
    )
