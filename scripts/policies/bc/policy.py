import numpy as np
import torch

from scripts.shared.normalization import MinMaxNormalizer, ZScoreNormalizer
from scripts.shared.policy_types import Policy, PolicyFactory


class BehaviorCloningPolicyModel(torch.nn.Module):
    observation_normalizer: ZScoreNormalizer
    action_normalizer: MinMaxNormalizer

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

    def normalize_observation(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, observation_dimensionality)
    ) -> torch.Tensor:  # (batch_size, observation_dimensionality)
        return self.observation_normalizer.normalize(value_batch=observation_batch)

    def normalize_action(
        self,
        *,
        action_batch: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        return self.action_normalizer.normalize(value_batch=action_batch)

    def predict_action(
        self,
        *,
        observation_batch: torch.Tensor,  # (batch_size, observation_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        normalized_action_batch = self.network(
            self.normalize_observation(observation_batch=observation_batch)
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

        def act(*, observations: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
            observation_batch = torch.from_numpy(
                np.stack(arrays=list(observations.values()))
            ).to(
                device=device, dtype=torch.float32
            )  # (episode_count, observation_dimensionality)

            with torch.no_grad():
                action_batch = (
                    policy_model.predict_action(observation_batch=observation_batch)
                    .cpu()
                    .numpy()
                )  # (episode_count, action_dimensionality)

            return dict(zip(observations, action_batch, strict=True))

        return act

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
