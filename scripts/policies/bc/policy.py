from pathlib import Path

import numpy as np
import torch

from scripts.shared.normalization import MinMaxNormalizer
from scripts.shared.paths import PROJECT_ROOT_DIRECTORY_PATH
from scripts.shared.policy_types import Policy, PolicyFactory


def _create_network(
    *,
    observation_dimensionality: int,
    action_dimensionality: int,
    hidden_dimensionality: int,
) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(
            in_features=observation_dimensionality, out_features=hidden_dimensionality
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(
            in_features=hidden_dimensionality, out_features=hidden_dimensionality
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(
            in_features=hidden_dimensionality, out_features=action_dimensionality
        ),
    )


class BehaviorCloningPolicy(torch.nn.Module):
    observation_normalizer: MinMaxNormalizer
    action_normalizer: MinMaxNormalizer

    def __init__(
        self,
        *,
        observation_dimensionality: int,
        action_dimensionality: int,
        hidden_dimensionality: int,
    ) -> None:
        super().__init__()

        self.observation_dimensionality = observation_dimensionality
        self.action_dimensionality = action_dimensionality
        self.hidden_dimensionality = hidden_dimensionality

        self.network = _create_network(
            observation_dimensionality=observation_dimensionality,
            action_dimensionality=action_dimensionality,
            hidden_dimensionality=hidden_dimensionality,
        )

        # Placeholders, overwritten later
        self.observation_normalizer = MinMaxNormalizer(
            minimum=torch.zeros(observation_dimensionality),
            maximum=torch.ones(observation_dimensionality),
        )
        self.action_normalizer = MinMaxNormalizer(
            minimum=torch.zeros(action_dimensionality),
            maximum=torch.ones(action_dimensionality),
        )

    @torch.no_grad()
    def initialize_normalizers(
        self,
        *,
        observation_batch: torch.Tensor,  # (N, observation_dimensionality)
        action_batch: torch.Tensor,  # (N, action_dimensionality)
    ) -> None:
        self.observation_normalizer = MinMaxNormalizer.from_values(
            value_batch=observation_batch
        )
        self.action_normalizer = MinMaxNormalizer.from_values(value_batch=action_batch)

    def normalize_observation(
        self,
        *,
        observation: torch.Tensor,  # (batch_size, observation_dimensionality)
    ) -> torch.Tensor:  # (batch_size, observation_dimensionality)
        return self.observation_normalizer.normalize(value_batch=observation)

    def normalize_action(
        self,
        *,
        action: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        return self.action_normalizer.normalize(value_batch=action)

    def predict_action(
        self,
        *,
        observation: torch.Tensor,  # (batch_size, observation_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        """Raw observation -> raw action."""
        normalized_action = self.network(
            self.normalize_observation(observation=observation)
        )  # (batch_size, action_dimensionality)
        return self.action_normalizer.denormalize(
            value_batch=normalized_action
        )  # (batch_size, action_dimensionality)


def save_behavior_cloning_policy(*, policy: BehaviorCloningPolicy, path: Path) -> None:
    """Save the policy. The normalization ranges ride along inside state_dict()."""
    torch.save(
        obj={
            "policy_state_dict": policy.state_dict(),
            "observation_dimensionality": policy.observation_dimensionality,
            "action_dimensionality": policy.action_dimensionality,
            "hidden_dimensionality": policy.hidden_dimensionality,
        },
        f=path,
    )


def load_behavior_cloning_policy(
    *, checkpoint_path: Path, device: torch.device
) -> PolicyFactory:
    checkpoint = torch.load(
        f=PROJECT_ROOT_DIRECTORY_PATH / checkpoint_path, map_location=device
    )

    policy_model = BehaviorCloningPolicy(
        observation_dimensionality=checkpoint["observation_dimensionality"],
        action_dimensionality=checkpoint["action_dimensionality"],
        hidden_dimensionality=checkpoint["hidden_dimensionality"],
    ).to(device=device)
    policy_model.load_state_dict(state_dict=checkpoint["policy_state_dict"])
    policy_model.eval()

    def make_policy(*, seeds: list[int]) -> Policy:  # seeds unused: BC is deterministic
        del seeds

        def policy(*, observations: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
            observation_tensor = torch.from_numpy(
                np.stack(arrays=list(observations.values()))
            ).to(
                device=device, dtype=torch.float32
            )  # (running_episode_count, observation_dimensionality)
            with torch.no_grad():
                action_batch = (
                    policy_model.predict_action(observation=observation_tensor)
                    .cpu()
                    .numpy()
                )  # (running_episode_count, action_dimensionality)
            return dict(zip(observations, action_batch, strict=True))

        return policy

    return make_policy
