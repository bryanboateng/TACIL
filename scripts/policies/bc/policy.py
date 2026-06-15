from pathlib import Path

import numpy as np
import torch

from scripts.shared.normalization_range import NormalizationRange
from scripts.shared.paths import PROJECT_ROOT_DIRECTORY_PATH
from scripts.shared.policy_types import Policy, PolicyFactory


def build_behavior_cloning_network(
    *,
    observation_dimensionality: int,
    action_dimensionality: int,
    hidden_dimensionality: int,
) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(observation_dimensionality, hidden_dimensionality),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dimensionality, hidden_dimensionality),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dimensionality, action_dimensionality),
    )


class BehaviorCloningPolicy(torch.nn.Module):
    _observation_minimum: torch.Tensor  # (observation_dimensionality,)
    _observation_maximum: torch.Tensor  # (observation_dimensionality,)
    _action_minimum: torch.Tensor  # (action_dimensionality,)
    _action_maximum: torch.Tensor  # (action_dimensionality,)

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

        self.network = build_behavior_cloning_network(
            observation_dimensionality=observation_dimensionality,
            action_dimensionality=action_dimensionality,
            hidden_dimensionality=hidden_dimensionality,
        )

        # Placeholders; overwritten by compute_normalization_ranges() or load_state_dict().
        self.register_buffer(
            "_observation_minimum", torch.zeros(observation_dimensionality)
        )
        self.register_buffer(
            "_observation_maximum", torch.ones(observation_dimensionality)
        )
        self.register_buffer("_action_minimum", torch.zeros(action_dimensionality))
        self.register_buffer("_action_maximum", torch.ones(action_dimensionality))

    @torch.no_grad()
    def compute_normalization_ranges(
        self,
        *,
        observation_batch: torch.Tensor,  # (N, observation_dimensionality)
        action_batch: torch.Tensor,  # (N, action_dimensionality)
    ) -> None:
        obs_range = NormalizationRange.from_values(values=observation_batch)
        action_range = NormalizationRange.from_values(values=action_batch)
        self._observation_minimum.copy_(obs_range.minimum)
        self._observation_maximum.copy_(obs_range.maximum)
        self._action_minimum.copy_(action_range.minimum)
        self._action_maximum.copy_(action_range.maximum)

    def observation_normalization_range(self) -> NormalizationRange:
        return NormalizationRange(
            minimum=self._observation_minimum, maximum=self._observation_maximum
        )

    def action_normalization_range(self) -> NormalizationRange:
        return NormalizationRange(
            minimum=self._action_minimum, maximum=self._action_maximum
        )

    def normalize_observation(
        self,
        *,
        observation: torch.Tensor,  # (batch_size, observation_dimensionality)
    ) -> torch.Tensor:  # (batch_size, observation_dimensionality)
        return self.observation_normalization_range().normalize(values=observation)

    def normalize_action(
        self,
        *,
        action: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        return self.action_normalization_range().normalize(values=action)

    def predict_action(
        self,
        *,
        observation: torch.Tensor,  # (batch_size, observation_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        """Raw observation -> raw action."""
        normalized_action = self.network(
            self.normalize_observation(observation=observation)
        )  # (batch_size, action_dimensionality)
        return self.action_normalization_range().denormalize(
            values=normalized_action
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

    def make_policy(*, seed: int) -> Policy:  # seed unused: BC is deterministic
        del seed

        def policy(*, observation: np.ndarray) -> np.ndarray:
            observation_tensor = torch.from_numpy(np.asarray(observation)).to(
                device=device, dtype=torch.float32
            )  # (observation_dimensionality,)
            with torch.no_grad():
                action_tensor = policy_model.predict_action(
                    observation=observation_tensor.unsqueeze(dim=0)
                )  # (1, action_dimensionality)
            return (
                action_tensor.squeeze(dim=0).cpu().numpy()
            )  # (action_dimensionality,)

        return policy

    return make_policy
