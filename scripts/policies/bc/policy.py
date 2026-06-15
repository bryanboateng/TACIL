from pathlib import Path

import numpy as np
import torch

from scripts.shared.normalization_range import NormalizationRange
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
    observation_range: NormalizationRange
    action_range: NormalizationRange

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

        # Placeholders, overwritten by compute_normalization_ranges() or load_state_dict().
        self.observation_range = NormalizationRange(
            minimum=torch.zeros(observation_dimensionality),
            maximum=torch.ones(observation_dimensionality),
        )
        self.action_range = NormalizationRange(
            minimum=torch.zeros(action_dimensionality),
            maximum=torch.ones(action_dimensionality),
        )

    @torch.no_grad()
    def compute_normalization_ranges(
        self,
        *,
        observation_batch: torch.Tensor,  # (N, observation_dimensionality)
        action_batch: torch.Tensor,  # (N, action_dimensionality)
    ) -> None:
        self.observation_range = NormalizationRange.from_values(
            value_batch=observation_batch
        )
        self.action_range = NormalizationRange.from_values(value_batch=action_batch)

    def normalize_observation(
        self,
        *,
        observation: torch.Tensor,  # (batch_size, observation_dimensionality)
    ) -> torch.Tensor:  # (batch_size, observation_dimensionality)
        return self.observation_range.normalize(value_batch=observation)

    def normalize_action(
        self,
        *,
        action: torch.Tensor,  # (batch_size, action_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        return self.action_range.normalize(value_batch=action)

    def predict_action(
        self,
        *,
        observation: torch.Tensor,  # (batch_size, observation_dimensionality)
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        """Raw observation -> raw action."""
        normalized_action = self.network(
            self.normalize_observation(observation=observation)
        )  # (batch_size, action_dimensionality)
        return self.action_range.denormalize(
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
