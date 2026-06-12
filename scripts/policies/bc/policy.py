from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.shared.normalization_range import NormalizationRange
from scripts.shared.paths import PROJECT_ROOT_DIRECTORY_PATH
from scripts.shared.policy_types import Policy, PolicyFactory


def build_behavior_cloning_network(
    *,
    observation_dimensionality: int,
    action_dimensionality: int,
    hidden_dimensionality: int,
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(observation_dimensionality, hidden_dimensionality),
        nn.ReLU(),
        nn.Linear(hidden_dimensionality, hidden_dimensionality),
        nn.ReLU(),
        nn.Linear(hidden_dimensionality, action_dimensionality),
    )


class BehaviorCloningPolicy(nn.Module):
    def __init__(
        self,
        *,
        observation_dimensionality: int,
        action_dimensionality: int,
        hidden_dimensionality: int,
        observation_normalization_range: NormalizationRange,
        action_normalization_range: NormalizationRange,
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

        # Buffers: saved in state_dict(), moved by .to(device), never receive gradients.
        self.register_buffer(
            "observation_minimum",
            torch.as_tensor(
                observation_normalization_range.minimum, dtype=torch.float32
            ).clone(),
        )
        self.register_buffer(
            "observation_maximum",
            torch.as_tensor(
                observation_normalization_range.maximum, dtype=torch.float32
            ).clone(),
        )
        self.register_buffer(
            "action_minimum",
            torch.as_tensor(
                action_normalization_range.minimum, dtype=torch.float32
            ).clone(),
        )
        self.register_buffer(
            "action_maximum",
            torch.as_tensor(
                action_normalization_range.maximum, dtype=torch.float32
            ).clone(),
        )

    def observation_normalization_range(self) -> NormalizationRange:
        """The fitted observation range, reconstructed from the buffers."""
        return NormalizationRange(
            minimum=self.observation_minimum, maximum=self.observation_maximum
        )

    def action_normalization_range(self) -> NormalizationRange:
        """The fitted action range, reconstructed from the buffers."""
        return NormalizationRange(
            minimum=self.action_minimum, maximum=self.action_maximum
        )

    def normalize_observation(self, *, observation: torch.Tensor) -> torch.Tensor:
        return self.observation_normalization_range().normalize(values=observation)

    def normalize_action(self, *, action: torch.Tensor) -> torch.Tensor:
        return self.action_normalization_range().normalize(values=action)

    def forward(
        self,
        *,
        normalized_observation: torch.Tensor,  # batch_size, observation_dimensionality
    ) -> torch.Tensor:  # (batch_size, action_dimensionality)
        """Normalized observation -> normalized action (use this for the training loss)."""
        return self.network(normalized_observation)

    @torch.no_grad()
    def act(self, *, observation: torch.Tensor) -> torch.Tensor:
        """Raw observation -> raw action (use this for rollout / evaluation)."""
        # observation: (batch_size, observation_dimensionality)
        # return: (batch_size, action_dimensionality)
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

    observation_dimensionality = checkpoint["observation_dimensionality"]
    action_dimensionality = checkpoint["action_dimensionality"]
    hidden_dimensionality = checkpoint["hidden_dimensionality"]

    # The real range values come from the state_dict below; these are just correctly
    # shaped placeholders so the buffers exist before load_state_dict overwrites them.
    placeholder_observation_range = NormalizationRange(
        minimum=torch.zeros(observation_dimensionality),
        maximum=torch.ones(observation_dimensionality),
    )
    placeholder_action_range = NormalizationRange(
        minimum=torch.zeros(action_dimensionality),
        maximum=torch.ones(action_dimensionality),
    )

    policy_model = BehaviorCloningPolicy(
        observation_dimensionality=observation_dimensionality,
        action_dimensionality=action_dimensionality,
        hidden_dimensionality=hidden_dimensionality,
        observation_normalization_range=placeholder_observation_range,
        action_normalization_range=placeholder_action_range,
    ).to(device=device)
    policy_model.load_state_dict(state_dict=checkpoint["policy_state_dict"])
    policy_model.eval()

    def make_policy(*, seed: int) -> Policy:  # seed unused: BC is deterministic
        del seed

        def policy(*, observation: np.ndarray) -> np.ndarray:
            observation_tensor = torch.from_numpy(np.asarray(observation)).to(
                device=device, dtype=torch.float32
            )  # (observation_dimensionality,)
            action_tensor = policy_model.act(
                observation=observation_tensor.unsqueeze(dim=0)
            )  # (1, action_dimensionality)
            return (
                action_tensor.squeeze(dim=0).cpu().numpy()
            )  # (action_dimensionality,)

        return policy

    return make_policy
