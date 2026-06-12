import numpy as np
import torch

from scripts.policies.gpi.config import Config
from scripts.policies.gpi.database import StateDatabase
from scripts.policies.gpi.dataset import PushTEpisodeDataset
from scripts.policies.gpi.geometry import relative_action_to_global
from scripts.policies.gpi.planning import PlanBuffer
from scripts.shared.data import EpisodeDataset
from scripts.shared.policy_types import Policy, PolicyFactory

Key = tuple[int, int]


def calculate_dynamic_lambda2(*, step_in_plan: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    progress = min(step_in_plan / (total_steps - 1), 1.0)
    # Maps the remaining plan length to the λ₂ attraction weight (Alg.1, line 7/8).
    return 0.5 + 0.5 * progress


class StateGPIPolicy:
    """Implements Algorithm 1 (GPI) on state observations."""

    def __init__(
        self,
        *,
        dataset: PushTEpisodeDataset,
        database: StateDatabase,
        config: Config,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.database = database
        self.plan = PlanBuffer(dataset=dataset, horizon=config.action_horizon)
        self.rng = np.random.default_rng(seed=seed if seed is not None else config.seed)

        self.recent_set: set[Key] = set()
        self.current_noise_std: float = config.noise.initial_std
        self.reset()

    # ----------------------------------------------------------- lifecycle
    def reset(self) -> None:
        self.current_noise_std = self.config.noise.initial_std
        self.plan.clear()
        self.recent_set.clear()

    # --------------------------------------------------------------- query
    def get_action(
        self,
        *,
        observation: np.ndarray,  # (1, observation_dimensionality)
    ) -> np.ndarray:  # (action_dimensionality,)
        current_observation = np.asarray(
            observation[-1], dtype=np.float32
        )  # (observation_dimensionality,)
        normalized_observation = self.dataset.normalize_observations(
            observations=current_observation
        )  # (observation_dimensionality,)
        noisy_normalized_observation = self._add_observation_noise(
            normalized_observation=normalized_observation
        )  # (observation_dimensionality,)

        # Steps 2-11: geometry-aware policy synthesis in normalised space.
        normalized_action = self._compute_action(
            normalized_observation=noisy_normalized_observation
        )  # (action_dimensionality,) or None
        if normalized_action is None:
            return np.zeros(shape=2, dtype=np.float32)

        denormalized = self.dataset.denormalize_actions(
            normalized_actions=normalized_action
        )  # (action_dimensionality,)
        return _to_global(observation=current_observation, action=denormalized)  # (2,)

    def _compute_action(
        self,
        *,
        normalized_observation: np.ndarray,  # (observation_dimensionality,)
    ) -> np.ndarray | None:  # (action_dimensionality,) or None
        if self.plan.is_empty():
            # Steps 5-6: pick the demonstration/time with minimal combined distance.
            _, keys = self.database.nearest(
                query=normalized_observation, exclude=self.recent_set
            )
            if not keys:
                return None
            start_key = keys[0]  # κ(x₀) = argmin d_t^{(i)}
            self.plan.load(episode_index=start_key[0], timestep=start_key[1])

        total_steps = len(self.plan.actions)
        action_norm, state_norm, timestep = self.plan.pop()
        # action_norm: (action_dimensionality,), state_norm: (observation_dimensionality,)
        step_in_plan = self.plan.pointer - 1

        if self.config.guidance.fixed_lambda2 is not None:
            lambda2 = float(self.config.guidance.fixed_lambda2)
        else:
            lambda2 = calculate_dynamic_lambda2(
                step_in_plan=step_in_plan, total_steps=total_steps
            )

        query_agent = normalized_observation[:2]  # (2,) — agent (x, y)
        neighbor_agent = state_norm[:2]  # (2,) — nearest neighbor's agent (x, y)

        # Step 6: progression flow u_prog = ẋ κ (x₀) following the local tangent.
        u_prog = action_norm[:2] - neighbor_agent  # (2,)
        # Step 7: attraction flow u_att = -∇ d_rob that steers toward the demo.
        u_att = neighbor_agent - query_agent  # (2,)
        # Steps 8-11: local policy π_i(x₀) = λ₁ u_prog + λ₂ u_att.
        agent_flow = (
            query_agent + self.config.guidance.lambda1 * u_prog + lambda2 * u_att
        )  # (2,)

        local_policy = action_norm.copy()  # (action_dimensionality,)
        local_policy[:2] = agent_flow

        self.recent_set.add((self.plan.episode_index, timestep))
        return local_policy  # (action_dimensionality,)

    # --------------------------------------------------------------- noise
    def _add_observation_noise(
        self,
        *,
        normalized_observation: np.ndarray,  # (observation_dimensionality,)
    ) -> np.ndarray:  # (observation_dimensionality,)
        noise = self.rng.normal(
            loc=0.0, scale=self.current_noise_std, size=normalized_observation.shape
        )  # (observation_dimensionality,)
        noisy = np.clip(
            normalized_observation + noise, -1.0, 1.0
        )  # (observation_dimensionality,)
        self.current_noise_std = max(
            self.current_noise_std * self.config.noise.decay_rate,
            self.config.noise.minimum_std,
        )
        return noisy


def load_geometric_policy(
    *, episode_dataset: EpisodeDataset, config: Config, device: torch.device
) -> PolicyFactory:
    """Build the demonstration index once, then return a per-seed policy factory.

    The expensive, seed-independent state (the device-resident demonstration
    database, including any random subset) is built a single time; each rollout
    seed gets a fresh policy that shares it, varying only the observation-noise RNG.
    """
    dataset = PushTEpisodeDataset(episode_dataset=episode_dataset)
    database = StateDatabase(
        dataset=dataset, device=str(device), subset_size=config.retrieval.subset_size
    )

    def make_policy(*, seed: int) -> Policy:
        policy = StateGPIPolicy(
            dataset=dataset, database=database, config=config, seed=seed
        )

        def act(*, observation: np.ndarray) -> np.ndarray:
            return policy.get_action(observation=observation[np.newaxis])

        return act

    return make_policy


def _to_global(
    *,
    observation: np.ndarray,  # (observation_dimensionality,)
    action: np.ndarray,  # (action_dimensionality,)
) -> np.ndarray:  # (2,)
    return (
        relative_action_to_global(
            observation=torch.from_numpy(observation).unsqueeze(dim=0),
            action=torch.from_numpy(action).unsqueeze(dim=0),
        )
        .squeeze(dim=0)
        .numpy()
    )
