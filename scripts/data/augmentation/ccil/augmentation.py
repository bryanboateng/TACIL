import torch
from wandb import Run

from scripts.data.augmentation.ccil.config import CcilConfig
from scripts.data.augmentation.ccil.create_augmented_data import (
    create_ccil_augmented_data,
)
from scripts.shared.episodes import StateActionEpisodes
from scripts.shared.state_action_transitions import StateActionTransitions


def augment(
    *,
    config: CcilConfig,
    expert_episodes: StateActionEpisodes,
    seed: int,
    device: torch.device,
    run: Run,
) -> StateActionTransitions:
    return create_ccil_augmented_data(
        config=config,
        expert_episodes=expert_episodes,
        seed=seed,
        device=device,
        run=run,
    )
