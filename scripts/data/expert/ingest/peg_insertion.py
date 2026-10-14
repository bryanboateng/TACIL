from scripts.data.expert.ingest import maniskill
from scripts.environments.peg_insertion.environment import (
    TASK_ID,
    create_base_environment,
)
from scripts.environments.peg_insertion.observation import (
    extract_state_from_environment,
)
from scripts.shared.episodes import RewardedEpisodes


def build_episodes() -> RewardedEpisodes:
    return maniskill.build_episodes(
        task_id=TASK_ID,
        create_base_environment=create_base_environment,
        extract_state=lambda environment: extract_state_from_environment(
            peg_insertion_side_env=environment
        ),
    )
