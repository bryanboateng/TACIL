from scripts.data.expert.ingest import maniskill
from scripts.environments.stack_cube.environment import TASK_ID, create_base_environment
from scripts.environments.stack_cube.observation import extract_state_from_environment
from scripts.shared.episodes import RewardedEpisodes


def build_episodes() -> RewardedEpisodes:
    return maniskill.build_episodes(
        task_id=TASK_ID,
        create_base_environment=create_base_environment,
        extract_state=lambda environment: extract_state_from_environment(
            stack_cube_env=environment
        ),
    )
