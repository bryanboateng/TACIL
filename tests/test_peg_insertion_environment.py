import json

import h5py
import pytest
from mani_skill.utils.download_demo import DATASET_SOURCES

from scripts.data.expert.ingest.peg_insertion import (
    _Recording,
    _check_against_replay,
    _check_recording,
    _restore_episode,
)
from scripts.environments.peg_insertion.condition import UNDISTURBED_CONDITION
from scripts.environments.peg_insertion.environment import (
    TASK_ID,
    PegInsertionEnvironment,
)
from scripts.shared.download import download_and_extract_zip

_EPISODE_COUNT = 10


@pytest.fixture(scope="module")
def demonstration_path():
    return (
        download_and_extract_zip(url=DATASET_SOURCES[TASK_ID].raw_dataset_url)
        / TASK_ID
        / "motionplanning"
    )


def test_environment_reproduces_recorded_demonstrations(demonstration_path):
    recorded_episodes = json.loads(
        s=(demonstration_path / "trajectory.json").read_text()
    )["episodes"][:_EPISODE_COUNT]

    with h5py.File(name=demonstration_path / "trajectory.h5", mode="r") as file:
        longest = max(
            len(file[f"traj_{episode['episode_id']}"]["actions"])
            for episode in recorded_episodes
        )
        gym_environment = PegInsertionEnvironment().create_gym_environment(
            condition=UNDISTURBED_CONDITION,
            render_settings=None,
            max_episode_steps=longest,
        )
        try:
            for recorded_episode in recorded_episodes:
                recording = _Recording(
                    group=file[f"traj_{recorded_episode['episode_id']}"],
                    seed=recorded_episode["episode_seed"],
                    episode_id=recorded_episode["episode_id"],
                )
                _check_recording(recording=recording)
                _check_against_replay(
                    gym_environment=gym_environment,
                    recording=recording,
                    episode=_restore_episode(
                        gym_environment=gym_environment, recording=recording
                    ),
                )
        finally:
            gym_environment.close()
