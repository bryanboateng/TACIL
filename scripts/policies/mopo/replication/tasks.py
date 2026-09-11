"""The D4RL benchmark tasks MOPO was published on.

Two tables exist, and they do not agree. The MOPO paper reports version 0 of
the datasets, OfflineRL-Kit reports version 2. A task on halfcheetah-medium
scores 42.3 in the paper and 72.4 in the library. Both are named here, so a
run says up front which table it aims at.
"""

from dataclasses import dataclass

_DATASET_BASE_URL = "http://rail.eecs.berkeley.edu/datasets/offline_rl"

# D4RL turns a return into a score between 0 and 100, where 0 is a random
# policy and 100 is an expert. These are the two returns it measures against.
_REFERENCE_RETURNS = {
    "halfcheetah": (-280.178953, 12135.0),
    "hopper": (-20.272305, 3234.3),
    "walker2d": (1.629008, 4592.3),
}

_ENVIRONMENT_IDS = {
    "halfcheetah": "HalfCheetah-v4",
    "hopper": "Hopper-v4",
    "walker2d": "Walker2d-v4",
}


@dataclass(frozen=True)
class Task:
    name: str
    robot: str
    environment_id: str
    dataset_url: str
    rollout_length: int
    penalty_coefficient: float
    published_score: float

    @property
    def dataset_file_name(self) -> str:
        return self.dataset_url.rsplit(sep="/", maxsplit=1)[-1]

    def score(self, *, episode_return: float) -> float:
        """Turns a return into the normalized score D4RL reports."""
        random_return, expert_return = _REFERENCE_RETURNS[self.robot]
        return (
            (episode_return - random_return) / (expert_return - random_return) * 100.0
        )


@dataclass(frozen=True)
class Target:
    """A published table, together with the settings that produced it.

    The appendix of the paper says spectral normalization is applied to the
    dynamics models. The authors' own code imports the function and never
    calls it, in neither mopo/models/fc.py nor mopo/models/bnn.py, so the
    published table was produced without it. The code is followed here,
    because the table is what is being reproduced.
    """

    name: str
    tasks: dict[str, Task]
    epoch_count: int


def _create_task(
    *,
    robot: str,
    dataset: str,
    dataset_path: str,
    rollout_length: int,
    penalty_coefficient: float,
    published_score: float,
) -> Task:
    return Task(
        name=f"{robot}-{dataset}",
        robot=robot,
        environment_id=_ENVIRONMENT_IDS[robot],
        dataset_url=f"{_DATASET_BASE_URL}/{dataset_path}",
        rollout_length=rollout_length,
        penalty_coefficient=penalty_coefficient,
        published_score=published_score,
    )


# The rollout length and the penalty coefficient come from Table 4 in the
# appendix of the paper, the scores from its Table 1. The paper calls the
# medium-replay datasets "mixed", and so do the version 0 file names.
_PAPER_SETTINGS = (
    ("halfcheetah", "random-v0", "gym_mujoco/halfcheetah_random.hdf5", 5, 0.5, 35.4),
    ("hopper", "random-v0", "gym_mujoco/hopper_random.hdf5", 5, 1.0, 11.7),
    ("walker2d", "random-v0", "gym_mujoco/walker2d_random.hdf5", 1, 1.0, 13.6),
    ("halfcheetah", "medium-v0", "gym_mujoco/halfcheetah_medium.hdf5", 1, 1.0, 42.3),
    ("hopper", "medium-v0", "gym_mujoco/hopper_medium.hdf5", 5, 5.0, 28.0),
    ("walker2d", "medium-v0", "gym_mujoco/walker2d_medium.hdf5", 5, 5.0, 17.8),
    ("halfcheetah", "mixed-v0", "gym_mujoco/halfcheetah_mixed.hdf5", 5, 1.0, 53.1),
    ("hopper", "mixed-v0", "gym_mujoco/hopper_mixed.hdf5", 5, 1.0, 67.5),
    # The version 0 file for this one is named "walker", not "walker2d".
    ("walker2d", "mixed-v0", "gym_mujoco/walker_mixed.hdf5", 1, 1.0, 39.0),
    (
        "halfcheetah",
        "medium-expert-v0",
        "gym_mujoco/halfcheetah_medium_expert.hdf5",
        5,
        1.0,
        63.3,
    ),
    (
        "hopper",
        "medium-expert-v0",
        "gym_mujoco/hopper_medium_expert.hdf5",
        5,
        1.0,
        23.7,
    ),
    (
        "walker2d",
        "medium-expert-v0",
        "gym_mujoco/walker2d_medium_expert.hdf5",
        1,
        2.0,
        44.6,
    ),
)

# The rollout length and the penalty coefficient are the ones suggested in
# run_example/run_mopo.py of OfflineRL-Kit, the scores are from its README.
# The library reports no result on the random datasets.
_OFFLINE_RL_KIT_SETTINGS = (
    ("halfcheetah", "medium-v2", "halfcheetah_medium-v2.hdf5", 5, 0.5, 72.4),
    ("hopper", "medium-v2", "hopper_medium-v2.hdf5", 5, 5.0, 62.8),
    ("walker2d", "medium-v2", "walker2d_medium-v2.hdf5", 5, 0.5, 84.1),
    (
        "halfcheetah",
        "medium-replay-v2",
        "halfcheetah_medium_replay-v2.hdf5",
        5,
        0.5,
        72.1,
    ),
    ("hopper", "medium-replay-v2", "hopper_medium_replay-v2.hdf5", 5, 2.5, 92.7),
    ("walker2d", "medium-replay-v2", "walker2d_medium_replay-v2.hdf5", 1, 2.5, 85.9),
    (
        "halfcheetah",
        "medium-expert-v2",
        "halfcheetah_medium_expert-v2.hdf5",
        5,
        2.5,
        83.6,
    ),
    ("hopper", "medium-expert-v2", "hopper_medium_expert-v2.hdf5", 5, 5.0, 74.6),
    ("walker2d", "medium-expert-v2", "walker2d_medium_expert-v2.hdf5", 1, 2.5, 108.2),
)


PAPER = Target(
    name="paper",
    tasks={
        task.name: task
        for task in (
            _create_task(
                robot=robot,
                dataset=dataset,
                dataset_path=dataset_path,
                rollout_length=rollout_length,
                penalty_coefficient=penalty_coefficient,
                published_score=published_score,
            )
            for robot, dataset, dataset_path, rollout_length, penalty_coefficient, published_score in _PAPER_SETTINGS
        )
    },
    # The authors' own configuration runs 1000 epochs of 1000 steps.
    epoch_count=1000,
)

OFFLINE_RL_KIT = Target(
    name="offline-rl-kit",
    tasks={
        task.name: task
        for task in (
            _create_task(
                robot=robot,
                dataset=dataset,
                dataset_path=f"gym_mujoco_v2/{dataset_file_name}",
                rollout_length=rollout_length,
                penalty_coefficient=penalty_coefficient,
                published_score=published_score,
            )
            for robot, dataset, dataset_file_name, rollout_length, penalty_coefficient, published_score in _OFFLINE_RL_KIT_SETTINGS
        )
    },
    epoch_count=3000,
)

ALL_TARGETS = {target.name: target for target in (PAPER, OFFLINE_RL_KIT)}
