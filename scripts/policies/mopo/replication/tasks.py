from dataclasses import dataclass

_DATASET_BASE_URL = "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2"

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

EPOCH_COUNT = 3000


@dataclass(frozen=True)
class Task:
    name: str
    robot: str
    environment_id: str
    dataset_url: str
    rollout_length: int
    penalty_coefficient: float
    published_score: float | None
    published_score_std: float | None

    @property
    def dataset_file_name(self) -> str:
        return self.dataset_url.rsplit(sep="/", maxsplit=1)[-1]

    def score(self, *, episode_return: float) -> float:
        """Turns a return into the normalized score D4RL reports."""
        random_return, expert_return = _REFERENCE_RETURNS[self.robot]
        return (
            (episode_return - random_return) / (expert_return - random_return) * 100.0
        )


_SETTINGS = (
    ("halfcheetah", "medium", 5, 0.5, 72.4, 4.2),
    ("hopper", "medium", 5, 5.0, 62.8, 38.1),
    ("walker2d", "medium", 5, 0.5, 84.1, 3.2),
    ("halfcheetah", "medium_replay", 5, 0.5, 72.1, 3.8),
    ("hopper", "medium_replay", 5, 2.5, 92.7, 20.7),
    ("walker2d", "medium_replay", 1, 2.5, 85.9, 5.3),
    ("halfcheetah", "medium_expert", 5, 2.5, 83.6, 12.5),
    ("hopper", "medium_expert", 5, 5.0, 74.6, 44.2),
    ("walker2d", "medium_expert", 1, 2.5, 108.2, 4.3),
    ("halfcheetah", "expert", 5, 2.5, None, None),
    ("hopper", "expert", 5, 5.0, None, None),
    ("walker2d", "expert", 1, 2.5, None, None),
)

ALL_TASKS = {
    task.name: task
    for task in (
        Task(
            name=f"{robot}-{dataset.replace('_', '-')}-v2",
            robot=robot,
            environment_id=_ENVIRONMENT_IDS[robot],
            dataset_url=f"{_DATASET_BASE_URL}/{robot}_{dataset}-v2.hdf5",
            rollout_length=rollout_length,
            penalty_coefficient=penalty_coefficient,
            published_score=published_score,
            published_score_std=published_score_std,
        )
        for (
            robot,
            dataset,
            rollout_length,
            penalty_coefficient,
            published_score,
            published_score_std,
        ) in _SETTINGS
    )
}
