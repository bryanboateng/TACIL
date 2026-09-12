"""The D4RL benchmark tasks this port is checked against.

The reference for MOPO in this repository is OfflineRL-Kit, the library that
accompanies MOBILE (Sun et al., ICML 2023). Its README prints a score per
task, and reaching those scores is what says the port computes what it
should. The settings below are the ones its run_example/run_mopo.py suggests.

The MOPO authors' own code at github.com/tianheyu927/mopo reports much lower
numbers on the same robots, for example 42.3 against 72.4 on the medium half
cheetah. It runs on TensorFlow 1 and mujoco-py and no longer starts, so it is
read rather than run. Reading it turned up places where the two disagree, and
they are recorded here because a reader of the thesis will ask.

Where OfflineRL-Kit departs from the authors, and this port follows
OfflineRL-Kit:

  actor learning rate       1e-4 against 3e-4
  temperature learning rate 1e-4 against 3e-4
  learning rate schedule    cosine on the actor against none
  temperature               held at or below one against unbounded
  action log std floor      -5 against -20
  squash correction         log(1 - a^2 + 1e-6) against the exact form
  target entropy            minus the action count against a fixed -3
  halfcheetah termination   out of range check against never terminating
  walker2d termination      an added out of range check
  dataset assembly          its own reader against d4rl.qlearning_dataset

And where the authors' code departs from their own paper:

  spectral normalization    the appendix describes it, fc.py imports the
                            function and never calls it
  halfcheetah medium-expert Table 4 prints a penalty of 1, the config sets 5
  target entropy            the paper never mentions it, base.py fixes -3
"""

from dataclasses import dataclass

_DATASET_BASE_URL = "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2"

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

EPOCH_COUNT = 3000


@dataclass(frozen=True)
class Task:
    name: str
    robot: str
    environment_id: str
    dataset_url: str
    rollout_length: int
    penalty_coefficient: float
    published_score: float
    # The spread over the four seeds the library averaged. It varies wildly
    # between tasks, so a task with a tight spread says far more about
    # whether the port is right than one with a loose one.
    published_score_std: float

    @property
    def dataset_file_name(self) -> str:
        return self.dataset_url.rsplit(sep="/", maxsplit=1)[-1]

    def score(self, *, episode_return: float) -> float:
        """Turns a return into the normalized score D4RL reports."""
        random_return, expert_return = _REFERENCE_RETURNS[self.robot]
        return (
            (episode_return - random_return) / (expert_return - random_return) * 100.0
        )


# Rollout length and penalty from run_example/run_mopo.py, scores from the
# README, which averages four seeds.
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
