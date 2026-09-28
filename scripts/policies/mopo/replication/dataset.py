import urllib.request
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy import typing as npt
from tqdm import tqdm

from scripts.policies.mopo.replication.tasks import Task
from scripts.shared.paths import OUTPUT_DIRECTORY_PATH

DATASET_DIRECTORY_PATH = OUTPUT_DIRECTORY_PATH / "d4rl"

_REQUIRED_FIELDS = ("observations", "actions", "rewards", "terminals")


@dataclass(frozen=True)
class D4rlDataset:
    states: npt.NDArray[np.float32]  # (transition_count, state_dimensionality)
    actions: npt.NDArray[np.float32]  # (transition_count, action_dimensionality)
    next_states: npt.NDArray[np.float32]  # (transition_count, state_dimensionality)
    rewards: npt.NDArray[np.float32]  # (transition_count, 1)
    terminations: npt.NDArray[np.float32]  # (transition_count, 1)

    @property
    def state_dimensionality(self) -> int:
        return self.states.shape[1]

    @property
    def action_dimensionality(self) -> int:
        return self.actions.shape[1]


def download_dataset(*, task: Task) -> Path:
    directory_path = DATASET_DIRECTORY_PATH
    directory_path.mkdir(parents=True, exist_ok=True)
    file_path = directory_path / task.dataset_file_name
    if file_path.exists():
        return file_path

    partial_file_path = file_path.with_suffix(suffix=".partial")
    with urllib.request.urlopen(url=task.dataset_url) as response:
        total_bytes = int(response.headers.get("Content-Length", 0))
        with (
            partial_file_path.open("wb") as file,
            tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                desc=task.dataset_file_name,
                dynamic_ncols=True,
            ) as progress_bar,
        ):
            while chunk := response.read(1 << 20):
                file.write(chunk)
                progress_bar.update(n=len(chunk))

    partial_file_path.rename(target=file_path)
    return file_path


def read_dataset(*, file_path: Path) -> D4rlDataset:

    with h5py.File(file_path, mode="r") as file:
        for name in _REQUIRED_FIELDS:
            if name not in file:
                raise ValueError(f"{file_path.name} carries no {name!r} field.")
        if "timeouts" not in file:
            raise ValueError(
                f"{file_path.name} carries no 'timeouts' field, so episode ends "
                f"cannot be found."
            )

        state_batch = np.asarray(file["observations"], dtype=np.float32)
        action_batch = np.asarray(file["actions"], dtype=np.float32)
        reward_batch = np.asarray(file["rewards"], dtype=np.float32)
        termination_batch = np.asarray(file["terminals"], dtype=np.bool_)
        timeout_batch = np.asarray(file["timeouts"], dtype=np.bool_)
        recorded_next_state_batch = (
            np.asarray(file["next_observations"], dtype=np.float32)
            if "next_observations" in file
            else None
        )

    transition_count = len(state_batch) - 1
    source_indices = np.arange(transition_count)

    keep_mask = ~timeout_batch[:transition_count]
    if recorded_next_state_batch is None:
        keep_mask &= ~termination_batch[:transition_count]
    kept_indices = source_indices[keep_mask]

    next_state_batch = (
        state_batch[kept_indices + 1]
        if recorded_next_state_batch is None
        else recorded_next_state_batch[kept_indices]
    )

    return D4rlDataset(
        states=state_batch[kept_indices],
        actions=action_batch[kept_indices],
        next_states=next_state_batch,
        rewards=reward_batch[kept_indices].reshape(-1, 1),
        terminations=termination_batch[kept_indices].astype(np.float32).reshape(-1, 1),
    )


def load_dataset(*, task: Task) -> D4rlDataset:
    return read_dataset(file_path=download_dataset(task=task))
