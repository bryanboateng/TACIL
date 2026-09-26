import faulthandler
import multiprocessing
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from time import monotonic
from typing import Any

import numpy as np
from gymnasium import Env
from tqdm import tqdm

from scripts.evaluation.core.rollout import Episode
from scripts.shared.log import create_logger
from scripts.shared.policy_types import PolicyFactory

_WORKER_TIMEOUT_SECONDS = 120.0
_WORKER_WARNING_SECONDS = 30.0
_WORKER_TRACEBACK_SECONDS = 60.0
logger = create_logger()


@dataclass(frozen=True)
class _EpisodeLocation:
    worker_index: int
    local_episode_index: int


@dataclass(frozen=True)
class _StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool


class ParallelEnvironmentPool:
    def __init__(
        self, *, environment_factories: dict[str, Callable[[], Env]], worker_count: int
    ) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        startup_start_timepoint = monotonic()
        process_context = multiprocessing.get_context(method="spawn")
        self._connections: list[Connection] = []
        self._workers: list[BaseProcess] = []

        try:
            for _ in range(worker_count):
                parent_connection, child_connection = process_context.Pipe()
                worker_process = process_context.Process(
                    target=_run_worker, args=(child_connection, environment_factories)
                )
                self._connections.append(parent_connection)
                try:
                    worker_process.start()
                finally:
                    child_connection.close()
                self._workers.append(worker_process)
            for connection, worker_process in zip(
                self._connections, self._workers, strict=True
            ):
                _receive_worker_result(
                    connection=connection,
                    worker=worker_process,
                    operation="starting worker",
                )
            logger.info(
                "Started %d scoring workers in %.2fs (reused across conditions).",
                worker_count,
                monotonic() - startup_start_timepoint,
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for connection in self._connections:
            try:
                connection.send(("shut_down", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()

        deadline_timepoint = monotonic() + 5.0
        for worker in self._workers:
            worker.join(timeout=max(0.0, deadline_timepoint - monotonic()))
        for worker in self._workers:
            if worker.is_alive():
                worker.terminate()
        for worker in self._workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=1.0)
            if worker.is_alive():
                logger.warning(
                    "CPU environment worker pid=%s survived being killed, so its "
                    "process handle stays open.",
                    worker.pid,
                )
                continue
            worker.close()
        self._connections.clear()
        self._workers.clear()

    def roll_out(
        self, *, seeds: list[int], policy_factory: PolicyFactory, condition_label: str
    ) -> list[Episode]:
        seeds_per_worker = _distribute_seeds_across_workers(
            seeds=seeds, worker_count=len(self._connections)
        )
        episode_locations = _get_episode_locations(seeds_per_worker=seeds_per_worker)
        observations = self._open_environments(
            seeds_per_worker=seeds_per_worker, condition_label=condition_label
        )

        actions: list[list[np.ndarray]] = [[] for _ in seeds]
        rewards: list[list[float]] = [[] for _ in seeds]
        terminations = [False for _ in seeds]
        policy = policy_factory(episode_seeds=seeds)
        running_episode_indices = list(range(len(seeds)))

        with tqdm(desc="Parallel rollout", unit="step", leave=False) as progress_bar:
            while len(running_episode_indices) != 0:
                action_per_episode_index = policy(
                    observations_by_episode_index={
                        episode_index: observations[episode_index]
                        for episode_index in running_episode_indices
                    }
                )
                episode_indices_per_worker = _group_episode_indices_per_worker(
                    episode_indices=running_episode_indices,
                    episode_locations=episode_locations,
                )
                self._send_actions_to_all_workers(
                    episode_indices_per_worker=episode_indices_per_worker,
                    episode_locations=episode_locations,
                    action_per_episode_index=action_per_episode_index,
                )
                step_result_per_episode_index = (
                    self._receive_step_results_from_all_workers(
                        episode_indices_per_worker=episode_indices_per_worker,
                        episode_locations=episode_locations,
                        condition_label=condition_label,
                    )
                )

                still_running_episode_indices = []
                for episode_index, step_result in step_result_per_episode_index.items():
                    observations[episode_index] = step_result.observation
                    actions[episode_index].append(
                        action_per_episode_index[episode_index]
                    )
                    rewards[episode_index].append(step_result.reward)
                    terminations[episode_index] = step_result.terminated
                    if not (step_result.terminated or step_result.truncated):
                        still_running_episode_indices.append(episode_index)
                running_episode_indices = sorted(still_running_episode_indices)
                progress_bar.update()
                progress_bar.set_postfix(
                    running=len(running_episode_indices), refresh=False
                )

        return [
            Episode(
                seed=seed,
                reward=max(rewards[episode_index], default=float("nan")),
                terminated=terminations[episode_index],
                final_observation=observations[episode_index],
                actions=actions[episode_index],
                rewards=rewards[episode_index],
            )
            for episode_index, seed in enumerate(seeds)
        ]

    def _send_actions_to_all_workers(
        self,
        *,
        episode_indices_per_worker: dict[int, list[int]],
        episode_locations: list[_EpisodeLocation],
        action_per_episode_index: dict[int, np.ndarray],
    ) -> None:
        for worker_index, worker_episode_indices in episode_indices_per_worker.items():
            action_per_local_episode_index = {
                episode_locations[episode_index].local_episode_index: (
                    action_per_episode_index[episode_index]
                )
                for episode_index in worker_episode_indices
            }
            self._connections[worker_index].send(
                ("apply_actions", action_per_local_episode_index)
            )

    def _receive_step_results_from_all_workers(
        self,
        *,
        episode_indices_per_worker: dict[int, list[int]],
        episode_locations: list[_EpisodeLocation],
        condition_label: str,
    ) -> dict[int, _StepResult]:
        step_result_per_episode_index: dict[int, _StepResult] = {}
        for worker_index, worker_episode_indices in episode_indices_per_worker.items():
            step_result_per_local_episode_index = _receive_worker_result(
                connection=self._connections[worker_index],
                worker=self._workers[worker_index],
                operation=f"stepping condition={condition_label}",
            )
            for episode_index in worker_episode_indices:
                observation, reward, terminated, truncated = (
                    step_result_per_local_episode_index[
                        episode_locations[episode_index].local_episode_index
                    ]
                )
                step_result_per_episode_index[episode_index] = _StepResult(
                    observation=observation,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
        return step_result_per_episode_index

    def _open_environments(
        self, *, seeds_per_worker: list[list[int]], condition_label: str
    ) -> list[np.ndarray]:
        opening_start_timepoint = monotonic()

        for connection, worker_seeds in zip(
            self._connections, seeds_per_worker, strict=True
        ):
            connection.send(("open_environments", (condition_label, worker_seeds)))

        observations: list[np.ndarray] = []
        for worker_index, worker_seeds in enumerate(seeds_per_worker):
            observations.extend(
                _receive_worker_result(
                    connection=self._connections[worker_index],
                    worker=self._workers[worker_index],
                    operation=(
                        f"preparing condition={condition_label}, "
                        f"{len(worker_seeds)} environments"
                    ),
                )
            )
        logger.debug(
            "Opened %d scoring environments in %.2fs.",
            sum(len(worker_seeds) for worker_seeds in seeds_per_worker),
            monotonic() - opening_start_timepoint,
        )
        return observations


def _distribute_seeds_across_workers(
    *, seeds: list[int], worker_count: int
) -> list[list[int]]:
    seed_count_per_worker, remainder = divmod(len(seeds), worker_count)
    seeds_per_worker = []
    start_index = 0
    for worker_index in range(worker_count):
        end_index = (
            start_index + seed_count_per_worker + (1 if worker_index < remainder else 0)
        )
        seeds_per_worker.append(seeds[start_index:end_index])
        start_index = end_index
    return seeds_per_worker


def _get_episode_locations(
    *, seeds_per_worker: list[list[int]]
) -> list[_EpisodeLocation]:
    return [
        _EpisodeLocation(
            worker_index=worker_index, local_episode_index=local_episode_index
        )
        for worker_index, worker_seeds in enumerate(seeds_per_worker)
        for local_episode_index in range(len(worker_seeds))
    ]


def _group_episode_indices_per_worker(
    *, episode_indices: list[int], episode_locations: list[_EpisodeLocation]
) -> dict[int, list[int]]:
    episode_indices_per_worker: dict[int, list[int]] = {}
    for episode_index in episode_indices:
        episode_indices_per_worker.setdefault(
            episode_locations[episode_index].worker_index, []
        ).append(episode_index)
    return episode_indices_per_worker


def _receive_worker_result(
    *, connection: Connection, worker: BaseProcess, operation: str
) -> Any:
    start_timepoint = monotonic()
    deadline_timepoint = start_timepoint + _WORKER_TIMEOUT_SECONDS
    worker_description = f"CPU environment worker pid={worker.pid}"
    while True:
        remaining_seconds = deadline_timepoint - monotonic()
        if remaining_seconds <= 0:
            raise RuntimeError(
                f"{worker_description} timed out after "
                f"{_WORKER_TIMEOUT_SECONDS:g}s while {operation}. "
                f"Exit code: {worker.exitcode}."
            )
        if not connection.poll(timeout=min(remaining_seconds, _WORKER_WARNING_SECONDS)):
            logger.warning(
                "%s still %s after %.1fs",
                worker_description,
                operation,
                monotonic() - start_timepoint,
            )
            continue
        try:
            message_kind, payload = connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(
                f"{worker_description} exited unexpectedly while {operation} "
                f"(exit code: {worker.exitcode})"
            ) from error

        match message_kind:
            case "progress":
                operation = payload
            case "error":
                raise RuntimeError(
                    f"{worker_description} failed while {operation}:\n{payload}"
                )
            case "success":
                return payload
            case _:
                raise RuntimeError(
                    f"{worker_description} sent an unknown message kind: "
                    f"{message_kind!r}"
                )


@contextmanager
def _announce_operation(*, connection: Connection, operation: str) -> Iterator[None]:
    connection.send(("progress", operation))
    faulthandler.dump_traceback_later(_WORKER_TRACEBACK_SECONDS, repeat=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def _run_worker(
    connection: Connection, environment_factories: dict[str, Callable[[], Env]]
) -> None:
    import torch

    environments: list[Env] = []
    try:
        torch.set_num_threads(1)
        connection.send(("success", None))
        while True:
            command, payload = connection.recv()
            if command == "shut_down":
                break
            if command == "open_environments":
                condition_label, seeds = payload
                _close_environments(environments=environments)
                environments = []
                observations = []
                for seed in seeds:
                    with _announce_operation(
                        connection=connection,
                        operation=(
                            f"creating environment condition={condition_label}, "
                            f"seed={seed}"
                        ),
                    ):
                        environment = environment_factories[condition_label]()
                    environments.append(environment)
                    with _announce_operation(
                        connection=connection,
                        operation=(
                            f"resetting environment condition={condition_label}, "
                            f"seed={seed}"
                        ),
                    ):
                        observation, _ = environment.reset(seed=seed)
                    environment.observation_space.seed(seed)
                    environment.action_space.seed(seed)
                    observations.append(observation)
                connection.send(("success", observations))
            elif command == "apply_actions":
                step_results = {}
                for local_episode_index, action in payload.items():
                    observation, reward, terminated, truncated, _ = environments[
                        local_episode_index
                    ].step(action)
                    step_results[local_episode_index] = (
                        observation,
                        reward,
                        terminated,
                        truncated,
                    )
                connection.send(("success", step_results))
            else:
                raise ValueError(f"Unknown worker command: {command}")
    except EOFError:
        pass  # Parent closed its pipe during cleanup.
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            _close_environments(environments=environments)
        finally:
            connection.close()


def _close_environments(*, environments: list[Env]) -> None:
    for environment in environments:
        try:
            environment.close()
        except Exception:  # noqa: BLE001
            logger.warning("Closing a scoring environment failed.", exc_info=True)
