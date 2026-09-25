import faulthandler
import multiprocessing
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from time import monotonic
from typing import Any, Self

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


class ParallelEnvironmentPool:
    def __init__(
        self, *, factories: list[Callable[[], Env]], worker_count: int
    ) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        startup_start = monotonic()
        # Never fork a process that has already initialized CUDA for training.
        context = multiprocessing.get_context(method="spawn")
        self._connections: list[Connection] = []
        self._workers: list[BaseProcess] = []

        try:
            for _ in range(worker_count):
                parent_connection, child_connection = context.Pipe()
                worker_process = context.Process(
                    target=_run_worker, args=(child_connection, factories)
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
                monotonic() - startup_start,
            )
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        for connection in self._connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()
        # One shared deadline, including when a worker is stuck in a simulator.
        deadline = monotonic() + 5.0
        for worker in self._workers:
            worker.join(timeout=max(0.0, deadline - monotonic()))
        for worker in self._workers:
            if worker.is_alive():
                worker.terminate()
        for worker in self._workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=1.0)
            worker.close()
        self._connections.clear()
        self._workers.clear()

    def roll_out(
        self, *, seeds: list[int], policy_factory: PolicyFactory, condition_index: int
    ) -> list[Episode]:
        # Every episode holds its own environment, so the policy sees them all at
        # once and the batch it plans on does not depend on the worker count.
        seed_chunks = _split_into_chunks(
            values=seeds, chunk_count=len(self._connections)
        )
        chunk_starts = np.cumsum([0, *(len(chunk) for chunk in seed_chunks)]).tolist()
        observations = self._open_environments(
            seed_chunks=seed_chunks, condition_index=condition_index
        )

        actions: list[list[np.ndarray]] = [[] for _ in seeds]
        rewards: list[list[float]] = [[] for _ in seeds]
        terminated = [False for _ in seeds]
        policy = policy_factory(episode_seeds=seeds)
        running = list(range(len(seeds)))

        with tqdm(desc="Parallel rollout", unit="step", leave=False) as progress:
            while running:
                action_batch = policy(
                    observations_by_episode_index={i: observations[i] for i in running}
                )
                slots_per_worker = _group_slots_per_worker(
                    episode_indices=running, chunk_starts=chunk_starts
                )
                # Dispatch every worker before waiting for any one simulator.
                for worker_index, slots in slots_per_worker.items():
                    self._connections[worker_index].send(
                        (
                            "step",
                            {
                                slot: action_batch[chunk_starts[worker_index] + slot]
                                for slot in slots
                            },
                        )
                    )
                still_running = []
                for worker_index, slots in slots_per_worker.items():
                    results = _receive_worker_result(
                        connection=self._connections[worker_index],
                        worker=self._workers[worker_index],
                        operation=f"stepping condition={condition_index}",
                    )
                    for slot in slots:
                        episode_index = chunk_starts[worker_index] + slot
                        observation, reward, done, truncated = results[slot]
                        observations[episode_index] = observation
                        actions[episode_index].append(action_batch[episode_index])
                        rewards[episode_index].append(float(reward))
                        terminated[episode_index] = bool(done)
                        if not (done or truncated):
                            still_running.append(episode_index)
                running = sorted(still_running)
                progress.update()
                progress.set_postfix(running=len(running), refresh=False)

        return [
            Episode(
                seed=seed,
                reward=max(rewards[i], default=float("nan")),
                terminated=terminated[i],
                final_observation=observations[i],
                actions=actions[i],
                rewards=rewards[i],
            )
            for i, seed in enumerate(seeds)
        ]

    def _open_environments(
        self, *, seed_chunks: list[list[int]], condition_index: int
    ) -> list[np.ndarray]:
        opening_start = monotonic()
        for connection, chunk in zip(self._connections, seed_chunks, strict=True):
            connection.send(("open", (condition_index, chunk)))
        observations: list[np.ndarray] = []
        for worker_index, chunk in enumerate(seed_chunks):
            observations.extend(
                _receive_worker_result(
                    connection=self._connections[worker_index],
                    worker=self._workers[worker_index],
                    operation=(
                        f"preparing condition={condition_index}, "
                        f"{len(chunk)} environments"
                    ),
                )
            )
        logger.debug(
            "Opened %d scoring environments in %.2fs.",
            sum(len(chunk) for chunk in seed_chunks),
            monotonic() - opening_start,
        )
        return observations


def _split_into_chunks(*, values: list[int], chunk_count: int) -> list[list[int]]:
    chunk_size, remainder = divmod(len(values), chunk_count)
    chunks = []
    start = 0
    for chunk_index in range(chunk_count):
        end = start + chunk_size + (1 if chunk_index < remainder else 0)
        chunks.append(values[start:end])
        start = end
    return chunks


def _group_slots_per_worker(
    *, episode_indices: list[int], chunk_starts: list[int]
) -> dict[int, list[int]]:
    slots_per_worker: dict[int, list[int]] = {}
    for episode_index in episode_indices:
        worker_index = (
            next(
                position
                for position, start in enumerate(chunk_starts)
                if start > episode_index
            )
            - 1
        )
        slots_per_worker.setdefault(worker_index, []).append(
            episode_index - chunk_starts[worker_index]
        )
    return slots_per_worker


def _receive_worker_result(
    *,
    connection: Connection,
    worker: BaseProcess | None = None,
    operation: str = "waiting for result",
    timeout_seconds: float = _WORKER_TIMEOUT_SECONDS,
) -> Any:
    start = monotonic()
    deadline = start + timeout_seconds
    identity = (
        f"CPU environment worker pid={worker.pid}"
        if worker
        else "CPU environment worker"
    )
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"{identity} timed out after {timeout_seconds:g}s while {operation}. "
                f"Exit code: {worker.exitcode if worker else 'unknown'}. "
                "Slow environment creation/reset prints worker tracebacks to stderr."
            )
        if not connection.poll(min(remaining, _WORKER_WARNING_SECONDS)):
            logger.warning(
                "%s still %s after %.1fs", identity, operation, monotonic() - start
            )
            continue
        try:
            success, result = connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(
                f"{identity} exited unexpectedly while {operation} "
                f"(exit code: {worker.exitcode if worker else 'unknown'})"
            ) from error
        if success == "status":
            # Status updates must not extend the deadline for a stuck reset.
            operation = result
            continue
        if not success:
            raise RuntimeError(f"{identity} failed while {operation}:\n{result}")
        return result


@contextmanager
def _announce_operation(*, connection: Connection, operation: str) -> Iterator[None]:
    connection.send(("status", operation))
    # The watchdog runs in native code, so it can expose a blocked simulator
    # call even when Python is unable to service another command from the parent.
    faulthandler.dump_traceback_later(_WORKER_TRACEBACK_SECONDS, repeat=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def _run_worker(connection: Connection, factories: list[Callable[[], Env]]) -> None:
    import torch

    environments: list[Env] = []
    try:
        # Avoid multiplying PyTorch's CPU thread count by the number of workers.
        torch.set_num_threads(1)
        connection.send((True, None))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "open":
                condition_index, seeds = payload
                _close_environments(environments=environments)
                environments = []
                observations = []
                for seed in seeds:
                    with _announce_operation(
                        connection=connection,
                        operation=(
                            f"creating environment condition={condition_index}, "
                            f"seed={seed}"
                        ),
                    ):
                        environment = factories[condition_index]()
                    environments.append(environment)
                    with _announce_operation(
                        connection=connection,
                        operation=(
                            f"resetting environment condition={condition_index}, "
                            f"seed={seed}"
                        ),
                    ):
                        observation, _ = environment.reset(seed=seed)
                    environment.observation_space.seed(seed)
                    environment.action_space.seed(seed)
                    observations.append(observation)
                connection.send((True, observations))
            elif command == "step":
                results = {}
                for slot, action in payload.items():
                    observation, reward, terminated, truncated, _ = environments[
                        slot
                    ].step(action)
                    results[slot] = (observation, reward, terminated, truncated)
                connection.send((True, results))
            else:
                raise ValueError(f"Unknown worker command: {command}")
    except EOFError:
        pass  # Parent closed its pipe during cleanup.
    except BaseException:
        try:
            connection.send((False, traceback.format_exc()))
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
