"""CPU simulation workers with batched policy inference in the parent process."""

import faulthandler
import logging
import multiprocessing
import traceback
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from time import monotonic
from typing import Any, Self

import numpy as np
import torch
from gymnasium import Env
from tqdm import tqdm

from scripts.evaluation.core.rollout import EpisodeRecord
from scripts.shared.policy_types import PolicyFactory

_WORKER_TIMEOUT_SECONDS = 120.0
_WORKER_WARNING_SECONDS = 30.0
_WORKER_TRACEBACK_SECONDS = 60.0
logger = logging.getLogger(__name__)


class ParallelEnvironmentPool:
    def __init__(
        self, *, factories: list[Callable[[], Env]], worker_count: int
    ) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        startup_start = monotonic()
        # Never fork a process that has already initialized CUDA for training.
        context = multiprocessing.get_context("spawn")
        self._connections: list[Connection] = []
        self._workers: list[BaseProcess] = []
        try:
            for _ in range(worker_count):
                parent, child = context.Pipe()
                worker = context.Process(target=_worker, args=(child, factories))
                self._connections.append(parent)
                try:
                    worker.start()
                finally:
                    child.close()
                self._workers.append(worker)
            for connection, worker in zip(
                self._connections, self._workers, strict=True
            ):
                _receive(connection, worker=worker, operation="starting worker")
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
    ) -> list[EpisodeRecord]:
        records: list[EpisodeRecord] = []
        batch_size = len(self._connections)
        for start in range(0, len(seeds), batch_size):
            batch_seeds = seeds[start : start + batch_size]
            records.extend(
                self._roll_out_batch(
                    seeds=batch_seeds,
                    policy_factory=policy_factory,
                    condition_index=condition_index,
                )
            )
        return records

    def _roll_out_batch(
        self, *, seeds: list[int], policy_factory: PolicyFactory, condition_index: int
    ) -> list[EpisodeRecord]:
        connections = self._connections[: len(seeds)]
        reset_start = monotonic()
        for connection, seed in zip(connections, seeds, strict=True):
            connection.send(("reset", (condition_index, seed)))
        observations = [
            _receive(
                connection,
                worker=self._workers[i],
                operation=f"preparing condition={condition_index}, seed={seeds[i]}",
            )
            for i, connection in enumerate(connections)
        ]
        logger.debug(
            "Prepared/reset %d scoring environments in %.2fs (first use includes creation).",
            len(seeds),
            monotonic() - reset_start,
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
                # Dispatch every step before waiting for any one simulator.
                for i in running:
                    connections[i].send(("step", action_batch[i]))
                still_running = []
                for i in running:
                    observation, reward, done, truncated = _receive(
                        connections[i],
                        worker=self._workers[i],
                        operation=f"stepping condition={condition_index}, seed={seeds[i]}",
                    )
                    observations[i] = observation
                    actions[i].append(action_batch[i])
                    rewards[i].append(float(reward))
                    terminated[i] = bool(done)
                    if not (done or truncated):
                        still_running.append(i)
                running = still_running
                progress.update()
                progress.set_postfix(running=len(running), refresh=False)
        return [
            EpisodeRecord(
                seed=seed,
                reward=max(rewards[i], default=float("nan")),
                terminated=terminated[i],
                final_observation=observations[i],
                actions=actions[i],
                rewards=rewards[i],
            )
            for i, seed in enumerate(seeds)
        ]


def _receive(
    connection: Connection,
    *,
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
def _worker_operation(connection: Connection, operation: str) -> Iterator[None]:
    connection.send(("status", operation))
    # The watchdog runs in native code, so it can expose a blocked simulator
    # call even when Python is unable to service another command from the parent.
    faulthandler.dump_traceback_later(_WORKER_TRACEBACK_SECONDS, repeat=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def _worker(connection: Connection, factories: list[Callable[[], Env]]) -> None:
    environment: Env | None = None
    environments: dict[int, Env] = {}
    resources = ExitStack()
    try:
        # Avoid multiplying PyTorch's CPU thread count by the number of workers.
        torch.set_num_threads(1)
        connection.send((True, None))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                condition_index, seed = payload
                if condition_index not in environments:
                    with _worker_operation(
                        connection,
                        f"creating environment condition={condition_index}, seed={seed}",
                    ):
                        environments[condition_index] = factories[condition_index]()
                    resources.callback(environments[condition_index].close)
                environment = environments[condition_index]
                with _worker_operation(
                    connection,
                    f"resetting environment condition={condition_index}, seed={seed}",
                ):
                    observation, _ = environment.reset(seed=seed)
                environment.observation_space.seed(seed)
                environment.action_space.seed(seed)
                connection.send((True, observation))
            elif command == "step":
                if environment is None:
                    raise RuntimeError("Reset the environment before stepping")
                observation, reward, terminated, truncated, _ = environment.step(
                    payload
                )
                connection.send((True, (observation, reward, terminated, truncated)))
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
            resources.close()
        finally:
            connection.close()
