"""CPU simulation workers with batched policy inference in the parent process."""

import multiprocessing
import traceback
from collections.abc import Callable
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


class ParallelEnvironmentPool:
    def __init__(self, *, factory: Callable[[], Env], worker_count: int) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        # Never fork a process that has already initialized CUDA for training.
        context = multiprocessing.get_context("spawn")
        self._connections: list[Connection] = []
        self._workers: list[BaseProcess] = []
        try:
            for _ in range(worker_count):
                parent, child = context.Pipe()
                worker = context.Process(target=_worker, args=(child, factory))
                self._connections.append(parent)
                try:
                    worker.start()
                finally:
                    child.close()
                self._workers.append(worker)
            for connection in self._connections:
                _receive(connection)
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
        self, *, seeds: list[int], policy_factory: PolicyFactory
    ) -> list[EpisodeRecord]:
        records: list[EpisodeRecord] = []
        batch_size = len(self._connections)
        for start in range(0, len(seeds), batch_size):
            batch_seeds = seeds[start : start + batch_size]
            records.extend(
                self._roll_out_batch(seeds=batch_seeds, policy_factory=policy_factory)
            )
        return records

    def _roll_out_batch(
        self, *, seeds: list[int], policy_factory: PolicyFactory
    ) -> list[EpisodeRecord]:
        connections = self._connections[: len(seeds)]
        for connection, seed in zip(connections, seeds, strict=True):
            connection.send(("reset", seed))
        observations = [_receive(connection) for connection in connections]
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
                    observation, reward, done, truncated = _receive(connections[i])
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


def _receive(connection: Connection) -> Any:
    if not connection.poll(_WORKER_TIMEOUT_SECONDS):
        raise RuntimeError("CPU environment worker timed out")
    try:
        success, result = connection.recv()
    except (EOFError, OSError) as error:
        raise RuntimeError("CPU environment worker exited unexpectedly") from error
    if not success:
        raise RuntimeError(f"CPU environment worker failed:\n{result}")
    return result


def _worker(connection: Connection, factory: Callable[[], Env]) -> None:
    environment: Env | None = None
    try:
        # Avoid multiplying PyTorch's CPU thread count by the number of workers.
        torch.set_num_threads(1)
        environment = factory()
        connection.send((True, None))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                observation, _ = environment.reset(seed=payload)
                environment.observation_space.seed(payload)
                environment.action_space.seed(payload)
                connection.send((True, observation))
            elif command == "step":
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
            if environment is not None:
                environment.close()
        finally:
            connection.close()
