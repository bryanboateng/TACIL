import types
from dataclasses import dataclass

import numpy as np
import pymunk
from gymnasium import Env

OFF_SCREEN_POSITION = (-200.0, -200.0)


@dataclass(frozen=True)
class DisturberConfig:
    attack_interval: int = 80
    attack_duration: int = 8


def setup_disturber(environment: Env, config: DisturberConfig) -> None:
    """Add the disturber body and patch step/reset onto the env.

    Reset the env ONCE before calling this so block/agent/space exist. Then, before
    each episode, set `environment.unwrapped.disturber_rng` to a fresh
    `np.random.default_rng(episode_seed)` (see evaluation.py).
    """
    _add_disturber(environment=environment)
    _patch_step_with_disturber(
        environment=environment,
        attack_interval=config.attack_interval,
        attack_duration=config.attack_duration,
    )
    _patch_reset_for_disturber(environment=environment)


def _add_disturber(environment: Env) -> None:
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    body.position = OFF_SCREEN_POSITION

    shape = pymunk.Circle(body=body, radius=15)
    shape.elasticity = 0.1

    unwrapped_environment = environment.unwrapped
    unwrapped_environment.space.add(body, shape)  # ty:ignore[unresolved-attribute]
    unwrapped_environment.disturber = body  # ty:ignore[unresolved-attribute]
    unwrapped_environment.disturber_t = 0  # ty:ignore[unresolved-attribute]


def _patch_step_with_disturber(
    environment: Env, attack_interval: int, attack_duration: int
) -> None:
    unwrapped_environment = environment.unwrapped
    original_step_function = unwrapped_environment.step

    def disturbed_step(env, action):
        phase = env.disturber_t % attack_interval

        disturbance_is_active = phase < attack_duration
        disturbance_has_just_started = phase == 0

        env.disturber_t += 1

        if not disturbance_is_active:
            env.disturber.velocity = (0, 0)
            env.disturber.position = OFF_SCREEN_POSITION
        else:
            block_point = _sample_random_block_point(env=env, rng=env.disturber_rng)

            # Spawn from a random direction in the hemisphere facing away from the
            # goal, so the disturber pushes the block further from the goal.
            if disturbance_has_just_started:
                approach_direction = _sample_random_direction_away_from_goal(
                    block_point=block_point, env=env, rng=env.disturber_rng
                )

                spawn_distance = env.disturber_rng.uniform(low=25.0, high=45.0)
                spawn_position = block_point - approach_direction * spawn_distance

                env.disturber.position = tuple(spawn_position)
                env.disturber_speed = env.disturber_rng.uniform(low=50.0, high=300.0)

            push_direction = _compute_push_direction(
                env=env, target_position=block_point
            )

            env.disturber.velocity = (
                pymunk.Vec2d(float(push_direction[0]), float(push_direction[1]))
                * env.disturber_speed
            )

        return original_step_function(action)

    unwrapped_environment.step = types.MethodType(disturbed_step, unwrapped_environment)  # ty:ignore[invalid-assignment]


def _patch_reset_for_disturber(environment: Env) -> None:
    unwrapped_environment = environment.unwrapped
    original_reset_function = unwrapped_environment.reset

    def new_reset(self, **kwargs):
        result = original_reset_function(**kwargs)

        # gym-pusht's reset() rebuilds the pymunk space from scratch, leaving our
        # disturber body with a stale `body.space`. Recreate it fresh each reset —
        # it's a 4-line object, the cost is negligible.
        self.disturber = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.disturber.position = OFF_SCREEN_POSITION
        disturber_shape = pymunk.Circle(body=self.disturber, radius=15)
        disturber_shape.elasticity = 0.1
        self.space.add(self.disturber, disturber_shape)

        self.disturber_t = 0
        return result

    unwrapped_environment.reset = types.MethodType(new_reset, unwrapped_environment)  # ty:ignore[invalid-assignment]


def _sample_random_block_point(env, rng: np.random.Generator) -> np.ndarray:
    vertices = []
    for shape in env.block.shapes:
        if hasattr(shape, "get_vertices"):
            vertices.extend(shape.get_vertices())

    random_vertex = vertices[rng.integers(len(vertices))]
    return np.array(env.block.local_to_world(random_vertex))


def _sample_random_direction_away_from_goal(
    block_point, env, rng: np.random.Generator
) -> np.ndarray:
    goal_position = np.array(env.goal_pose[:2])

    goal_to_block = block_point - goal_position
    goal_to_block_distance = np.linalg.norm(goal_to_block)

    if goal_to_block_distance > 1e-6:
        goal_to_block /= goal_to_block_distance
    else:
        goal_to_block = np.array([1.0, 0.0])

    random_angle = rng.uniform(low=0.0, high=2 * np.pi)
    random_unit_vector = np.array([np.cos(random_angle), np.sin(random_angle)])

    # If the random direction points toward the goal, flip it.
    if np.dot(random_unit_vector, goal_to_block) < 0:
        random_unit_vector *= -1

    return random_unit_vector


def _compute_push_direction(env, target_position) -> np.ndarray:
    push_direction = target_position - np.array(env.disturber.position)
    push_distance = np.linalg.norm(push_direction)

    if push_distance > 1e-6:
        push_direction /= push_distance
    else:
        push_direction = np.zeros(2)

    return push_direction
