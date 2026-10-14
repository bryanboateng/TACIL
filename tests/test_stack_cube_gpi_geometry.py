import unittest

import numpy as np
import torch
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_matrix

from scripts.environments.stack_cube.gpi_geometry import StackCubeGpiGeometry
from scripts.environments.stack_cube.observation import (
    CUBE_A_ORIENTATION_SLICE,
    CUBE_A_POSITION_SLICE,
    CUBE_B_ORIENTATION_SLICE,
    CUBE_B_POSITION_SLICE,
    GRIPPER_OPENING_INDEX,
    STATE_DIMENSIONALITY,
    TCP_ORIENTATION_SLICE,
    TCP_POSITION_SLICE,
)
from scripts.shared.normalization import MinMaxNormalizer

POSITION_SLICES = (TCP_POSITION_SLICE, CUBE_A_POSITION_SLICE, CUBE_B_POSITION_SLICE)
ORIENTATION_SLICES = (
    TCP_ORIENTATION_SLICE,
    CUBE_A_ORIENTATION_SLICE,
    CUBE_B_ORIENTATION_SLICE,
)


def rotation_to_6d(rotation):  # (3, 3) -> (6,), see observation.py
    return rotation.T[:2].reshape(-1)


def create_state(generator):
    state = torch.zeros(STATE_DIMENSIONALITY, dtype=torch.float64)
    for position_slice in POSITION_SLICES:
        state[position_slice] = torch.from_numpy(generator.uniform(-0.5, 0.5, 3))
    for orientation_slice in ORIENTATION_SLICES:
        axis_angle = torch.from_numpy(generator.uniform(-1.0, 1.0, 3))
        state[orientation_slice] = rotation_to_6d(
            axis_angle_to_matrix(axis_angle.unsqueeze(0))[0]
        )
    state[GRIPPER_OPENING_INDEX] = generator.uniform(0.0, 0.04)
    return state


def rotate_tcp(state, axis_angle):
    rotated = state.clone()
    six_d = state[TCP_ORIENTATION_SLICE].reshape(2, 3)
    rotation = torch.stack(
        [six_d[0], six_d[1], torch.cross(six_d[0], six_d[1], dim=0)]
    ).T
    delta = axis_angle_to_matrix(torch.tensor([axis_angle], dtype=torch.float64))[0]
    rotated[TCP_ORIENTATION_SLICE] = rotation_to_6d(delta @ rotation)
    return rotated


def move_scene(states, rotation, shift):
    moved = states.clone()
    for position_slice in POSITION_SLICES:
        moved[:, position_slice] = states[:, position_slice] @ rotation.T + shift
    for orientation_slice in ORIENTATION_SLICES:
        six_d = states[:, orientation_slice].reshape(-1, 2, 3)
        columns = torch.stack(
            [six_d[:, 0], six_d[:, 1], torch.cross(six_d[:, 0], six_d[:, 1], dim=-1)],
            dim=-1,
        )  # (batch_size, 3, 3)
        moved[:, orientation_slice] = (
            (rotation @ columns).transpose(-2, -1)[:, :2].reshape(-1, 6)
        )
    return moved


class StackCubeGpiDistanceTests(unittest.TestCase):
    def setUp(self):
        self.geometry = StackCubeGpiGeometry()
        generator = np.random.default_rng(0)
        self.states = torch.stack([create_state(generator) for _ in range(64)])
        self.features = self.geometry.project_observations(
            observation_batch=self.states
        )
        self.normalizer = MinMaxNormalizer(value_batch=self.features)

    def distances(self, feature_batch, query_feature):
        return self.geometry.calculate_observation_distances(
            feature_batch=feature_batch,
            query_feature=query_feature,
            feature_normalizer=self.normalizer,
            robot_distance_weight=1.0,
            environment_distance_weight=1.0,
        )

    def distance_after(self, state, modified_state):
        feature_batch = self.geometry.project_observations(
            observation_batch=modified_state.unsqueeze(0)
        )
        query_feature = self.geometry.project_observations(
            observation_batch=state.unsqueeze(0)
        )[0]
        return float(self.distances(feature_batch, query_feature)[0])

    def test_one_controller_step_of_translation_is_distance_one(self):
        state = self.states[0]
        moved = state.clone()
        moved[TCP_POSITION_SLICE] += torch.tensor([0.06, 0.08, 0.0])
        self.assertAlmostEqual(self.distance_after(state, moved), 1.0, places=6)

    def test_one_controller_step_of_rotation_is_distance_one(self):
        state = self.states[0]
        rotated = rotate_tcp(state, [0.0, 0.1, 0.0])
        self.assertAlmostEqual(self.distance_after(state, rotated), 1.0, places=6)

    def test_full_gripper_range_is_distance_one(self):
        state = self.states[0]
        opened = state.clone()
        opened[GRIPPER_OPENING_INDEX] += 0.05
        self.assertAlmostEqual(self.distance_after(state, opened), 1.0, places=6)

    def test_base_cube_orientation_does_not_count(self):
        state = self.states[0]
        turned = state.clone()
        turned[CUBE_B_ORIENTATION_SLICE] = self.states[1][CUBE_B_ORIENTATION_SLICE]
        self.assertAlmostEqual(self.distance_after(state, turned), 0.0, places=6)

    def test_shifting_only_the_base_cube_counts_as_relative_motion(self):
        state = self.states[0]
        shifted = state.clone()
        shifted[CUBE_B_POSITION_SLICE] += torch.tensor([0.0, 0.1, 0.0])
        self.assertAlmostEqual(self.distance_after(state, shifted), 2.0, places=6)

    def test_moving_the_whole_scene_leaves_distances_unchanged(self):
        generator = np.random.default_rng(1)
        shift = torch.from_numpy(generator.uniform(-1.0, 1.0, 3))
        rotation = axis_angle_to_matrix(
            torch.from_numpy(generator.uniform(-1.0, 1.0, 3)).unsqueeze(0)
        )[0]
        moved_features = self.geometry.project_observations(
            observation_batch=move_scene(self.states, rotation, shift)
        )
        torch.testing.assert_close(
            self.distances(moved_features, moved_features[5]),
            self.distances(self.features, self.features[5]),
            rtol=0.0,
            atol=1e-6,
        )


class StackCubeGpiActionTests(unittest.TestCase):
    def setUp(self):
        self.geometry = StackCubeGpiGeometry()
        generator = np.random.default_rng(2)
        self.states = torch.stack([create_state(generator) for _ in range(16)])
        # The controller clips rotations to unit norm, so only those round-trip.
        self.actions = torch.from_numpy(
            np.concatenate(
                [
                    generator.uniform(-0.8, 0.8, (16, 3)),
                    generator.uniform(-0.55, 0.55, (16, 3)),
                    generator.choice([-1.0, 1.0], (16, 1)),
                ],
                axis=-1,
            )
        )

    def test_actions_survive_the_round_trip_through_the_object_frame(self):
        object_frame_actions = self.geometry.convert_actions_to_object_frame(
            observation_batch=self.states, action_batch=self.actions
        )
        torch.testing.assert_close(
            self.geometry.convert_actions_to_world_frame(
                observation_batch=self.states, action_batch=object_frame_actions
            ),
            self.actions,
            rtol=0.0,
            atol=1e-6,
        )

    def test_standing_on_the_recorded_step_gives_the_recorded_action(self):
        object_frame_actions = self.geometry.convert_actions_to_object_frame(
            observation_batch=self.states, action_batch=self.actions
        )
        for index in range(len(self.states)):
            flow_action = self.geometry.calculate_action(
                recorded_observation=self.states[index],
                recorded_action=object_frame_actions[index],
                current_observation=self.states[index],
                progression_weight=1.0,
                attraction_weight=1.0,
            )
            torch.testing.assert_close(
                self.geometry.convert_actions_to_world_frame(
                    observation_batch=self.states[index].unsqueeze(0),
                    action_batch=flow_action.unsqueeze(0),
                )[0],
                self.actions[index],
                rtol=0.0,
                atol=1e-6,
            )

    def test_shifting_the_base_cube_shifts_the_attraction_target(self):
        state = self.states[0]
        shifted = state.clone()
        shifted[CUBE_B_POSITION_SLICE] += torch.tensor([0.0, 0.05, 0.0])
        flow_action = self.geometry.calculate_action(
            recorded_observation=state,
            recorded_action=torch.zeros(7, dtype=torch.float64),
            current_observation=shifted,
            progression_weight=1.0,
            attraction_weight=1.0,
        )
        torch.testing.assert_close(
            flow_action[:3],
            torch.tensor([0.0, 0.05, 0.0], dtype=torch.float64),
            rtol=0.0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
