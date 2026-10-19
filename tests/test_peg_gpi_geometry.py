import unittest

import numpy as np
import torch
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_matrix

from scripts.environments.peg_insertion.gpi_geometry import PegInsertionGpiGeometry
from scripts.environments.peg_insertion.observation import (
    GRIPPER_OPENING_INDEX,
    HOLE_ORIENTATION_SLICE,
    HOLE_POSITION_SLICE,
    HOLE_RADIUS_INDEX,
    PEG_HALF_SIZE_SLICE,
    PEG_ORIENTATION_SLICE,
    PEG_POSITION_SLICE,
    STATE_DIMENSIONALITY,
    TCP_ORIENTATION_SLICE,
    TCP_POSITION_SLICE,
)
from scripts.shared.normalization import MinMaxNormalizer


def rotation_to_6d(rotation):  # (3, 3) -> (6,), see observation.py
    return rotation.T[:2].reshape(-1)


def create_state(generator):
    state = torch.zeros(STATE_DIMENSIONALITY, dtype=torch.float64)
    for position_slice in (TCP_POSITION_SLICE, PEG_POSITION_SLICE, HOLE_POSITION_SLICE):
        state[position_slice] = torch.from_numpy(generator.uniform(-0.5, 0.5, 3))
    for orientation_slice in (
        TCP_ORIENTATION_SLICE,
        PEG_ORIENTATION_SLICE,
        HOLE_ORIENTATION_SLICE,
    ):
        axis_angle = torch.from_numpy(generator.uniform(-1.0, 1.0, 3))
        state[orientation_slice] = rotation_to_6d(
            axis_angle_to_matrix(axis_angle.unsqueeze(0))[0]
        )
    state[GRIPPER_OPENING_INDEX] = generator.uniform(0.0, 0.04)
    state[PEG_HALF_SIZE_SLICE] = torch.from_numpy(generator.uniform(0.01, 0.12, 3))
    state[HOLE_RADIUS_INDEX] = generator.uniform(0.018, 0.028)
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


class PegGpiDistanceTests(unittest.TestCase):
    def setUp(self):
        self.geometry = PegInsertionGpiGeometry()
        generator = np.random.default_rng(0)
        self.states = torch.stack([create_state(generator) for _ in range(64)])
        self.features = self.geometry.project_observations(
            observation_batch=self.states
        )
        self.normalizer = MinMaxNormalizer(value_batch=self.features)

    def distances(
        self,
        feature_batch,
        query_feature,
        environment_distance_share=0.5,
        rotation_distance_share=0.5,
    ):
        return self.geometry.calculate_observation_distances(
            feature_batch=feature_batch,
            query_feature=query_feature,
            feature_normalizer=self.normalizer,
            environment_distance_share=environment_distance_share,
            rotation_distance_share=rotation_distance_share,
        )

    def distance_after(
        self,
        state,
        modified_state,
        environment_distance_share=0.5,
        rotation_distance_share=0.5,
    ):
        feature_batch = self.geometry.project_observations(
            observation_batch=modified_state.unsqueeze(0)
        )
        query_feature = self.geometry.project_observations(
            observation_batch=state.unsqueeze(0)
        )[0]
        return float(
            self.distances(
                feature_batch,
                query_feature,
                environment_distance_share=environment_distance_share,
                rotation_distance_share=rotation_distance_share,
            )[0]
        )

    def test_translation_counts_in_metres(self):
        state = self.states[0]
        moved = state.clone()
        moved[TCP_POSITION_SLICE] += torch.tensor([0.06, 0.08, 0.0])
        distance = self.distance_after(
            state, moved, environment_distance_share=0.0, rotation_distance_share=0.0
        )
        self.assertAlmostEqual(distance, 0.1, places=6)

    def test_rotation_counts_in_radians(self):
        state = self.states[0]
        rotated = rotate_tcp(state, [0.0, 0.1, 0.0])
        distance = self.distance_after(
            state, rotated, environment_distance_share=0.0, rotation_distance_share=1.0
        )
        self.assertAlmostEqual(distance, 0.1, places=6)

    def test_rotation_share_weighs_radians_against_metres(self):
        state = self.states[0]
        moved = rotate_tcp(state, [0.0, 0.1, 0.0])
        moved[TCP_POSITION_SLICE] += torch.tensor([0.0, 0.2, 0.0])
        distance = self.distance_after(
            state, moved, environment_distance_share=0.0, rotation_distance_share=0.25
        )
        self.assertAlmostEqual(distance, 0.75 * 0.2 + 0.25 * 0.1, places=6)

    def test_environment_share_weighs_the_object_against_the_robot(self):
        state = self.states[0]
        moved = state.clone()
        moved[TCP_POSITION_SLICE] += torch.tensor([0.0, 0.1, 0.0])
        self.assertAlmostEqual(
            self.distance_after(
                state,
                moved,
                environment_distance_share=1.0,
                rotation_distance_share=0.0,
            ),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            self.distance_after(
                state,
                moved,
                environment_distance_share=0.25,
                rotation_distance_share=0.0,
            ),
            0.075,
            places=6,
        )

    def test_gripper_opening_counts_in_metres(self):
        state = self.states[0]
        opened = state.clone()
        opened[GRIPPER_OPENING_INDEX] += 0.05
        distance = self.distance_after(
            state, opened, environment_distance_share=0.0, rotation_distance_share=0.0
        )
        self.assertAlmostEqual(distance, 0.05, places=6)

    def test_hole_radius_and_peg_cross_section_do_not_count(self):
        state = self.states[0]
        resized = state.clone()
        resized[HOLE_RADIUS_INDEX] += 0.01
        resized[PEG_HALF_SIZE_SLICE.start + 1 : PEG_HALF_SIZE_SLICE.stop] += 0.01
        self.assertAlmostEqual(self.distance_after(state, resized), 0.0, places=6)

    def test_peg_half_length_counts_in_metres(self):
        state = self.states[0]
        lengthened = state.clone()
        lengthened[PEG_HALF_SIZE_SLICE.start] += 0.04
        distance = self.distance_after(
            state,
            lengthened,
            environment_distance_share=1.0,
            rotation_distance_share=0.0,
        )
        self.assertAlmostEqual(distance, 0.04, places=6)

    def test_moving_the_whole_scene_leaves_distances_unchanged(self):
        generator = np.random.default_rng(1)
        shift = torch.from_numpy(generator.uniform(-1.0, 1.0, 3))
        rotation = axis_angle_to_matrix(
            torch.from_numpy(generator.uniform(-1.0, 1.0, 3)).unsqueeze(0)
        )[0]
        moved = self.states.clone()
        for position_slice in (
            TCP_POSITION_SLICE,
            PEG_POSITION_SLICE,
            HOLE_POSITION_SLICE,
        ):
            moved[:, position_slice] = (
                self.states[:, position_slice] @ rotation.T + shift
            )
        for orientation_slice in (
            TCP_ORIENTATION_SLICE,
            PEG_ORIENTATION_SLICE,
            HOLE_ORIENTATION_SLICE,
        ):
            six_d = self.states[:, orientation_slice].reshape(-1, 2, 3)
            columns = torch.stack(
                [
                    six_d[:, 0],
                    six_d[:, 1],
                    torch.cross(six_d[:, 0], six_d[:, 1], dim=-1),
                ],
                dim=-1,
            )  # (batch_size, 3, 3)
            moved[:, orientation_slice] = (
                (rotation @ columns).transpose(-2, -1)[:, :2].reshape(-1, 6)
            )
        moved_features = self.geometry.project_observations(observation_batch=moved)
        torch.testing.assert_close(
            self.distances(moved_features, moved_features[5]),
            self.distances(self.features, self.features[5]),
            rtol=0.0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
