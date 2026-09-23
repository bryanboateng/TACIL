"""CPU peg insertion that skips the render materials a headless run never draws."""

import numpy as np
import sapien
import torch
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose

TASK_ID = "HeadlessPegInsertionSide-v1"


@register_env(TASK_ID, max_episode_steps=450)
class HeadlessPegInsertionSideEnv(PegInsertionSideEnv):
    """Build the same collision geometry as ManiSkill without its unused visuals.

    BaseEnv._reconfigure still clears and rebuilds the whole scene on every
    episode, so resets stay independent of how many episodes ran before.
    """

    def _load_scene(self, options: dict):
        if self.scene.can_render():
            return super()._load_scene(options)

        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            lengths = np.asarray(self._batched_episode_rng.uniform(0.085, 0.125))
            radii = np.asarray(self._batched_episode_rng.uniform(0.015, 0.025))
            centers = (
                0.5
                * (lengths - radii)[:, None]
                * self._batched_episode_rng.uniform(-1, 1, size=(2,))
            )

            self.peg_half_sizes = common.to_tensor(np.vstack([lengths, radii, radii])).T
            peg_head_offsets = torch.zeros((self.num_envs, 3))
            peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
            self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

            box_hole_offsets = torch.zeros((self.num_envs, 3))
            box_hole_offsets[:, 1:] = common.to_tensor(centers)
            self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
            self.box_hole_radii = common.to_tensor(radii + self._clearance)

            pegs = []
            boxes = []

            for i in range(self.num_envs):
                scene_idxs = [i]
                length = lengths[i]
                radius = radii[i]
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[length, radius, radius])
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                builder.set_scene_idxs(scene_idxs)
                peg = builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)

                inner_radius, outer_radius, depth = (
                    radius + self._clearance,
                    length,
                    length,
                )
                builder = _build_collision_box_with_hole(
                    self.scene, inner_radius, outer_radius, depth, center=centers[i]
                )
                builder.initial_pose = sapien.Pose(p=[0, 1, 0.1])
                builder.set_scene_idxs(scene_idxs)
                box = builder.build_kinematic(f"box_with_hole_{i}")
                self.remove_from_state_dict_registry(box)
                pegs.append(peg)
                boxes.append(box)

            self.peg = Actor.merge(pegs, "peg")
            self.box = Actor.merge(boxes, "box_with_hole")

            self.add_to_state_dict_registry(self.peg)
            self.add_to_state_dict_registry(self.box)


def _build_collision_box_with_hole(scene, inner_radius, outer_radius, depth, center):
    """Build ManiSkill's four collision boxes without its unused render materials."""
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]),
        sapien.Pose([0, -offset + half_center[0], 0]),
        sapien.Pose([0, 0, offset + half_center[1]]),
        sapien.Pose([0, 0, -offset + half_center[1]]),
    ]
    for half_size, pose in zip(half_sizes, poses, strict=True):
        builder.add_box_collision(pose, half_size)
    return builder
