"""CPU peg insertion that retains the robot and table between episodes."""

import numpy as np
import sapien
import torch
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Pose

TASK_ID = "ReusablePegInsertionSide-v1"


@register_env(TASK_ID, max_episode_steps=450)
class ReusablePegInsertionSideEnv(PegInsertionSideEnv):
    """Rebuild seeded task geometry without reloading the unchanged CPU robot.

    BaseEnv.reset still seeds geometry and episode initialization separately,
    clears velocities, and resets the agent/controllers on every episode.
    """

    def _reconfigure(self, options: dict | None = None):
        options = {} if options is None else options
        # Rendering and GPU simulation keep the upstream scene lifecycle.
        if self.scene is None or self.gpu_sim_enabled or self.scene.can_render():
            return super()._reconfigure(options)

        self.remove_from_state_dict_registry(self.peg)
        self.remove_from_state_dict_registry(self.box)
        for i in range(self.num_envs):
            for name in (f"peg_{i}", f"box_with_hole_{i}"):
                actor = self.scene.actors.pop(name)
                # Actor._objs contains Entities. ManiSkill's remove_actor helper
                # in the installed version incorrectly accesses Entity.entity.
                self.scene.sub_scenes[i].remove_entity(actor._objs[0])
        self._load_objects()
        self.scene._setup(enable_gpu=False)
        self._reconfig_counter = self.reconfiguration_freq
        self.__dict__.pop("segmentation_id_map", None)

    def _load_objects(self):
        # Mirrors ManiSkill's PegInsertionSideEnv._load_scene geometry, omitting
        # the unchanged table and unused visuals. Keep RNG draw order and collision shapes
        # identical; integration tests compare the full state against upstream.
        with torch.device(self.device):
            lengths = np.asarray(self._batched_episode_rng.uniform(0.085, 0.125))
            radii = np.asarray(self._batched_episode_rng.uniform(0.015, 0.025))
            centers = (
                0.5
                * (lengths - radii)[:, None]
                * self._batched_episode_rng.uniform(-1, 1, size=(2,))
            )

            # save some useful values for use later
            self.peg_half_sizes = common.to_tensor(np.vstack([lengths, radii, radii])).T
            peg_head_offsets = torch.zeros((self.num_envs, 3))
            peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
            self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

            box_hole_offsets = torch.zeros((self.num_envs, 3))
            box_hole_offsets[:, 1:] = common.to_tensor(centers)
            self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
            self.box_hole_radii = common.to_tensor(radii + self._clearance)

            # in each parallel env we build a different box with a hole and peg (the task is meant to be quite difficult)
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
                # box with hole

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

            # to support heterogeneous simulation state dictionaries we register merged versions
            # of the parallel actors
            self.add_to_state_dict_registry(self.peg)
            self.add_to_state_dict_registry(self.box)


def _build_collision_box_with_hole(scene, inner_radius, outer_radius, depth, center):
    """Same four collision boxes as ManiSkill, without unused render materials."""
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
