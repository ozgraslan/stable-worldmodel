"""Panda stick controller with XY actions and a reset-anchored Z/orientation."""

from copy import deepcopy

import numpy as np
import torch
from gymnasium import spaces
from mani_skill.agents.controllers.pd_ee_pose import PDEEPosController
from mani_skill.agents.robots.panda.panda_stick import PandaStick
from mani_skill.utils.structs import Pose


class PDEEPlanarController(PDEEPosController):
    """XY deltas in the robot base frame; hold reset height and orientation.

    use_target=True keeps the anchor in the parent's serialized target pose and
    lets its reset implementation update only the environments being reset.
    XY remains relative to the measured TCP, avoiding accumulated XY targets.
    """

    def _initialize_action_space(self):
        self.single_action_space = spaces.Box(
            np.broadcast_to(self.config.pos_lower, 2).astype(np.float32),
            np.broadcast_to(self.config.pos_upper, 2).astype(np.float32),
            dtype=np.float32,
        )

    def _preprocess_action(self, action):
        xy = super()._preprocess_action(action)
        return torch.cat((xy, torch.zeros_like(xy[:, :1])), dim=-1)

    def compute_target_pose(self, prev_ee_pose_at_base, action):
        position = prev_ee_pose_at_base.p.clone()
        position[:, :2] = self.ee_pose_at_base.p[:, :2] + action[:, :2]
        return Pose.create_from_pq(
            p=position, q=prev_ee_pose_at_base.q.clone()
        )


class PlanarPandaStick(PandaStick):
    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        planar = deepcopy(configs['pd_ee_delta_pos']['arm'])
        # Retain PDEEPosControllerConfig's exact type for ManiSkill's position
        # controller dispatch, while replacing the controller implementation.
        planar.controller_cls = PDEEPlanarController
        planar.use_target = True
        configs['pd_ee_delta_xy'] = {'arm': planar}
        return configs
