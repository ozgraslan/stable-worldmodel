"""PushT with independently sampled goal position and orientation each reset."""

import math

import torch
from mani_skill.envs.tasks.tabletop.push_t import PushTEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


def quaternion_yaw(quaternion):
    """Signed yaw of a wxyz quaternion, invariant to quaternion sign."""
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def world_to_goal(position, quaternion):
    yaw = quaternion_yaw(quaternion)
    c, s = yaw.cos(), yaw.sin()
    transform = torch.eye(
        3, device=position.device, dtype=position.dtype
    ).repeat(len(position), 1, 1)
    transform[:, 0, 0], transform[:, 0, 1] = c, s
    transform[:, 1, 0], transform[:, 1, 1] = -s, c
    transform[:, 0, 2] = -c * position[:, 0] - s * position[:, 1]
    transform[:, 1, 2] = s * position[:, 0] - c * position[:, 1]
    return transform


@register_env('RandomGoalPushT-v1', max_episode_steps=100)
class RandomGoalPushTEnv(PushTEnv):
    """Original block/robot reset distribution; goal XY +/-8 cm, yaw +/-pi.

    Goal sampling is independent of block initialization. The state adds
    goal_orientation=[cos(yaw), sin(yaw)] to the original PushT fields.
    """

    def __init__(self, *args, goal_xy_range=0.08, **kwargs):
        if not math.isfinite(goal_xy_range) or not 0 <= goal_xy_range <= 0.1:
            raise ValueError('goal_xy_range must be between 0 and 0.1 meters')
        self.goal_xy_range = goal_xy_range
        super().__init__(*args, **kwargs)

    def _load_scene(self, options):
        super()._load_scene(options)
        self.world_to_goal_trans = self.world_to_goal_trans.repeat(
            self.num_envs, 1, 1
        )

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        count = len(env_idx)
        position = self.goal_tee.pose.p[env_idx].clone()
        position[:, :2] += (
            torch.rand((count, 2), device=self.device) * 2 - 1
        ) * self.goal_xy_range
        yaw = (torch.rand(count, device=self.device) * 2 - 1) * torch.pi
        quaternion = torch.zeros((count, 4), device=self.device)
        quaternion[:, 0], quaternion[:, 3] = (yaw / 2).cos(), (yaw / 2).sin()
        # ManiSkill's reset mask restricts this batched actor update to env_idx.
        self.goal_tee.set_pose(Pose.create_from_pq(p=position, q=quaternion))
        self.world_to_goal_trans[env_idx] = world_to_goal(position, quaternion)

    def quat_to_z_euler(self, quats):
        # Keep the parent overlap renderer and goal transforms on the same
        # signed-yaw convention, independent of the quaternion's sign.
        return quaternion_yaw(quats)

    def _get_obs_extra(self, info):
        obs = super()._get_obs_extra(info)
        if self.obs_mode_struct.use_state:
            yaw = quaternion_yaw(self.goal_tee.pose.q)
            obs['goal_orientation'] = torch.stack(
                (yaw.cos(), yaw.sin()), dim=-1
            )
        return obs

    def compute_dense_reward(self, obs, action, info):
        # Preserve the original shaping weights, using the actual per-env goal.
        yaw_error = quaternion_yaw(self.tee.pose.q) - quaternion_yaw(
            self.goal_tee.pose.q
        )
        reward = ((yaw_error.cos() + 1) / 2).square() / 2
        distance = torch.linalg.vector_norm(
            self.tee.pose.p[:, :2] - self.goal_tee.pose.p[:, :2], dim=-1
        )
        reward += (1 - torch.tanh(5 * distance)).square() / 2
        tcp_distance = torch.linalg.vector_norm(
            self.tee.pose.p - self.agent.tcp.pose.p, dim=-1
        )
        reward += (1 - torch.tanh(5 * tcp_distance)).sqrt() / 20
        reward[info['success']] = 3
        return reward
