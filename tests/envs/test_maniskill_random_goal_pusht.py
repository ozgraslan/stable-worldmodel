"""Geometry/reward tests; add RUN_MANISKILL_GPU_TESTS=1 for simulator checks."""

import os
import unittest
from types import SimpleNamespace

import pytest

pytest.importorskip('mani_skill')

import torch

from stable_worldmodel.envs.maniskill.random_goal_pusht import (
    RandomGoalPushTEnv,
    quaternion_yaw,
    world_to_goal,
)


def yaw_quaternion(yaw):
    q = torch.zeros((len(yaw), 4), device=yaw.device)
    q[:, 0], q[:, 3] = (yaw / 2).cos(), (yaw / 2).sin()
    return q


class GoalTests(unittest.TestCase):
    def test_transform_and_quaternion_sign(self):
        yaw = torch.tensor([-2.0, -0.5, 0.0, 0.7, 2.8])
        q = yaw_quaternion(yaw)
        torch.testing.assert_close(quaternion_yaw(q), yaw)
        torch.testing.assert_close(quaternion_yaw(-q), yaw)
        positions = torch.tensor([[0.1, -0.2, 0.001]]).repeat(5, 1)
        transform = world_to_goal(positions, q)
        origin = torch.cat((positions[:, :2], torch.ones((5, 1))), dim=-1)
        torch.testing.assert_close(
            (transform @ origin.unsqueeze(-1)).squeeze(-1),
            torch.tensor([[0.0, 0.0, 1.0]]).repeat(5, 1),
        )
        x_axis = origin.clone()
        x_axis[:, :2] += torch.stack((yaw.cos(), yaw.sin()), dim=-1)
        torch.testing.assert_close(
            (transform @ x_axis.unsqueeze(-1)).squeeze(-1),
            torch.tensor([[1.0, 0.0, 1.0]]).repeat(5, 1),
        )

    def test_reward_tracks_each_goal_orientation(self):
        yaw = torch.tensor([-1.0, 1.5])
        position = torch.tensor([[-0.1, -0.1, 0.021], [-0.2, 0.0, 0.021]])
        actor = lambda p, q: SimpleNamespace(pose=SimpleNamespace(p=p, q=q))
        env = SimpleNamespace(
            tee=actor(position, yaw_quaternion(yaw)),
            goal_tee=actor(position, yaw_quaternion(yaw)),
            agent=SimpleNamespace(tcp=actor(position, yaw_quaternion(yaw))),
        )
        info = {'success': torch.zeros(2, dtype=torch.bool)}
        matching = RandomGoalPushTEnv.compute_dense_reward(
            env, None, None, info
        )
        torch.testing.assert_close(matching, torch.full((2,), 1.05))
        env.goal_tee.pose.q = yaw_quaternion(yaw + torch.pi)
        rotated = RandomGoalPushTEnv.compute_dense_reward(
            env, None, None, info
        )
        self.assertTrue((rotated < matching).all())

    @unittest.skipUnless(
        os.environ.get('RUN_MANISKILL_GPU_TESTS') == '1',
        'requires a GPU simulator',
    )
    def test_gpu_resets_overlap_and_observations(self):
        import gymnasium as gym
        from mani_skill.utils.common import flatten_state_dict
        from mani_skill.utils.structs.pose import Pose

        env = gym.make(
            'RandomGoalPushT-v1',
            num_envs=4,
            obs_mode='state',
            sim_backend='physx_cuda',
            reconfiguration_freq=0,
        )
        try:
            obs, _ = env.reset(seed=123)
            base = env.unwrapped
            goals = base.goal_tee.pose.raw_pose.clone()
            transforms = base.world_to_goal_trans.clone()
            self.assertEqual(torch.unique(goals, dim=0).shape[0], 4)
            center = torch.tensor([-0.156, -0.1], device=base.device)
            self.assertTrue(((goals[:, :2] - center).abs() <= 0.08001).all())
            torch.testing.assert_close(
                obs,
                flatten_state_dict(
                    base.get_obs(unflattened=True), use_torch=True
                ),
            )
            self.assertEqual(
                base.get_obs(unflattened=True)['extra'][
                    'goal_orientation'
                ].shape,
                (4, 2),
            )
            env.reset(
                options={'env_idx': torch.tensor([1], device=base.device)}
            )
            torch.testing.assert_close(
                base.goal_tee.pose.raw_pose[[0, 2, 3]], goals[[0, 2, 3]]
            )
            torch.testing.assert_close(
                base.world_to_goal_trans[[0, 2, 3]], transforms[[0, 2, 3]]
            )
            self.assertFalse(
                torch.equal(base.goal_tee.pose.raw_pose[1], goals[1])
            )
            position = base.goal_tee.pose.p.clone()
            position[:, 2] = 0.021
            base.tee.set_pose(
                Pose.create_from_pq(p=position, q=base.goal_tee.pose.q)
            )
            base.scene._gpu_apply_all()
            base.scene._gpu_fetch_all()
            info = base.get_info()
            self.assertTrue(
                info['success'].all(), base.pseudo_render_intersection()
            )
            torch.testing.assert_close(
                base.compute_normalized_dense_reward(None, None, info),
                torch.ones(4, device=base.device),
            )
            # Matching reset seeds reproduce randomized targets.
            env.reset(seed=123)
            torch.testing.assert_close(base.goal_tee.pose.raw_pose, goals)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
