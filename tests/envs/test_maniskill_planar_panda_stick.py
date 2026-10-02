"""Controller contract tests without rendering or a physics allocation."""

import os
import unittest
from types import SimpleNamespace

import pytest

pytest.importorskip('mani_skill')
from unittest.mock import Mock

import torch
from gymnasium.vector.utils import batch_space
from mani_skill.utils.structs import Pose

from stable_worldmodel.envs.maniskill.planar_panda_stick import (
    PDEEPlanarController,
    PlanarPandaStick,
)


class PlanarControllerTests(unittest.TestCase):
    def setUp(self):
        config = PlanarPandaStick.__new__(
            PlanarPandaStick
        )._controller_configs['pd_ee_delta_xy']['arm']
        self.c = c = config.controller_cls.__new__(config.controller_cls)
        c.config = config
        c.scene = SimpleNamespace(
            num_envs=2,
            gpu_sim_enabled=True,
            _reset_mask=torch.tensor([True, True]),
        )
        c.articulation = SimpleNamespace(
            device=torch.device('cpu'), get_qpos=lambda: torch.zeros(2, 7)
        )
        c.active_joint_indices = torch.arange(7)
        c.root_link = SimpleNamespace(
            pose=Pose.create_from_pq(p=torch.zeros(2, 3))
        )
        c.ee_link = SimpleNamespace(
            pose=Pose.create_from_pq(
                p=torch.tensor([[0.1, 0.2, 0.03], [0.3, 0.4, 0.04]])
            )
        )
        c._initialize_action_space()
        c._normalize_action = True
        c._clip_and_scale_action_space()
        c.action_space = batch_space(c.single_action_space, 2)
        c.kinematics = SimpleNamespace(
            compute_ik=Mock(return_value=torch.zeros(2, 7))
        )
        c.set_drive_targets = Mock()
        c.reset()

    def test_xy_scaling_and_fixed_pose_despite_measured_drift(self):
        c = self.c
        self.assertEqual(c.single_action_space.shape, (2,))
        anchor = c._target_pose.raw_pose.clone()
        c.ee_link.pose = Pose.create_from_pq(
            p=torch.tensor([[0.2, 0.3, 0.08], [0.4, 0.5, 0.01]]),
            q=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        )
        c.set_action(torch.tensor([[1.0, -1.0], [2.0, -2.0]]))
        torch.testing.assert_close(
            c._target_pose.p[:, :2], torch.tensor([[0.3, 0.2], [0.5, 0.4]])
        )
        torch.testing.assert_close(
            c._target_pose.raw_pose[:, 2:], anchor[:, 2:]
        )
        call = c.kinematics.compute_ik.call_args.kwargs
        self.assertFalse(call['is_delta_pose'])
        torch.testing.assert_close(
            call['pose'].raw_pose, c._target_pose.raw_pose
        )
        # Repeating the action must not accumulate an unreachable XY target.
        c.set_action(torch.tensor([[1.0, -1.0], [1.0, -1.0]]))
        torch.testing.assert_close(
            c._target_pose.p[:, :2], torch.tensor([[0.3, 0.2], [0.5, 0.4]])
        )

    def test_partial_reset_and_state_restore_preserve_anchors(self):
        c = self.c
        original = c._target_pose.raw_pose.clone()
        c.scene._reset_mask = torch.tensor([False, True])
        c.ee_link.pose = Pose.create_from_pq(
            p=torch.tensor([[0.2, 0.3, 0.08], [0.4, 0.5, 0.02]])
        )
        c.reset()
        torch.testing.assert_close(c._target_pose.raw_pose[0], original[0])
        torch.testing.assert_close(
            c._target_pose.raw_pose[1], c.ee_pose_at_base.raw_pose[1]
        )
        saved = {key: value.clone() for key, value in c.get_state().items()}
        c.set_action(torch.ones(2, 2))
        c.set_state(saved)
        torch.testing.assert_close(
            c._target_pose.raw_pose, saved['target_pose']
        )


@unittest.skipUnless(
    os.environ.get('RUN_MANISKILL_GPU_TESTS') == '1',
    'requires a GPU simulator',
)
class OverheadPlanarIntegrationTests(unittest.TestCase):
    def test_rgb_actions_height_and_partial_reset(self):
        import gymnasium as gym
        import numpy as np
        import sapien

        from stable_worldmodel.envs.maniskill import overhead_pusht

        env = gym.make(
            'OverheadPushT-v1',
            num_envs=2,
            obs_mode='rgb',
            control_mode='pd_ee_delta_xy',
            sim_backend='physx_cuda',
            reconfiguration_freq=0,
            robot_init_qpos_noise=0,
        )
        try:
            obs, _ = env.reset(seed=123)
            base = env.unwrapped
            self.assertEqual(base.single_action_space.shape, (2,))
            self.assertEqual(
                obs['sensor_data']['overhead_camera']['rgb'].shape,
                (2, 128, 128, 3),
            )
            self.assertEqual(set(obs['sensor_data']), {'overhead_camera'})
            before_video = base.get_state().clone()
            video = base.render_video()
            self.assertEqual(video.shape, (2, 128, 384, 3))
            self.assertGreater(video.float().std().item(), 0)
            torch.testing.assert_close(
                base.get_state(), before_video, rtol=0, atol=1e-6
            )
            goal_panel = video[:, :, 256:].clone()
            base.render_mode = 'all'
            torch.testing.assert_close(base.render()[:, :, 256:], goal_panel)
            c = base.agent.controller.controllers['arm']
            self.assertIsInstance(c, PDEEPlanarController)
            anchor = c._target_pose.raw_pose.clone()
            for _ in range(20):
                env.step(
                    torch.tensor(
                        [[0.02, 0.0], [0.0, -0.02]], device=base.device
                    )
                )
                torch.testing.assert_close(
                    c._target_pose.raw_pose[:, 2:], anchor[:, 2:]
                )
            self.assertTrue(torch.isfinite(base.agent.robot.get_qpos()).all())
            self.assertLess(
                (c.ee_pose_at_base.p[:, 2] - anchor[:, 2]).abs().max().item(),
                0.01,
            )
            before_reset = c._target_pose.raw_pose.clone()
            env.reset(
                options={'env_idx': torch.tensor([1], device=base.device)}
            )
            torch.testing.assert_close(
                c._target_pose.raw_pose[0], before_reset[0]
            )
            torch.testing.assert_close(
                c._target_pose.raw_pose[1], c.ee_pose_at_base.raw_pose[1]
            )
            for entity in base.table_scene.table._objs:
                body = entity.find_component_by_type(
                    sapien.render.RenderBodyComponent
                )
                for shape in body.render_shapes:
                    for part in shape.parts:
                        np.testing.assert_allclose(
                            part.material.base_color,
                            overhead_pusht.OverheadPushTEnv.TABLE_COLOR,
                        )
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
