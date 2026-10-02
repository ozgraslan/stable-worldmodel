"""Adapter contract checks; enable rendered checks with RUN_MANISKILL_GPU_TESTS=1."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip('mani_skill')

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

from stable_worldmodel.envs.maniskill.pusht import PushTSWMEnv


class FakeScene(gym.Env):
    device = 'cpu'
    gpu_sim_enabled = False
    single_action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

    def __init__(self):
        self.tee = SimpleNamespace(
            pose=Pose.create(
                torch.tensor([[0.0, 0.0, 0.021, 1.0, 0.0, 0.0, 0.0]])
            )
        )
        self.tee.set_pose = lambda pose: setattr(self.tee, 'pose', pose)
        self.goal_tee = SimpleNamespace(
            pose=Pose.create(
                torch.tensor([[0.1, -0.1, 0.001, 1.0, 0.0, 0.0, 0.0]])
            )
        )
        self.steps = 0

    def get_obs(self):
        value = 99 if self.tee.pose.p[0, 0] > 0 else self.steps
        return {
            'agent': {'qpos': torch.zeros(1, 2)},
            'extra': {},
            'sensor_data': {
                'base_camera': {
                    'rgb': torch.full((1, 8, 8, 3), value, dtype=torch.uint8)
                }
            },
        }

    def get_state(self):
        return torch.tensor([[float(self.steps)]])

    def reset(self, **kwargs):
        self.steps = 0
        return self.get_obs(), {'success': torch.tensor([False])}

    def step(self, action):
        assert action.shape == (1, 2)
        self.steps += 1
        return (
            self.get_obs(),
            torch.tensor([1.0]),
            torch.tensor([True]),
            torch.tensor([False]),
            {'success': torch.tensor([True])},
        )


class AdapterTests(unittest.TestCase):
    def test_eef_position_goal_is_captured_and_required_for_success(self):
        class EEFScene(FakeScene):
            @property
            def agent(self):
                pose = SimpleNamespace(
                    p=torch.tensor([[0.0, 0.0, self.steps * 0.1]])
                )
                return SimpleNamespace(tcp=SimpleNamespace(pose=pose))

        scene = EEFScene()
        with patch(
            'stable_worldmodel.envs.maniskill.pusht.gym.make',
            return_value=scene,
        ):
            env = PushTSWMEnv(
                image_size=8,
                goal_step_distance=5,
                goal_eef_position_tolerance=0.02,
            )
        env._expert = SimpleNamespace(
            get_action=lambda *args, **kwargs: torch.zeros(1, 2)
        )
        _, info = env.reset(seed=42)
        # Expert terminates after one step, before the requested offset.
        torch.testing.assert_close(
            env._goal_eef_position, torch.tensor([[0.0, 0.0, 0.1]])
        )
        self.assertTrue(env.replay_report['passed'])
        self.assertAlmostEqual(env.replay_report['eef_position_error_m'], 0)
        self.assertFalse(info['success'])
        self.assertAlmostEqual(env._goal_errors()['eef_position_error_m'], 0.1)
        _, _, terminated, _, info = env.step(np.zeros(2))
        self.assertTrue(terminated)
        self.assertTrue(info['success'])
        # XYZ distance matters, but no EEF orientation is required.
        env._goal_eef_position[:, 0] += 0.01
        self.assertTrue(env._goal_reached())
        env._goal_eef_position[:, 1] += 0.03
        self.assertFalse(env._goal_reached())
        env._goal_eef_position = scene.agent.tcp.pose.p.clone()
        env._goal_pose[:, 0] += 0.1
        self.assertFalse(env._goal_reached())

    def test_video_uses_evaluation_goal_without_changing_policy_render(self):
        env = PushTSWMEnv.__new__(PushTSWMEnv)
        env._pixels = np.zeros((8, 8, 3), dtype=np.uint8)
        env._goal = np.full((8, 8, 3), 77, dtype=np.uint8)
        third_person = np.full((8, 8, 3), 33, dtype=np.uint8)

        def render_video(goal):
            self.assertIs(goal, env._goal)
            return torch.from_numpy(
                np.concatenate([env._pixels, third_person, goal], axis=1)
            )[None]

        env.env = SimpleNamespace(
            unwrapped=SimpleNamespace(render_video=render_video)
        )
        video = env.render_video()
        np.testing.assert_array_equal(video[:, :8], env._pixels)
        np.testing.assert_array_equal(video[:, 8:16], third_person)
        np.testing.assert_array_equal(video[:, 16:], env._goal)
        self.assertEqual(env.render().shape, (8, 8, 3))

    def test_overhead_rgb_expert_goal(self):
        expert = SimpleNamespace()
        expert.to = lambda device: expert
        expert.eval = lambda: None
        expert.load_state_dict = lambda weights: None
        inputs = []

        def get_action(obs, deterministic):
            inputs.append(obs)
            self.assertEqual(set(obs), {'rgb'})
            self.assertTrue(deterministic)
            return torch.zeros(1, 2)

        expert.get_action = get_action
        scene = FakeScene()
        with (
            patch(
                'stable_worldmodel.envs.maniskill.pusht.gym.make',
                return_value=scene,
            ) as make,
            patch(
                'stable_worldmodel.envs.maniskill.experts.RGBExpert',
                return_value=expert,
            ),
            patch(
                'stable_worldmodel.envs.maniskill.pusht.torch.load',
                return_value={},
            ),
        ):
            env = PushTSWMEnv(
                env_id='OverheadPushT-v1',
                image_size=8,
                control_mode='pd_ee_delta_xy',
                expert_policy_type='rgb',
                expert_checkpoint='/tmp/nonexistent-expert.pt',
                state_goal=True,
            )
            _, info = env.reset(seed=42)
        self.assertEqual(make.call_args.args[0], 'OverheadPushT-v1')
        self.assertEqual(
            make.call_args.kwargs['control_mode'], 'pd_ee_delta_xy'
        )
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0]['rgb'].shape, (1, 8, 8, 3))
        self.assertIsNotNone(env.goal_observation)
        self.assertTrue((info['goal'] == 1).all())
        self.assertEqual(scene.steps, 0)
        self.assertTrue(env.replay_report['passed'])
        np.testing.assert_array_equal(
            info['goal_obs_agent_qpos'], env.goal_observation['obs_agent_qpos']
        )

    def test_expert_frame_offset_and_reset(self):
        class ExpertScene(FakeScene):
            def get_obs(self):
                obs = super().get_obs()
                obs['agent']['qpos'] = torch.full((1, 2), float(self.steps))
                return obs

            def step(self, action):
                self.steps += 1
                return (
                    self.get_obs(),
                    torch.tensor([1.0]),
                    torch.tensor([self.steps == 3]),
                    torch.tensor([False]),
                    {'success': torch.tensor([self.steps == 3])},
                )

        for requested, expected in [(2, 2), (5, 3)]:
            with self.subTest(requested=requested):
                scene = ExpertScene()
                with patch(
                    'stable_worldmodel.envs.maniskill.pusht.gym.make',
                    return_value=scene,
                ):
                    env = PushTSWMEnv(
                        image_size=8, goal_step_distance=requested
                    )
                env._expert = SimpleNamespace(
                    get_action=lambda *args, **kwargs: torch.zeros(1, 2)
                )
                env.state_goal = True
                obs, info = env.reset(seed=42)
                np.testing.assert_array_equal(
                    env.goal_observation['obs_agent_qpos'],
                    [expected, expected],
                )
                np.testing.assert_array_equal(obs['obs_agent_qpos'], [0, 0])
                self.assertEqual(scene.steps, 0)
                self.assertTrue((env.render() == 0).all())
                self.assertTrue((info['goal'] == expected).all())
                self.assertEqual(info['goal_step_distance'], expected)
                self.assertEqual(info['expert_length'], 3)
                self.assertTrue(info['expert_task_success'])
                self.assertTrue(env.replay_report['passed'])
                self.assertEqual(env.replay_report['pixel_max_error'], 0)
                # Intermediate-goal completion checks the captured object pose.
                self.assertTrue(env._goal_reached())
                env._goal_pose[:, 0] += 0.1
                self.assertFalse(env._goal_reached())

    def test_replay_image_mismatch_is_rejected(self):
        class BrokenReset(FakeScene):
            def reset(self, **kwargs):
                self.resets = getattr(self, 'resets', 0) + 1
                return super().reset(**kwargs)

            def get_obs(self):
                obs = super().get_obs()
                if getattr(self, 'resets', 0) >= 2:
                    obs['sensor_data']['base_camera']['rgb'] += 5
                return obs

        scene = BrokenReset()
        with patch(
            'stable_worldmodel.envs.maniskill.pusht.gym.make',
            return_value=scene,
        ):
            env = PushTSWMEnv(image_size=8)
        env._expert = SimpleNamespace(
            get_action=lambda *a, **kw: torch.zeros(1, 2)
        )
        with self.assertRaisesRegex(RuntimeError, 'Expert replay failed'):
            env.reset(seed=42)

    def test_goal_render_restores_scene_and_gym_shapes(self):
        scene = FakeScene()
        with patch(
            'stable_worldmodel.envs.maniskill.pusht.gym.make',
            return_value=scene,
        ):
            env = PushTSWMEnv(image_size=8)
        before = scene.tee.pose.raw_pose.clone()
        obs, info = env.reset(seed=4)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(env.action_space.shape, (2,))
        self.assertTrue((info['goal'] == 99).all())
        self.assertTrue((env.render() == 0).all())
        torch.testing.assert_close(scene.tee.pose.raw_pose, before)
        self.assertEqual(scene.steps, 0)
        _, reward, terminated, truncated, info = env.step(np.zeros(2))
        self.assertEqual(reward, 1.0)
        self.assertIs(terminated, True)
        self.assertIs(truncated, False)
        self.assertTrue((info['goal'] == 99).all())
        with (
            patch.object(
                env, '_camera', side_effect=RuntimeError('render failure')
            ),
            self.assertRaisesRegex(RuntimeError, 'render failure'),
        ):
            env._render_goal()
        torch.testing.assert_close(scene.tee.pose.raw_pose, before)

    @unittest.skipUnless(
        os.environ.get('RUN_MANISKILL_GPU_TESTS') == '1',
        'requires rendering and ManiSkill assets',
    )
    def test_real_scene_and_swm_world(self):
        import stable_worldmodel as swm

        world = swm.World(
            'swm/ManiSkillPushT-v1',
            num_envs=1,
            image_shape=(64, 64),
            image_size=64,
        )
        try:
            world.reset(seed=42)
            adapter = world.envs.envs[0].unwrapped
            base = adapter.env.unwrapped
            before = base.get_state().clone()
            goal = adapter._render_goal()
            torch.testing.assert_close(base.get_state(), before)
            self.assertEqual(goal.shape, (64, 64, 3))
            self.assertEqual(world.infos['pixels'].shape, (1, 1, 64, 64, 3))
            self.assertEqual(world.infos['goal'].shape, (1, 1, 64, 64, 3))
            world.envs.step(
                np.zeros(world.envs.action_space.shape, dtype=np.float32)
            )
        finally:
            world.close()


def test_current_state_expert_query_does_not_advance_live_scene():
    live = FakeScene()
    oracle = FakeScene()
    env = PushTSWMEnv.__new__(PushTSWMEnv)
    env.env = live
    env.env_id = 'test'
    env.camera_name = 'base_camera'
    env.image_size = 8
    env.expert_policy_type = 'rgb'
    env.action_space = live.single_action_space
    env._planning_oracle = oracle
    env._evaluation_seed = 42
    env._evaluation_options = {}
    env._executed_actions = [np.zeros(2), np.zeros(2)]
    env._expert = SimpleNamespace(
        get_action=lambda obs, **kwargs: torch.full(
            (1, 2), float(obs['rgb'][0, 0, 0, 0]) / 10
        )
    )
    for action in env._executed_actions:
        live.step(torch.as_tensor(action)[None])
    before = live.get_state().clone()
    goal = np.full((8, 8, 3), 77, dtype=np.uint8)
    env._goal = goal.copy()
    actions, metadata = env.planning_expert_actions(3)
    np.testing.assert_allclose(actions[:, 0], [0.2, 0.3, 0.4])
    torch.testing.assert_close(live.get_state(), before)
    np.testing.assert_array_equal(env._goal, goal)
    assert len(env._executed_actions) == 2
    assert metadata['env_step'] == 2
    assert oracle.steps == 5
    # Repeated queries reset and replay, rather than continuing the oracle.
    repeated, _ = env.planning_expert_actions(3)
    np.testing.assert_array_equal(repeated, actions)
    torch.testing.assert_close(live.get_state(), before)


def test_current_state_expert_query_rejects_unmatched_replay():
    env = PushTSWMEnv.__new__(PushTSWMEnv)
    env.env = FakeScene()
    env._planning_oracle = FakeScene()
    env._expert = SimpleNamespace()
    env._evaluation_seed = 42
    env._evaluation_options = {}
    env._executed_actions = [np.zeros(2)]
    with pytest.raises(AssertionError, match='current MPC state'):
        env.planning_expert_actions(2)
    assert env.env.steps == 0


if __name__ == '__main__':
    unittest.main()
