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

    @property
    def agent(self):
        pose = SimpleNamespace(p=torch.tensor([[0.0, 0.0, self.steps * 0.1]]))
        return SimpleNamespace(tcp=SimpleNamespace(pose=pose))

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

        for requested, expected, beginning, start in [
            (2, 2, True, 0),
            (5, 3, True, 0),
            (2, 3, False, 1),
            (5, 3, False, 0),
        ]:
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
                obs, info = env.reset(
                    seed=0, options={'start_from_beginning': beginning}
                )
                np.testing.assert_array_equal(
                    env.goal_observation['obs_agent_qpos'],
                    [expected, expected],
                )
                np.testing.assert_array_equal(
                    obs['obs_agent_qpos'], [start, start]
                )
                self.assertEqual(scene.steps, start)
                self.assertTrue((env.render() == start).all())
                self.assertTrue((info['goal'] == expected).all())
                self.assertEqual(info['goal_step_distance'], expected - start)
                self.assertEqual(info['expert_start_step'], start)
                self.assertEqual(info['expert_length'], 3 - start)
                self.assertEqual(len(env._executed_actions), start)
                self.assertEqual(env.expert_goal_index, expected - start)
                self.assertEqual(len(env.expert_actions), 3 - start)
                self.assertTrue((env.expert_frames[0] == start).all())
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


class RetryScene(FakeScene):
    """A seed controls native success and physical start-goal displacement."""

    def __init__(self, settings, retries=()):
        super().__init__()
        self.settings = dict(settings)
        self.retry_settings = iter(retries)
        self.seed = next(iter(settings))
        self.reset_calls = []

    @property
    def agent(self):
        eef_step = self.settings[self.seed][2]
        pose = SimpleNamespace(
            p=torch.tensor([[self.steps * eef_step, 0.0, 0.1]])
        )
        return SimpleNamespace(tcp=SimpleNamespace(pose=pose))

    def get_obs(self):
        object_step = self.settings[self.seed][1]
        pose = self.tee.pose.raw_pose.clone()
        pose[:, 0] = self.steps * object_step
        self.tee.set_pose(Pose.create(pose))
        obs = super().get_obs()
        obs['agent']['qpos'] = torch.full((1, 2), float(self.steps))
        return obs

    def get_state(self):
        return torch.tensor([[float(self.seed), float(self.steps)]])

    def reset(self, *, seed, options):
        self.seed = seed
        if seed not in self.settings:
            self.settings[seed] = next(self.retry_settings)
        self.reset_calls.append((seed, dict(options)))
        return super().reset()

    def step(self, action):
        self.steps += 1
        done = self.steps == 2
        success = done and self.settings[self.seed][0]
        return (
            self.get_obs(),
            torch.tensor([0.0]),
            torch.tensor([success]),
            torch.tensor([done and not success]),
            {'success': torch.tensor([success])},
        )


def retry_env(scene, **kwargs):
    with patch(
        'stable_worldmodel.envs.maniskill.pusht.gym.make', return_value=scene
    ):
        env = PushTSWMEnv(image_size=8, goal_step_distance=1, **kwargs)
    env._expert = SimpleNamespace(
        get_action=lambda *args, **kwargs: torch.zeros(1, 2)
    )
    env.state_goal = True
    return env


@pytest.mark.parametrize('beginning', [True, False])
def test_expert_retries_failure_and_static_pair_with_reproducible_seed(
    tmp_path, beginning
):
    scene = RetryScene(
        {10: (False, 0.04, 0.05)},
        retries=[(True, 0.001, 0.001), (True, 0.03, 0.04)],
    )
    env = retry_env(scene, expert_max_attempts=3, expert_output_dir=tmp_path)
    options = {'start_from_beginning': beginning, 'custom_option': 7}
    obs, info = env.reset(seed=10, options=options)
    assert info['expert_rollout_attempts'] == 3
    assert info['expert_task_success']
    assert info['expert_requested_seed'] == 10
    accepted_seed = info['expert_seed']
    assert accepted_seed == env._evaluation_seed == scene.seed
    attempted_seeds = [seed for seed, _ in scene.reset_calls[:3]]
    assert len(set(attempted_seeds)) == 3
    assert attempted_seeds[0] == 10
    assert attempted_seeds[1] != 11
    assert accepted_seed not in {10, 11, 12}
    assert info['expert_start_goal_object_distance_m'] == pytest.approx(0.03)
    assert info['expert_start_goal_eef_distance_m'] == pytest.approx(0.04)
    start = info['expert_start_step']
    assert scene.steps == start == len(env._executed_actions)
    np.testing.assert_array_equal(obs['obs_agent_qpos'], [start, start])
    np.testing.assert_array_equal(
        info['goal_obs_agent_qpos'], [start + 1, start + 1]
    )
    assert env.expert_goal_index == 1
    assert env.replay_report['passed']
    assert all(
        options == {'custom_option': 7} for _, options in scene.reset_calls
    )
    with np.load(tmp_path / f'expert_seed_{accepted_seed}.npz') as saved:
        assert saved['rollout_attempts'] == 3
        assert saved['requested_seed'] == 10
        assert saved['seed'] == accepted_seed
        assert saved['eef_position'].shape == (3, 3)
    assert not (tmp_path / 'expert_seed_10.npz').exists()
    _, repeated = env.reset(seed=10, options=options)
    assert repeated['expert_seed'] == accepted_seed
    assert repeated['expert_start_step'] == start


@pytest.mark.parametrize('beginning', [True, False])
def test_retry_seed_does_not_overlap_adjacent_evaluation_reset(beginning):
    scene = RetryScene(
        {10: (False, 0.04, 0.05), 11: (True, 0.03, 0.04)},
        retries=[(True, 0.03, 0.04)],
    )
    env = retry_env(scene, expert_max_attempts=2)
    options = {'start_from_beginning': beginning}
    _, first = env.reset(seed=10, options=options)
    _, second = env.reset(seed=11, options=options)
    assert first['expert_rollout_attempts'] == 2
    assert second['expert_rollout_attempts'] == 1
    assert first['expert_seed'] != second['expert_seed'] == 11


def test_retry_seed_generation_skips_previously_attempted_seeds():
    scene = RetryScene(
        {10: (False, 0.04, 0.05)},
        retries=[(False, 0.04, 0.05), (True, 0.03, 0.04)],
    )
    env = retry_env(scene, expert_max_attempts=3)
    # The retry RNG first repeats the original seed, then a previous retry.
    rng = SimpleNamespace(integers=lambda *args: next(draws))
    draws = iter([10, 1000, 1000, 2000])
    with patch(
        'stable_worldmodel.envs.maniskill.pusht.np.random.default_rng',
        return_value=rng,
    ):
        _, info = env.reset(seed=10)
    assert [seed for seed, _ in scene.reset_calls[:3]] == [10, 1000, 2000]
    assert info['expert_seed'] == 2000


@pytest.mark.parametrize(
    'object_step,eef_step',
    [
        (0.001, 0.03),
        (0.03, 0.001),
        (0.03, 0.03),
        (0.03125, 0.03125),
    ],
)
def test_expert_accepts_pair_unless_both_displacements_are_below_threshold(
    object_step, eef_step
):
    scene = RetryScene({10: (True, object_step, eef_step)})
    # Use an exactly representable threshold for the equality case.
    threshold = 0.03125 if object_step == eef_step == 0.03125 else 0.02
    env = retry_env(
        scene,
        expert_max_attempts=1,
        expert_min_object_displacement=threshold,
        expert_min_eef_displacement=threshold,
    )
    _, info = env.reset(seed=10)
    assert info['expert_rollout_attempts'] == 1
    assert info['expert_seed'] == 10


@pytest.mark.parametrize(
    'success,object_step,eef_step,reason',
    [
        (False, 0.03, 0.04, 'did not solve'),
        (True, 0.001, 0.001, 'movement too small'),
    ],
)
def test_expert_retries_stop_at_attempt_limit(
    success, object_step, eef_step, reason
):
    scene = RetryScene(
        {10: (success, object_step, eef_step)},
        retries=[(success, object_step, eef_step)],
    )
    env = retry_env(scene, expert_max_attempts=2)
    with pytest.raises(RuntimeError, match=f'after 2 attempts.*{reason}'):
        env.reset(seed=10)
    attempted = [seed for seed, _ in scene.reset_calls]
    assert len(attempted) == len(set(attempted)) == 2
    assert attempted[0] == 10
    assert attempted[1] != 11


@pytest.mark.parametrize(
    'settings',
    [
        {'expert_max_attempts': 0},
        {'expert_max_attempts': 1.5},
        {'expert_max_attempts': True},
        {'expert_min_object_displacement': -0.01},
        {'expert_min_object_displacement': float('nan')},
        {'expert_min_eef_displacement': float('inf')},
    ],
)
def test_expert_retry_settings_are_validated_before_simulator_creation(
    settings,
):
    with patch('stable_worldmodel.envs.maniskill.pusht.gym.make') as make:
        with pytest.raises(ValueError):
            PushTSWMEnv(**settings)
        make.assert_not_called()


if __name__ == '__main__':
    unittest.main()


def test_comparison_replays_both_sequences_without_advancing_live_scene(
    tmp_path, monkeypatch
):
    live = FakeScene()
    oracle = FakeScene()
    live.steps = 2
    env = PushTSWMEnv.__new__(PushTSWMEnv)
    env.env = live
    env.camera_name = 'base_camera'
    env.image_size = 8
    env.action_space = live.single_action_space
    env._planning_oracle = oracle
    env._evaluation_seed = 42
    env._evaluation_options = {}
    env._executed_actions = [np.zeros(2), np.zeros(2)]
    env._goal = np.full((8, 8, 3), 77, dtype=np.uint8)
    env._goal_pose = live.tee.pose.raw_pose.clone()
    env.goal_position_tolerance = 0.02
    env.goal_yaw_tolerance = 0.15
    env.goal_eef_position_tolerance = None
    captured = []

    def save(output, panels, fps):
        output.mkdir(parents=True)
        captured.append(panels)

    monkeypatch.setattr('stable_worldmodel.plot.save_panel_videos', save)
    results = env.planning_action_comparison(
        {
            'expert': np.zeros((3, 2)),
            'selected': np.ones((3, 2)),
        },
        tmp_path / 'comparison',
    )
    assert live.steps == 2
    assert len(env._executed_actions) == 2
    panels = captured[0]
    assert set(panels) == {'expert', 'selected', 'goal'}
    for name in ('expert', 'selected'):
        assert panels[name][0].shape == (4, 8, 8, 3)
        assert (panels[name][0][0] == 2).all()
        assert (panels[name][0][-1] == 5).all()
        assert results[name]['final_goal_success']
    with np.load(tmp_path / 'comparison' / 'actions.npz') as saved:
        np.testing.assert_array_equal(saved['expert'], np.zeros((3, 2)))
        np.testing.assert_array_equal(saved['selected'], np.ones((3, 2)))
