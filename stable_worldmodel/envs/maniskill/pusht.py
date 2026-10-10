"""Gymnasium/SWM bridge for RandomGoal and Overhead ManiSkill PushT.

Gymnasium imports this module via swm/ManiSkillPushT-v1. With expert_checkpoint,
goal images are selected from a deterministic PPO trajectory and the environment
is restored to a configurable trajectory start before planning. Without an
expert, synthetic goal rendering places the T on the target without advancing
physics. The default CPU physics backend allows SWM to own several independent
scenes; planning and rendering can still use a GPU.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

from .observations import (
    flatten_policy_state,
    named_observation_leaves,
    policy_observation,
)
from .random_goal_pusht import quaternion_yaw


def single(value):
    """Copy the first item of a ManiSkill batch to a NumPy value on CPU."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)[0].copy()


class PushTSWMEnv(gym.Env):
    """Expose one ManiSkill scene with SWM observations and evaluation goals."""

    metadata: ClassVar[dict] = {
        'render_modes': ['rgb_array'],
        'render_fps': 20,
    }

    def __init__(
        self,
        render_mode='rgb_array',
        image_size=224,
        camera_name='base_camera',
        control_mode='pd_joint_delta_pos',
        sim_backend='physx_cpu',
        expert_checkpoint=None,
        goal_step_distance=25,
        expert_max_steps=100,
        expert_max_attempts=20,
        expert_min_object_displacement=0.02,
        expert_min_eef_displacement=0.02,
        goal_position_tolerance=0.02,
        goal_yaw_tolerance=0.15,
        goal_eef_position_tolerance=None,
        expert_output_dir=None,
        state_goal=False,
        env_id='RandomGoalPushT-v1',
        expert_policy_type='state',
        expert_encoder='spatial_softmax',
        simulation_max_episode_steps=None,
        **kwargs,
    ):
        """Create the scene and optionally load a compatible PPO expert.

        Expert checkpoints provide trajectory goals; otherwise the native task
        target supplies a rendered goal. State goals require an expert so that
        each goal field comes from an observation of a reachable state.
        """
        super().__init__()
        if render_mode != 'rgb_array':
            raise ValueError("Only render_mode='rgb_array' is supported")
        if image_size < 1:
            raise ValueError('image_size must be positive')
        self.render_mode = render_mode
        self.env_id = env_id
        self.camera_name = camera_name
        self.image_size = image_size
        self._pixels = None
        self._goal = None
        self.state_goal = state_goal
        self.goal_observation = None
        self.expert_policy_type = expert_policy_type
        saved = None
        if expert_checkpoint is not None:
            from .experts import collection_defaults

            # Match the expert's training setup before building its simulator.
            inferred, saved = collection_defaults(expert_checkpoint)
            expected = {
                'env_id': env_id,
                'control_mode': control_mode,
                'policy_type': expert_policy_type,
            }
            if expert_policy_type == 'rgb':
                expected.update(
                    camera_name=camera_name,
                    image_size=image_size,
                    encoder=expert_encoder,
                )
            for key, value in expected.items():
                if key in inferred and inferred[key] != value:
                    raise ValueError(
                        f'Expert training config mismatch for {key}: '
                        f'{value!r} != {inferred[key]!r}'
                    )
        if state_goal and expert_checkpoint is None:
            raise ValueError('State goals require an expert_checkpoint')
        if goal_step_distance < 1 or expert_max_steps < goal_step_distance:
            raise ValueError(
                'Require 1 <= goal_step_distance <= expert_max_steps'
            )
        if not (
            0 < goal_position_tolerance < float('inf')
            and 0 < goal_yaw_tolerance <= np.pi
        ):
            raise ValueError(
                'Goal tolerances must be positive and finite; yaw <= pi'
            )
        if goal_eef_position_tolerance is not None and not (
            0 < goal_eef_position_tolerance < float('inf')
        ):
            raise ValueError(
                'EEF position tolerance must be positive and finite'
            )
        if (
            isinstance(expert_max_attempts, bool)
            or not isinstance(expert_max_attempts, (int, np.integer))
            or expert_max_attempts < 1
        ):
            raise ValueError('expert_max_attempts must be a positive integer')
        for name, threshold in (
            ('expert_min_object_displacement', expert_min_object_displacement),
            ('expert_min_eef_displacement', expert_min_eef_displacement),
        ):
            if not 0 <= threshold < float('inf'):
                raise ValueError(f'{name} must be nonnegative and finite')
        self.expert_max_attempts = expert_max_attempts
        self.expert_min_object_displacement = expert_min_object_displacement
        self.expert_min_eef_displacement = expert_min_eef_displacement
        self.goal_eef_position_tolerance = goal_eef_position_tolerance
        self._goal_eef_position = None
        self.goal_step_distance = goal_step_distance
        self.expert_max_steps = expert_max_steps
        self.goal_position_tolerance = goal_position_tolerance
        self.goal_yaw_tolerance = goal_yaw_tolerance
        self.expert_output_dir = expert_output_dir
        self._expert = None
        self._goal_pose = None
        self._expert_metadata = {}
        sensor_configs = {'width': image_size, 'height': image_size}
        if saved:
            kwargs.setdefault(
                'reward_mode', saved['environment']['kwargs']['reward_mode']
            )
            if expert_policy_type == 'rgb':
                import sapien

                # Restore camera settings used to train the RGB expert.
                sensor_configs = {}
                for name, camera in saved['environment']['cameras'].items():
                    settings = dict(camera)
                    pose = settings.pop('pose', None)
                    if pose is not None:
                        settings['pose'] = sapien.Pose(
                            p=pose['p'], q=pose['q']
                        )
                    sensor_configs[name] = settings
        if env_id == 'OverheadPushT-v1':
            from . import overhead_pusht  # noqa: F401
        if simulation_max_episode_steps is not None:
            kwargs['max_episode_steps'] = simulation_max_episode_steps
        # Reuse these settings for the separate expert-query simulator.
        self._planning_env_kwargs = {
            'num_envs': 1,
            'obs_mode': 'rgb+state_dict',
            'render_mode': render_mode,
            'control_mode': control_mode,
            'sim_backend': sim_backend,
            'reconfiguration_freq': 1,
            'sensor_configs': sensor_configs,
            **kwargs,
        }
        self._planning_oracle = None
        self._executed_actions = []
        self.env = gym.make(env_id, **self._planning_env_kwargs)
        self.action_space = self.env.unwrapped.single_action_space
        if env_id == 'OverheadPushT-v1' and self.action_space.shape != (2,):
            self.env.close()
            raise ValueError(
                'Overhead evaluation requires the 2D XY action space'
            )
        if saved:
            spec = saved['environment']
            if (
                list(self.action_space.shape) != spec['action_shape']
                or not np.array_equal(
                    self.action_space.low, spec['action_low']
                )
                or not np.array_equal(
                    self.action_space.high, spec['action_high']
                )
            ):
                self.env.close()
                raise ValueError(
                    'Expert action space differs from its training config'
                )
        # Observations are flat named fields; SWM lifts these into info.
        # BaseEnv has already initialized its spaces and scene at construction.
        obs = self.env.unwrapped.get_obs()
        self.observation_space = gym.spaces.Dict(
            {
                key: gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=tuple(value.shape[1:]),
                    dtype=np.float32,
                )
                for key, (_, value) in named_observation_leaves(obs).items()
            }
        )
        if expert_checkpoint is not None:
            device = self.env.unwrapped.device
            # Rebuild the network with its original input and action shapes.
            if expert_policy_type == 'rgb':
                from .experts import RGBExpert as Agent

                options = (
                    {
                        key: saved['agent'][key]
                        for key in (
                            'std_mode',
                            'initial_std',
                            'min_std',
                            'max_std',
                        )
                    }
                    if saved
                    else {}
                )
                self._expert = Agent(
                    self.env,
                    policy_observation(obs, device, 'rgb'),
                    encoder=expert_encoder,
                    **options,
                ).to(device)
            elif expert_policy_type == 'state':
                from .experts import StateExpert as Agent

                state = flatten_policy_state(obs, device)
                spaces = SimpleNamespace(
                    single_observation_space=gym.spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=tuple(state.shape[1:]),
                        dtype=np.float32,
                    ),
                    single_action_space=self.action_space,
                )
                self._expert = Agent(spaces).to(device)
            else:
                self.env.close()
                raise ValueError(
                    f'Unknown expert policy type: {expert_policy_type}'
                )
            self._expert.load_state_dict(
                torch.load(
                    expert_checkpoint, map_location=device, weights_only=True
                )
            )
            self._expert.eval()

    def _expert_goal(self, obs, seed, options, *, start_from_beginning=True):
        """Select an expert trajectory goal and restore the planning start.

        Retry failed expert tasks or pairs with both object XY and EEF XYZ
        displacement below their minimums (meters). Verify goal replay, then
        reset and replay only the prefix preceding the selected start. Frames
        and poses include the initial state, so action i leads to frame i + 1.
        """
        base = self.env.unwrapped
        requested_seed = seed
        # Keep retry randomness separate from start-index sampling. Consecutive
        # retry seeds would overlap the pool's seed + environment-index resets.
        retry_rng = np.random.default_rng(
            np.random.SeedSequence(seed, spawn_key=(1,))
        )
        attempted_seeds = {int(seed)}
        for attempt in range(1, self.expert_max_attempts + 1):
            initial_state = base.get_state().clone()
            frames = [self._camera(obs)]
            poses = [single(base.tee.pose.raw_pose)]
            eef_positions = [single(base.agent.tcp.pose.p)]
            actions = []
            observations = [self._observation(obs)] if self.state_goal else []
            success = False
            for _ in range(self.expert_max_steps):
                with torch.inference_mode():
                    action = self._expert.get_action(
                        policy_observation(
                            obs, base.device, self.expert_policy_type
                        ),
                        deterministic=True,
                    )
                action = torch.as_tensor(
                    np.clip(
                        single(action),
                        self.action_space.low,
                        self.action_space.high,
                    ),
                    device=base.device,
                )[None]
                if not torch.isfinite(action).all():
                    raise ValueError('Expert produced nonfinite action')
                obs, _, terminated, truncated, info = self.env.step(action)
                actions.append(single(action))
                frames.append(self._camera(obs))
                poses.append(single(base.tee.pose.raw_pose))
                eef_positions.append(single(base.agent.tcp.pose.p))
                if self.state_goal:
                    observations.append(self._observation(obs))
                success = success or bool(single(info['success']))
                if bool(single(terminated)) or bool(single(truncated)):
                    break
            start_index = 0
            if not start_from_beginning:
                # Leave room for the goal offset if the rollout is long enough.
                start_index = int(
                    self.np_random.integers(
                        max(0, len(actions) - self.goal_step_distance) + 1
                    )
                )
            index = start_index + min(self.goal_step_distance, len(actions))
            object_displacement = float(
                np.linalg.norm(poses[index][:2] - poses[start_index][:2])
            )
            eef_displacement = float(
                np.linalg.norm(
                    eef_positions[index] - eef_positions[start_index]
                )
            )
            too_static = (
                object_displacement < self.expert_min_object_displacement
                and eef_displacement < self.expert_min_eef_displacement
            )
            if success and not too_static:
                break
            reason = (
                'expert did not solve the native task'
                if not success
                else f'start-goal movement too small (object={object_displacement:.4f} '
                f'm, EEF={eef_displacement:.4f} m)'
            )
            print(
                f'Rejecting expert seed={seed}, attempt={attempt}: {reason}',
                flush=True,
            )
            if attempt < self.expert_max_attempts:
                # A deterministic expert needs a different scene to retry.
                seed = int(retry_rng.integers(0, 2**31 - 1))
                while seed in attempted_seeds:
                    seed = int(retry_rng.integers(0, 2**31 - 1))
                attempted_seeds.add(seed)
                obs, _ = self.env.reset(seed=seed, options=options)
        else:
            raise RuntimeError(
                f'No valid expert rollout after {self.expert_max_attempts} '
                f'attempts from seed {requested_seed}; last seed={seed}: {reason}'
            )
        # Oracle replay must reconstruct the accepted scene, not the first try.
        self._evaluation_seed = seed
        if self.state_goal:
            self.goal_observation = observations[index]
        self._goal = frames[index].copy()
        self._goal_pose = torch.as_tensor(poses[index], device=base.device)[
            None
        ]
        if self.goal_eef_position_tolerance is not None:
            self._goal_eef_position = torch.as_tensor(
                eef_positions[index], device=base.device
            )[None]
        self._expert_metadata = {
            'goal_step_distance': index - start_index,
            'expert_start_step': start_index,
            'expert_length': len(actions) - start_index,
            'expert_task_success': success,
            'expert_rollout_attempts': attempt,
            'expert_seed': seed,
            'expert_requested_seed': requested_seed,
            'expert_start_goal_object_distance_m': object_displacement,
            'expert_start_goal_eef_distance_m': eef_displacement,
        }
        # Reset also restores controller targets and time-limit counters, which
        # BaseEnv.set_state alone does not fully restore in ManiSkill.
        obs, info = self.env.reset(seed=seed, options=options)
        torch.testing.assert_close(
            base.get_state(),
            initial_state,
            rtol=0,
            atol=1e-6,
            msg='Expert and LEWM resets must reproduce the same start',
        )
        self.expert_actions = np.stack(actions[start_index:])
        self.expert_frames = np.stack(frames[start_index:])
        self.expert_goal_index = index - start_index
        # Replay the actual commands, including the final post-action frame.
        for action in actions[:index]:
            obs, _, _, _, _ = self.env.step(
                torch.as_tensor(action, device=base.device)[None]
            )
        replay_image = self._camera(obs)
        errors = self._goal_errors()
        self.replay_report = {
            **errors,
            'success': self._goal_reached(errors),
            'pixel_mae': float(
                np.abs(replay_image.astype(float) - self._goal).mean()
            ),
            'pixel_max_error': int(
                np.abs(replay_image.astype(int) - self._goal.astype(int)).max()
            ),
        }
        self.replay_report['passed'] = (
            self.replay_report['success']
            and self.replay_report['pixel_max_error'] <= 1
        )
        obs, info = self.env.reset(seed=seed, options=options)
        torch.testing.assert_close(
            base.get_state(), initial_state, rtol=0, atol=1e-6
        )
        # Replay the prefix to preserve both physics and controller targets.
        self._executed_actions = [
            action.copy() for action in actions[:start_index]
        ]
        for action in self._executed_actions:
            obs, _, _, _, info = base.step(
                torch.as_tensor(action, device=base.device)[None]
            )
        # Prefix replay is setup; planning gets a fresh time-limit budget.
        if hasattr(base, '_elapsed_steps'):
            base._elapsed_steps.zero_()
        if self.expert_output_dir:
            output = Path(self.expert_output_dir)
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output / f'expert_seed_{seed}.npz',
                pixels=np.stack(frames),
                action=np.stack(actions),
                obj_pose=np.stack(poses),
                eef_position=np.stack(eef_positions),
                rollout_attempts=attempt,
                requested_seed=requested_seed,
                goal_index=index,
                start_index=start_index,
                requested_goal_step_distance=self.goal_step_distance,
                task_success=success,
                seed=seed,
            )
            (output / f'replay_seed_{seed}.json').write_text(
                json.dumps(self.replay_report, indent=2)
            )
        if not self.replay_report['passed']:
            raise RuntimeError(
                f'Expert replay failed for seed {seed}: {self.replay_report}'
            )
        print(
            f'Expert seed={seed}: length={len(actions)}, goal_step={index}, '
            f'task_success={success}',
            flush=True,
        )
        return obs, info

    def _goal_errors(self, base=None):
        """Measure object XY/yaw and optional end-effector XYZ goal errors."""
        if base is None:
            base = self.env.unwrapped
        pose = base.tee.pose
        xy_error = torch.linalg.vector_norm(
            pose.p[:, :2] - self._goal_pose[:, :2], dim=-1
        )
        yaw_error = quaternion_yaw(pose.q) - quaternion_yaw(
            self._goal_pose[:, 3:]
        )
        # Wrap the angle difference to [-pi, pi] before taking its magnitude.
        yaw_error = torch.atan2(yaw_error.sin(), yaw_error.cos()).abs()
        errors = {
            'xy_error_m': float(xy_error.item()),
            'yaw_error_rad': float(yaw_error.item()),
        }
        if self.goal_eef_position_tolerance is not None:
            eef_error = torch.linalg.vector_norm(
                base.agent.tcp.pose.p - self._goal_eef_position,
                dim=-1,
            )
            errors['eef_position_error_m'] = float(eef_error.item())
        return errors

    def _goal_reached(self, errors=None):
        """Check goal tolerances, reusing measured errors when supplied."""
        if errors is None:
            errors = self._goal_errors()
        return (
            errors['xy_error_m'] <= self.goal_position_tolerance
            and errors['yaw_error_rad'] <= self.goal_yaw_tolerance
            and (
                self.goal_eef_position_tolerance is None
                or errors['eef_position_error_m']
                <= self.goal_eef_position_tolerance
            )
        )

    def _camera(self, obs):
        """Extract an unbatched RGB image and validate its shape and dtype."""
        sensors = obs['sensor_data']
        if self.camera_name not in sensors:
            raise ValueError(
                f'Unknown camera {self.camera_name!r}; available: {list(sensors)}'
            )
        pixels = single(sensors[self.camera_name]['rgb'])
        if (
            pixels.shape != (self.image_size, self.image_size, 3)
            or pixels.dtype != np.uint8
        ):
            raise ValueError(
                f'Unexpected camera RGB: {pixels.shape}, {pixels.dtype}'
            )
        return pixels

    def _sync_pose(self):
        """Synchronize changed actor poses with GPU physics when enabled."""
        base = self.env.unwrapped
        if base.gpu_sim_enabled:
            base.scene._gpu_apply_all()
            base.scene._gpu_fetch_all()

    def _render_goal(self):
        """Render the object at the native target, then restore its pose.

        No physics step is taken; the finally block restores the live scene
        even when observation capture fails.
        """
        base = self.env.unwrapped
        original = base.tee.pose.raw_pose.clone()
        target = base.goal_tee.pose.raw_pose.clone()
        # Target marker is on the table; the physical block center is 2 cm higher.
        target[:, 2] = 0.021
        try:
            base.tee.set_pose(Pose.create(target))
            self._sync_pose()
            return self._camera(base.get_obs())
        finally:
            base.tee.set_pose(Pose.create(original))
            self._sync_pose()

    def _observation(self, obs):
        """Flatten state fields to NumPy arrays and cache the current RGB view."""
        self._pixels = self._camera(obs)
        return {
            key: single(value).astype(np.float32)
            for key, (_, value) in named_observation_leaves(obs).items()
        }

    def reset(self, *, seed=None, options=None):
        """Reset the scene and select an image goal plus optional state fields.

        The adapter consumes start_from_beginning; remaining options are
        forwarded to ManiSkill and saved with the seed for expert replay.
        """
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        self._evaluation_seed = seed
        options = dict(options or {})
        start_from_beginning = options.pop('start_from_beginning', True)
        self._evaluation_options = copy.deepcopy(options)
        self._executed_actions = []
        obs, info = self.env.reset(seed=seed, options=options)
        if self._expert is not None:
            obs, info = self._expert_goal(
                obs,
                seed,
                options,
                start_from_beginning=start_from_beginning,
            )
        observation = self._observation(obs)
        if self._expert is None:
            self._goal = self._render_goal()
        self._pose_trace = []
        if self._expert is not None:
            self._record_pose(False)
        return observation, {
            **self._state_goal_info(),
            'goal': self._goal.copy(),
            'success': self._goal_reached()
            if self._expert is not None
            else bool(single(info['success'])),
            'task_success': bool(single(info['success'])),
            **self._expert_metadata,
        }

    def step(self, action):
        """Apply a clipped action and translate completion into SWM semantics.

        Goal success terminates the episode. Native task termination without
        goal success becomes truncation; action history supports expert replay.
        """
        action = np.asarray(action, dtype=np.float32)
        if (
            action.shape != self.action_space.shape
            or not np.isfinite(action).all()
        ):
            raise ValueError(
                f'Expected finite action of shape {self.action_space.shape}'
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)
        batched = torch.as_tensor(
            action, device=self.env.unwrapped.device
        ).unsqueeze(0)
        obs, reward, terminated, truncated, info = self.env.step(batched)
        self._executed_actions.append(action.copy())
        task_success = bool(single(info['success']))
        success = (
            self._goal_reached() if self._expert is not None else task_success
        )
        # SWM counts termination as success. A task failure ends via truncation.
        done = bool(single(terminated))
        if self._expert is not None:
            self._record_pose(task_success)
        return (
            self._observation(obs),
            float(single(reward)),
            success,
            bool(single(truncated)) or (done and not success),
            {
                **self._state_goal_info(),
                'goal': self._goal.copy(),
                'success': success,
                'task_success': task_success,
                **self._expert_metadata,
            },
        )

    def _planning_replay(self):
        """Reconstruct the current live state in the isolated oracle scene."""
        if self._planning_oracle is None:
            self._planning_oracle = gym.make(
                self.env_id, **self._planning_env_kwargs
            )
        oracle = self._planning_oracle
        base = oracle.unwrapped
        obs, _ = oracle.reset(
            seed=self._evaluation_seed,
            options=copy.deepcopy(self._evaluation_options),
        )
        for action in self._executed_actions:
            obs, _, _, _, _ = base.step(
                torch.as_tensor(action, device=base.device)[None]
            )
        # Fail rather than score a witness from an unrelated state.
        torch.testing.assert_close(
            base.get_state(),
            self.env.unwrapped.get_state(),
            rtol=0,
            atol=1e-6,
            msg='Expert replay must match the current MPC state',
        )
        pixel_error = int(
            np.abs(
                self._camera(obs).astype(int)
                - self._camera(self.env.unwrapped.get_obs()).astype(int)
            ).max()
        )
        if pixel_error > 1:
            raise RuntimeError(
                'Expert replay does not match current MPC image'
            )
        return oracle, obs, pixel_error

    @torch.inference_mode()
    def planning_action_comparison(self, sequences, output):
        """Render full action sequences from the same start without moving live state."""
        from stable_worldmodel.plot import save_panel_videos

        output = Path(output)
        panels = {}
        results = {}
        saved_actions = {}
        for label, sequence in sequences.items():
            actions = np.asarray(sequence, dtype=np.float32)
            if (
                actions.ndim != 2
                or actions.shape[1:] != self.action_space.shape
                or not len(actions)
                or not np.isfinite(actions).all()
            ):
                raise ValueError(
                    'Comparison requires finite controller actions'
                )
            actions = np.clip(
                actions, self.action_space.low, self.action_space.high
            )
            oracle, obs, _ = self._planning_replay()
            base = oracle.unwrapped
            frames = [self._camera(obs)]
            for action in actions:
                obs, _, _, _, _ = base.step(
                    torch.as_tensor(action, device=base.device)[None]
                )
                frames.append(self._camera(obs))
            errors = self._goal_errors(base)
            results[label] = {
                'final_goal_success': self._goal_reached(errors),
                'final_goal_errors': errors,
            }
            saved_actions[label] = actions
            panels[label] = [np.stack(frames)]
        panels['goal'] = [self._goal.copy()]
        save_panel_videos(output, panels, fps=self.metadata['render_fps'])
        np.savez_compressed(output / 'actions.npz', **saved_actions)
        return results

    @torch.inference_mode()
    def planning_expert_actions(self, horizon):
        """Query PPO from the current state in an isolated replay simulator.

        Reset and replay reconstruct controller targets as well as physics.
        The evaluated image goal remains fixed; PPO retains its trained
        native-task goal. Horizon actions are generated even if the oracle
        reports native task completion, matching fixed-horizon planning.
        """
        if self._expert is None:
            raise ValueError(
                'Current-state expert queries require a checkpoint'
            )
        if horizon < 1:
            raise ValueError('Expert horizon must be positive')
        oracle, obs, pixel_error = self._planning_replay()
        base = oracle.unwrapped
        actions = []
        for _ in range(horizon):
            action = single(
                self._expert.get_action(
                    policy_observation(
                        obs, base.device, self.expert_policy_type
                    ),
                    deterministic=True,
                )
            )
            action = np.clip(
                action, self.action_space.low, self.action_space.high
            ).astype(np.float32)
            if not np.isfinite(action).all():
                raise ValueError('Expert produced nonfinite action')
            actions.append(action.copy())
            obs, _, _, _, _ = base.step(
                torch.as_tensor(action, device=base.device)[None]
            )
        return np.stack(actions), {
            'env_step': len(self._executed_actions),
            'seed': int(self._evaluation_seed),
            'expert_target': 'native_task_goal',
            'current_state_replay_pixel_max_error': pixel_error,
        }

    def _state_goal_info(self):
        """Copy selected state-goal fields into info under goal_ names."""
        if not self.state_goal:
            return {}
        return {
            f'goal_{key}': value.copy()
            for key, value in self.goal_observation.items()
        }

    def _record_pose(self, task_success):
        """Append goal errors to the episode trace and optionally save it."""
        errors = self._goal_errors()
        self._pose_trace.append(
            {
                'step': len(self._pose_trace),
                **errors,
                'goal_success': self._goal_reached(errors),
                'task_success': task_success,
            }
        )
        if self.expert_output_dir:
            path = (
                Path(self.expert_output_dir)
                / f'pose_errors_seed_{self._evaluation_seed}.json'
            )
            path.write_text(json.dumps(self._pose_trace, indent=2))

    def render(self):
        """Return a copy of the RGB frame cached by the last reset or step."""
        if self._pixels is None:
            raise RuntimeError('Call reset before render')
        return self._pixels.copy()

    def render_video(self):
        """Render diagnostic panels using the actual evaluation goal."""
        if self._pixels is None:
            raise RuntimeError('Call reset before render_video')
        renderer = getattr(self.env.unwrapped, 'render_video', None)
        if renderer is not None:
            return single(renderer(goal=self._goal))
        return np.concatenate([self.render(), self._goal], axis=1)

    def close(self):
        """Release the live scene and the expert-query simulator, if created."""
        if self._planning_oracle is not None:
            self._planning_oracle.close()
        self.env.close()
