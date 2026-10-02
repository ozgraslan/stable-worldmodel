"""PushT with a fixed, mostly downward-facing policy camera."""

import numpy as np
import sapien
import torch
import torch.nn.functional as F
from mani_skill.envs.tasks.tabletop.push_t import PushTEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .planar_panda_stick import PlanarPandaStick


@register_env('OverheadPushT-v1', max_episode_steps=100)
class OverheadPushTEnv(PushTEnv):
    # Slate blue contrasts with the white robot and the red/green task objects.
    TABLE_COLOR = np.array([0.12, 0.20, 0.28, 1.0], dtype=np.float32)

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        options = kwargs.get('options') or {}
        indices = options.get('env_idx')
        if (
            indices is None
            or options.get('reconfigure', False)
            or getattr(self, '_video_goal_valid', None) is None
        ):
            self._video_goal = None
            self._video_goal_valid = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        else:
            self._video_goal_valid[indices] = False
        return result

    def _video_overhead(self):
        return self._get_obs_sensor_data()['overhead_camera']['rgb'].clone()

    def _sync_video_pose(self):
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene._gpu_fetch_all()

    def _task_goal_video(self):
        """Freeze a task-target illustration per episode without advancing physics."""
        missing = ~self._video_goal_valid
        if missing.any():
            original = self.tee.pose.raw_pose.clone()
            target = self.goal_tee.pose.raw_pose.clone()
            target[:, 2] = 0.021
            try:
                self.tee.set_pose(Pose.create(target))
                self._sync_video_pose()
                goal = self._video_overhead()
            finally:
                self.tee.set_pose(Pose.create(original))
                self._sync_video_pose()
            if self._video_goal is None:
                self._video_goal = goal
            else:
                self._video_goal[missing] = goal[missing]
            self._video_goal_valid[missing] = True
        return self._video_goal

    def render_video(self, goal=None):
        """Batched RGB row: current overhead | original third-person | goal overhead.

        An explicit goal is the expert-selected image used by LeWM. Otherwise,
        PPO recordings use a cached illustration of the environment task target.
        This method never adds cameras to the policy observation dictionary.
        """
        if goal is None:
            goal = self._task_goal_video()
        overhead = self._video_overhead()
        third_person = super().render_rgb_array()
        goal = torch.as_tensor(goal, device=overhead.device, dtype=torch.uint8)
        if goal.ndim == 3:
            goal = goal.unsqueeze(0)
        if goal.shape != overhead.shape:
            raise ValueError(
                f'Goal image shape {goal.shape} does not match overhead {overhead.shape}'
            )
        third_person = torch.as_tensor(third_person, device=overhead.device)
        if third_person.shape[1:3] != overhead.shape[1:3]:
            third_person = F.interpolate(
                third_person.permute(0, 3, 1, 2).float(),
                size=overhead.shape[1:3],
                mode='bilinear',
                align_corners=False,
                antialias=True,
            )
            third_person = (
                third_person.round()
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
            )
        return torch.cat([overhead, third_person, goal], dim=2)

    def render_rgb_array(self, camera_name=None):
        if camera_name is not None:
            return super().render_rgb_array(camera_name)
        return self.render_video()

    def render_all(self):
        return self.render_video()

    def _load_agent(self, options):
        # Use a local subclass without changing the globally registered Panda.
        # Restore the UID because PushT's scene builder uses it during reset.
        robot_uids = self.robot_uids
        if robot_uids == 'panda_stick':
            self.robot_uids = PlanarPandaStick
        try:
            super()._load_agent(options)
        finally:
            self.robot_uids = robot_uids

    def _load_scene(self, options):
        super()._load_scene(options)
        for entity in self.table_scene.table._objs:
            body = entity.find_component_by_type(
                sapien.render.RenderBodyComponent
            )
            for shape in body.render_shapes:
                for part in shape.parts:
                    part.material.set_base_color(self.TABLE_COLOR)

    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                uid='overhead_camera',
                # World-fixed view of the table's working area. A slight tilt
                # gives the stick some visible length instead of an end-on view.
                pose=sapien_utils.look_at(
                    eye=[-0.10, -0.12, 0.85],
                    target=[-0.10, 0.03, 0.0],
                    up=[0, 1, 0],
                ),
                width=128,
                height=128,
                fov=np.deg2rad(55),
                near=0.01,
                far=10,
            )
        ]
