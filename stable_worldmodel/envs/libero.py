import os
import sys
import logging
import math
import numpy as np

import gymnasium as gym
from gymnasium import spaces
import h5py
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import stable_worldmodel as swm

BFDP = get_libero_path("bddl_files")
BENCHMARKDICT = {}
benchmark_dict = benchmark.get_benchmark_dict()
BENCHMARKDICT['libero_spatial'] = benchmark_dict['libero_spatial']()
# BENCHMARKDICT['libero_object'] = benchmark_dict['libero_object']()
# BENCHMARKDICT['libero_goal'] = benchmark_dict['libero_goal']()
# BENCHMARKDICT['libero_10'] = benchmark_dict['libero_10']()  

def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def gripper_2d_to_1d(gripper_qpos):
    """
    Convert 2D gripper position to 1D representation.
    Args:
        gripper_qpos: tensor of shape (2,) for gripper position
    Returns:
        tensor of shape (1,) for gripper state
    """
    return gripper_qpos[0:1] - gripper_qpos[1:2]

def get_proprio(obs):
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            gripper_2d_to_1d(obs["robot0_gripper_qpos"]),
        )
    ).astype(np.float32)

def load_demo_states(task_name="spatial", file_path="/network/projects/real-g-grp/libero"):
    benchmark_instance = BENCHMARKDICT[f"libero_{task_name}"]
    tasks_dict = {}
    for task_id in range(10):
        demo_files = os.path.join(file_path, benchmark_instance.get_task_demonstration(task_id))
        state_list = []
        for demo_id in range(50):
            with h5py.File(demo_files, "r") as f:
                states = f[f"data/demo_{demo_id}/states"][()]
            state_list.append(states)
        tasks_dict[task_id] = state_list
    return tasks_dict

DEMODICT = {
    "spatial": load_demo_states("spatial"),
}

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class Libero(gym.Env):
    """
    Libero environment wrapper compatible with stable_worldmodel.
    """

    metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': 10,
    }
    def __init__(
        self,
        env=None,
        cfg={},
        env_name="spatial",
        task_id=0,
        camera_name="agentview",
        render_mode="rgb_array",
        *args,
        **kwargs,
    ):
        super().__init__()
        if env is None:
            logger.info(f'Creating Libero task_name and task_id: {env_name}, {task_id}')
            benchmark_instance = BENCHMARKDICT[f"libero_{env_name}"]
            task = benchmark_instance.get_task(task_id)
            env_args = {
                "bddl_file_name": os.path.join(BFDP, task.problem_folder, task.bddl_file),
                "camera_heights": 224,
                "camera_widths": 224,
                "hard_reset": False,
                "camera_names": [camera_name],
            }
            env = OffScreenRenderEnv(**env_args)

        self.env = env
        self.task_id = task_id
        self.cfg = cfg
        self.obj_of_interest = self.env.obj_of_interest
        self.eef_name = "robot0_eef"

        self.task_name = env_name
        self.camera_name = camera_name
        self.custom_camera_name = self.camera_name
        self.camera_width = self.env.env.camera_widths[0]
        self.camera_height = self.env.env.camera_heights[0]
        self.full_action_dim = self.env.env.action_dim
        self.action_dim = self.full_action_dim

        self.imp_obj_threshold = (
            cfg.get('task_specification', {})
            .get('env', {})
            .get('imp_obj_threshold', 0.1)
        )

        logger.info(f'Set {self.imp_obj_threshold=}')

        self._goal_obj_pos_dict = {imp_obj: None for imp_obj in self.obj_of_interest}
        self._goal_eef_pos = None
        self._goal = None
        self._goal_sim_state = None
        self._init_sim_state = None

        # Gym spaces
        self.action_space = gym.spaces.Box(
            low=np.full(7, -1.0),
            high=np.full(7, 1.0),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                'proprio': spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(7,),
                    dtype=np.float32,
                ),
            }
        )

        # Variation space
        self.variation_space = swm.spaces.Dict(
            {
                # 'camera': swm.spaces.Dict(
                #     {
                #         'height': swm.spaces.Box(
                #             low=np.array([128], dtype=np.float32),
                #             high=np.array([512], dtype=np.float32),
                #             init_value=np.array([224], dtype=np.float32),
                #             shape=(1,),
                #             dtype=np.float32,
                #         ),
                #         'width': swm.spaces.Box(
                #             low=np.array([128], dtype=np.float32),
                #             high=np.array([512], dtype=np.float32),
                #             init_value=np.array([224], dtype=np.float32),
                #             shape=(1,),
                #             dtype=np.float32,
                #         ),
                #     }
                # ),
            }
        )


    def reset(self, seed=None, options=None, **kwargs):
        if seed is not None:
            self.seed(seed)

        info = self.env.reset()

        options = options or {}
        sample_goal = options.get('sample_goal', False)

        if sample_goal:
            goal_distance = options.get('goal_distance', 'max')
            init_sim_state, goal_sim_state = self.sample_init_goal_sim_state(goal_distance)
            goal_obs = self._restore_sim_state(goal_sim_state)
            self._goal = goal_obs[f"{self.camera_name}_image"][::-1, ::-1].copy()
            self._goal_eef_pos = goal_obs[f"{self.eef_name}_pos"].copy()
            self._goal_obj_pos_dict = {obj: goal_obs[f"{obj}_pos"].copy() for obj in self.obj_of_interest}
            self._goal_sampling_success = True
            self._init_sim_state = init_sim_state
            self._goal_sim_state = goal_sim_state
            info = self._restore_sim_state(init_sim_state) 
        else:
            self._goal = self.render()
            self._goal_eef_pos = None
            self._goal_obj_pos_dict = {imp_obj: None for imp_obj in self.obj_of_interest}
            self._goal_sampling_success = False
            self._init_sim_state = None
            self._goal_sim_state = None
        

        obs, info = self.get_obs_proprio_succ_from_info(info)

        info['goal'] = self._goal
        info['goal_eef_pos'] = self._goal_eef_pos
        for obj in self.obj_of_interest:
            info[f"goal_{obj}_pos"] = self._goal_obj_pos_dict[obj]

        info['goal_sampling_success'] = self._goal_sampling_success
        info['goal_state'] = self._goal_sim_state
        info['state'] = self._save_sim_state()
        info.pop('reward', None)
        info.pop('success', None)

        return obs, info

    def step(self, action):
        info, _, _, _ = self.env.step(action)

        obs, info = self.get_obs_proprio_succ_from_info(info)

        info['goal'] = self._goal
        info['goal_eef_pos'] = self._goal_eef_pos
        info['goal_sampling_success'] = getattr(
            self, '_goal_sampling_success', False
        )
        for obj in self.obj_of_interest:
            info[f"goal_{obj}_pos"] = self._goal_obj_pos_dict[obj]
        info['goal_state'] = self._goal_sim_state
        info["state"] = self._save_sim_state()
        
        reward, success = info.pop('reward', 0), info.pop('success', False)
        if success:
            logger.info('RoboCasaWrapper: Task success detected in step()')
        return obs, reward, success, False, info
    
    def render(self):
        result = self.env.sim.render(
            height=self.camera_height,
            width=self.camera_width,
            camera_name=self.camera_name,
        )[::-1, ::-1].copy()
        return result
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)


    def update_env(self, env_info):
        pass
    
    def sample_init_goal_sim_state(self, goal_distance="max"):
        """
        Sample state from the demo_dataset.
        """
        demo_id = self.np_random.randint(50)
        states = DEMODICT[self.task_name][self.task_id][demo_id]
        if goal_distance == "max":
            init_state = states[0]
            goal_state = states[-1]
        else:
            start_id = self.np_random.randint(states.shape[0])
            init_state = states[start_id]
            goal_state = states[min(start_id + self.frame_distance, states.shape[0] - 1)]
        init_state[1+self.env.sim.model.nq:] = 0
        init_state[0] = 0
        goal_state[1+self.env.sim.model.nq:] = 0
        goal_state[0] = 0
        return  init_state, goal_state
   
    def prepare(self, seed, init_state, env_info=None):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.seed(seed)
        self.reset()
        info = self._restore_sim_state(init_state)
        obs, info = self.get_obs_proprio_succ_from_info(info)
        return obs, self._save_sim_state()


    def _save_sim_state(self):
        return self.env.get_sim_state()

    def _restore_sim_state(self, state):
        return self.env.set_init_state(state)

    def _set_state(self, sim_state):
        self._init_sim_state = sim_state.copy()
        self._restore_sim_state(sim_state)

    def _set_goal_state(self, goal_sim_state):
        sim_state = self._save_sim_state()
        goal_obs = self._restore_sim_state(goal_sim_state)
        self._goal = goal_obs[f"{self.camera_name}_image"][::-1, ::-1].copy()
        self._goal_eef_pos = goal_obs[f"{self.eef_name}_pos"].copy()
        self._goal_obj_pos_dict = {obj: goal_obs[f"{obj}_pos"].copy() for obj in self.obj_of_interest}
        self._goal_sampling_success = True
        self._goal_sim_state = goal_sim_state.copy()
        self._restore_sim_state(sim_state)


    def _set_goal(self, goal, goal_eef_pos, goal_sim_state, **goal_obj_pos):
        # print(goal_obj_pos)
        self._goal = goal.copy()
        # print(self._goal.shape)
        self._goal_eef_pos = goal_eef_pos.copy()
        self._goal_obj_pos_dict = {obj.split('_pos')[0]: goal_obj_pos[obj].copy() for obj in goal_obj_pos.keys()}
        self._goal_sim_state = goal_sim_state.copy()
        # print(self._goal_obj_pos_dict)
        # print("HERE")
    def check_success(self, info):
        eef_pos = info[f"{self.eef_name}_pos"]
        eef_dist = np.linalg.norm(eef_pos - self._goal_eef_pos) if self._goal_eef_pos is not None else np.inf
        success = eef_dist < self.imp_obj_threshold
        obj_dist_dict = {}
        for obj in self.obj_of_interest:
            obj_pos = info[f"{obj}_pos"]
            obj_dist_dict[obj] = np.linalg.norm(self._goal_obj_pos_dict[obj] - obj_pos) if self._goal_obj_pos_dict[obj] is not None else np.inf
            success = success and obj_dist_dict[obj] < self.imp_obj_threshold
        dist = eef_dist + sum(obj_dist_dict.values())
        reward = -dist
        return success, reward
    
    def get_obs_proprio_succ_from_info(self, info):
        proprio = get_proprio(info)
        obs = {"proprio": proprio}
        success, reward = self.check_success(info)
        info["success"] = success
        info["reward"] = reward
        return obs, info

    def _get_current_eef_pos(self):
        """Get the current end-effector position from the simulator."""
        # Get EEF position from the robot's site
        robot = self.env.robots[0]
        eef_site_id = robot.eef_site_id

        # eef_site_id might be an integer ID or a string name
        if isinstance(eef_site_id, str):
            eef_site_id = self.env.sim.model.site_name2id(eef_site_id)

        return self.env.sim.data.site_xpos[eef_site_id].copy()
    
    def close(self):
        self.env.close()