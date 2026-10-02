"""State architecture and integration with the native SWM train/eval scripts."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import gymnasium as gym
import hydra
import lance
import numpy as np
import pyarrow as pa
import stable_pretraining as spt
import torch
from torch import nn

import stable_worldmodel as swm
from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
from stable_worldmodel.wm.lewm import StateLeWM

ROOT = Path(__file__).resolve().parents[2]
SWM_ROOT = Path(
    os.environ.get('SWM_DIR', Path(swm.__file__).resolve().parent.parent)
)


def load_script(relative, name):
    spec = importlib.util.spec_from_file_location(name, SWM_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tiny_config():
    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(ROOT / 'scripts/train/config')
    ):
        cfg = hydra.compose(
            config_name='state_lewm_debug',
            overrides=['+trainer.enable_progress_bar=false'],
        )
    cfg.state_columns = ['obs_a', 'obs_b']
    cfg.state_dim = 3
    cfg.img_size = 8
    cfg.embed_dim = 8
    cfg.wm.history_size = 2
    cfg.data.dataset.frameskip = 1
    cfg.data.dataset.keys_to_load = ['action', 'obs_a', 'obs_b']
    cfg.model.encoder.hidden_dim = 16
    cfg.model.predictor.depth = 1
    cfg.model.predictor.heads = 2
    cfg.model.predictor.dim_head = 4
    cfg.model.predictor.mlp_dim = 16
    cfg.model.projector.hidden_dim = 16
    cfg.model.pred_proj.hidden_dim = 16
    cfg.loss.sigreg.kwargs.knots = 3
    cfg.loss.sigreg.kwargs.num_proj = 8
    return cfg


class ToyEnv(gym.Env):
    metadata: ClassVar[dict] = {
        'render_modes': ['rgb_array'],
        'render_fps': 20,
    }
    render_mode = 'rgb_array'
    observation_space = gym.spaces.Dict(
        {
            'obs_a': gym.spaces.Box(-np.inf, np.inf, (1,)),
            'obs_b': gym.spaces.Box(-np.inf, np.inf, (2,)),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, (1,))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.obs(), self.info()

    def obs(self):
        return {
            'obs_a': np.array([self.steps], dtype=np.float32),
            'obs_b': np.array([0.0, 1.0], dtype=np.float32),
        }

    def info(self):
        return {
            'goal': np.zeros((8, 8, 3), dtype=np.uint8),
            'goal_obs_a': np.array([2.0], dtype=np.float32),
            'goal_obs_b': np.array([0.0, 1.0], dtype=np.float32),
        }

    def step(self, action):
        self.steps += 1
        return self.obs(), 0.0, self.steps == 2, False, self.info()

    def render(self):
        return np.full((8, 8, 3), self.steps * 30, dtype=np.uint8)


class StateModelTests(unittest.TestCase):
    def test_state_columns_goal_mapping_and_action_gradients(self):
        torch.set_num_threads(1)
        cfg = tiny_config()
        cfg.model.action_encoder.input_dim = 1
        model = hydra.utils.instantiate(cfg.model).eval()
        # Native AdaLN starts with zero action gates; emulate learned gates
        # so this test checks gradient flow through the state adapter.
        for module in model.modules():
            if hasattr(module, 'adaLN_modulation'):
                nn.init.normal_(module.adaLN_modulation[-1].weight, std=0.02)
        state = torch.randn(1, 1, 1, 3)
        goal = torch.randn(1, 1, 1, 3)
        actions = torch.randn(1, 1, 3, 1, requires_grad=True)
        info = {
            'obs_a': state[..., :1],
            'obs_b': state[..., 1:],
            'goal_obs_a': goal[..., :1],
            'goal_obs_b': goal[..., 1:],
            # Dataset goals may carry action columns; they must not be encoded.
            'goal_action': torch.randn(1, 1, 1, 99),
        }
        cost = ShootingCostEvaluator(model, GoalMSE()).get_cost(info, actions)
        expected = model.encode(
            {'obs_a': goal[:, 0, :, :1], 'obs_b': goal[:, 0, :, 1:]}
        )['emb']
        torch.testing.assert_close(info['goal_emb'], expected)
        self.assertEqual(cost.shape, (1, 1))
        self.assertEqual(info['predicted_emb'].shape, (1, 1, 4, 8))
        cost.sum().backward()
        self.assertTrue(torch.isfinite(actions.grad).all())
        self.assertGreater(actions.grad.abs().sum(), 0)
        # Configured columns win over simulator restoration state and pixels.
        inputs = {'obs_a': state[:, 0, :, :1], 'obs_b': state[:, 0, :, 1:]}
        expected = model.encode(dict(inputs))['emb']
        actual = model.encode(
            {
                **inputs,
                'state': torch.zeros(1, 1, 99),
                'pixels': torch.randn(1, 1, 3, 8, 8),
            }
        )['emb']
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(model.observation(inputs), state[:, 0])

    def test_vector_state_goal_without_images(self):
        cfg = tiny_config()
        cfg.model.action_encoder.input_dim = 1
        cfg.state_columns = []
        model = hydra.utils.instantiate(cfg.model).eval()
        info = {
            'state': torch.randn(1, 2, 1, 3),
            'goal_state': torch.randn(1, 2, 1, 3),
        }
        expected = model.encode({'state': info['goal_state'][:, 0]})['emb']
        cost = ShootingCostEvaluator(model, GoalMSE()).get_cost(
            info, torch.randn(1, 2, 2, 1)
        )
        torch.testing.assert_close(info['goal_emb'], expected)
        self.assertEqual(cost.shape, (1, 2))
        self.assertNotIn('pixels', info)

    def test_history_and_missing_numeric_goal(self):
        cfg = tiny_config()
        cfg.model.action_encoder.input_dim = 1
        model = hydra.utils.instantiate(cfg.model).eval()
        info = {
            'obs_a': torch.randn(1, 2, 2, 1),
            'obs_b': torch.randn(1, 2, 2, 2),
            'goal_obs_a': torch.randn(1, 2, 1, 1),
            'goal_obs_b': torch.randn(1, 2, 1, 2),
            'action_history': torch.randn(1, 2, 1, 1),
        }
        actions = torch.randn(1, 2, 3, 1)
        evaluator = ShootingCostEvaluator(model, GoalMSE())
        incomplete = {k: v for k, v in info.items() if k != 'goal_obs_b'}
        with self.assertRaises(KeyError):
            evaluator.get_cost(incomplete, actions)
        cost = evaluator.get_cost(info, actions)
        self.assertEqual(cost.shape, (1, 2))
        self.assertEqual(info['predicted_emb'].shape, (1, 2, 5, 8))
        torch.testing.assert_close(
            info['action'][:, :, :1], info['action_history']
        )
        self.assertNotIn('pixels', info)

    def test_native_swm_training_export_and_evaluation(self):
        torch.set_num_threads(1)
        trainer = load_script('scripts/train/lewm.py', 'swm_test_train')
        evaluator = load_script('scripts/plan/eval_wm.py', 'swm_test_eval')
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {'STABLEWM_HOME': tmp}),
        ):
            old_cache = spt.get_config().cache_dir
            spt.get_config().cache_dir = tmp
            self.addCleanup(setattr, spt.get_config(), 'cache_dir', old_cache)
            root = Path(tmp)
            dataset = root / 'train.lance'
            lance.write_dataset(
                pa.table(
                    {
                        'episode_idx': [0] * 4 + [1] * 4,
                        'step_idx': list(range(4)) * 2,
                        'action': [[float(i % 3) / 2] for i in range(8)],
                        'obs_a': [[float(i)] for i in range(8)],
                        'obs_b': [[float(i % 2), 1.0] for i in range(8)],
                    }
                ),
                str(dataset),
            )
            cfg = tiny_config()
            cfg.data.dataset.name = str(dataset)
            cfg.subdir = 'native'
            cfg.output_model_name = 'native'
            cfg.train_split = 0.5
            cfg.loader.batch_size = 2
            cfg.loader.num_workers = 0
            cfg.trainer.accelerator = 'cpu'
            cfg.trainer.devices = 1
            cfg.trainer.precision = '32-true'
            cfg.trainer.max_epochs = 2
            cfg.trainer.enable_progress_bar = False
            cfg.wandb.enabled = False
            trainer.run.__wrapped__(cfg)
            checkpoint = root / 'checkpoints/native/weights_epoch_2.pt'
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((checkpoint.parent / 'config.yaml').is_file())
            model = swm.wm.utils.load_pretrained(str(checkpoint))
            self.assertIsInstance(model, StateLeWM)
            self.assertEqual(model.state_columns, ['obs_a', 'obs_b'])
            self.assertEqual(cfg.model.action_encoder.input_dim, 1)

            with hydra.initialize_config_dir(
                version_base=None, config_dir=str(ROOT / 'scripts/plan/config')
            ):
                cfg = hydra.compose(config_name='maniskill_state_pusht')
            cfg.policy = str(checkpoint)
            cfg.subdir = 'native_eval'
            cfg.dataset.keys_to_cache = ['action', 'obs_a', 'obs_b']
            cfg.eval.dataset_name = str(dataset)
            cfg.eval.num_eval = 2
            cfg.world.num_envs = 1
            cfg.world.expert_checkpoint = 'unused.pt'
            cfg.world.max_episode_steps = 3
            cfg.eval.eval_budget = 3
            cfg.eval.img_size = 8
            cfg.plan_config.action_block = 1
            cfg.plan_config.horizon = 2
            cfg.solver.device = 'cpu'
            cfg.solver.num_samples = 4
            cfg.solver.topk = 2
            cfg.solver.n_steps = 1
            # Only the simulator is substituted; the native training, scaler,
            # checkpoint, policy, CEM, metrics and video paths are exercised.
            with patch(
                'gymnasium.make', side_effect=lambda *a, **kw: ToyEnv()
            ):
                evaluator.run.__wrapped__(cfg)
            output = checkpoint.parent / 'evals/native_eval'
            self.assertIn(
                'success_rate', (output / cfg.output.filename).read_text()
            )
            self.assertTrue(list(output.glob('*.mp4')))


if __name__ == '__main__':
    unittest.main()
