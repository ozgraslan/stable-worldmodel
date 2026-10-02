"""PPO checkpoint architectures for expert goal generation (no trainer)."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .spatial_rgb_encoder import FlattenCNN, SpatialSoftmaxCNN


def layer_init(layer, std=2**0.5, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class StateExpert(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(), 256
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(), 256
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(
                nn.Linear(256, np.prod(envs.single_action_space.shape)),
                std=0.01 * np.sqrt(2),
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        sample_rgb = self.rgb_to_channels(sample_obs['rgb'])
        in_channels = sample_rgb.shape[1]

        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_rgb.float().cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors['rgb'] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if 'state' in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs['state'].flatten(start_dim=1).shape[-1]
            extractors['state'] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    @staticmethod
    def rgb_to_channels(rgb):
        if rgb.ndim == 5:  # FrameStack: (batch, time, height, width, channels)
            return rgb.permute(0, 1, 4, 2, 3).flatten(1, 2)
        return rgb.permute(0, 3, 1, 2)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == 'rgb':
                obs = self.rgb_to_channels(obs).float()
                obs = obs / 255
            elif key == 'state':
                obs = obs.flatten(start_dim=1)
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


class RGBExpert(nn.Module):
    def __init__(
        self,
        envs,
        sample_obs,
        encoder='spatial_softmax',
        std_mode='learned',
        initial_std=0.3,
        min_std=0.1,
        max_std=0.5,
    ):
        super().__init__()
        if std_mode not in ('learned', 'bounded', 'fixed'):
            raise ValueError(f'Unknown std mode: {std_mode}')
        if not all(
            math.isfinite(v) and v > 0 for v in (initial_std, min_std, max_std)
        ):
            raise ValueError('Standard deviations must be finite and positive')
        if min_std > max_std:
            raise ValueError('min_std must not exceed max_std')
        if std_mode == 'bounded' and not min_std <= initial_std <= max_std:
            raise ValueError(
                'initial_std must lie within the bounded std range'
            )
        self.std_mode = std_mode
        self.initial_std = initial_std
        self.logstd_min, self.logstd_max = math.log(min_std), math.log(max_std)
        encoders = {
            'nature': NatureCNN,
            'spatial_softmax': SpatialSoftmaxCNN,
            'flatten': FlattenCNN,
        }
        if encoder not in encoders:
            raise ValueError(f'Unknown encoder: {encoder}')
        self.feature_net = encoders[encoder](sample_obs=sample_obs)
        # latent_size = np.array(envs.unwrapped.single_observation_space.shape).prod()
        latent_size = self.feature_net.out_features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(
                nn.Linear(
                    512, np.prod(envs.unwrapped.single_action_space.shape)
                ),
                std=0.01 * np.sqrt(2),
            ),
        )
        initial_logstd = (
            -0.5 if std_mode == 'learned' else math.log(initial_std)
        )
        self.actor_logstd = nn.Parameter(
            torch.full(
                (1, int(np.prod(envs.unwrapped.single_action_space.shape))),
                initial_logstd,
            ),
            requires_grad=std_mode != 'fixed',
        )

    def get_logstd(self):
        if self.std_mode == 'fixed':
            # CLI noise settings take precedence over a checkpoint's old noise.
            return torch.full_like(
                self.actor_logstd, math.log(self.initial_std)
            )
        if self.std_mode == 'bounded':
            return self.actor_logstd.clamp(self.logstd_min, self.logstd_max)
        return self.actor_logstd

    @torch.no_grad()
    def constrain_std(self):
        # Project after updates/load so a clipped parameter can learn inward
        # again instead of getting stuck outside the clamp's gradient support.
        if self.std_mode == 'bounded':
            self.actor_logstd.clamp_(self.logstd_min, self.logstd_max)
        elif self.std_mode == 'fixed':
            self.actor_logstd.fill_(math.log(self.initial_std))

    def get_features(self, x):
        return self.feature_net(x)

    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.get_logstd().expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.get_logstd().expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


def collection_defaults(checkpoint):
    path = Path(checkpoint).parent / 'training_config.json'
    if not path.exists():
        return {}, None
    config = json.loads(path.read_text())
    if config.get('schema_version') != 1:
        raise ValueError(f'Unsupported PPO config version in {path}')
    agent, env = config['agent'], config['environment']
    defaults = {
        'policy_type': agent['policy_type'],
        'env_id': env['env_id'],
        'control_mode': env['control_mode'],
        'reward_mode': env['kwargs']['reward_mode'],
    }
    if agent['policy_type'] == 'rgb':
        if (
            agent['include_state']
            or agent['frame_stack'] != 1
            or agent['action_repeat'] != 1
        ):
            raise ValueError(
                'RGB collector requires include_state=false, frame_stack=1, action_repeat=1'
            )
        defaults.update(
            {
                key: agent[key]
                for key in (
                    'encoder',
                    'std_mode',
                    'initial_std',
                    'min_std',
                    'max_std',
                )
            }
        )
        cameras = env['cameras']
        if len(cameras) != 1:
            raise ValueError(
                'Config-based RGB collection currently requires one camera'
            )
        name, camera = next(iter(cameras.items()))
        if camera['width'] != camera['height']:
            raise ValueError(
                'Collector currently requires square camera images'
            )
        defaults.update(camera_name=name, image_size=camera['width'])
    return defaults, config
