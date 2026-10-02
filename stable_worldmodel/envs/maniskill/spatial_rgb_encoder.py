"""RGB features represented as learned spatial feature-point coordinates."""

import torch
from torch import nn


class SpatialSoftmax(nn.Module):
    """One expected (x, y) coordinate per channel, normalized to [-1, 1]."""

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError(
                'SpatialSoftmax expects (batch, channels, height, width)'
            )
        batch, channels, height, width = features.shape
        # Flatten only to normalize each channel across spatial locations;
        # the policy receives coordinates, never the flattened feature map.
        attention = features.reshape(batch, channels, height * width).softmax(
            dim=-1
        )
        y, x = torch.meshgrid(
            torch.linspace(
                -1, 1, height, device=features.device, dtype=features.dtype
            ),
            torch.linspace(
                -1, 1, width, device=features.device, dtype=features.dtype
            ),
            indexing='ij',
        )
        expected_x = (attention * x.reshape(-1)).sum(dim=-1)
        expected_y = (attention * y.reshape(-1)).sum(dim=-1)
        return torch.stack((expected_x, expected_y), dim=-1).reshape(
            batch, 2 * channels
        )


class SpatialSoftmaxCNN(nn.Module):
    """3x3 convolutions with 4x total downsampling, followed by spatial softmax."""

    def __init__(self, sample_obs):
        super().__init__()
        channels = self.rgb_to_channels(sample_obs['rgb']).shape[1]
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.spatial_softmax = SpatialSoftmax()
        self.projection = nn.Sequential(nn.Linear(128, 256), nn.ReLU())
        self.out_features = 256
        self.state_encoder = None
        if 'state' in sample_obs:
            state_size = sample_obs['state'].flatten(start_dim=1).shape[-1]
            self.state_encoder = nn.Linear(state_size, 256)
            self.out_features += 256

    @staticmethod
    def rgb_to_channels(rgb):
        if rgb.ndim == 5:
            return rgb.permute(0, 1, 4, 2, 3).flatten(1, 2)
        return rgb.permute(0, 3, 1, 2)

    def forward(self, observations):
        rgb = self.rgb_to_channels(observations['rgb']).float() / 255.0
        encoded = self.projection(self.reduce_features(self.cnn(rgb)))
        if self.state_encoder is not None:
            state = self.state_encoder(
                observations['state'].flatten(start_dim=1)
            )
            encoded = torch.cat((encoded, state), dim=-1)
        return encoded

    def reduce_features(self, features):
        return self.spatial_softmax(features)


class FlattenCNN(SpatialSoftmaxCNN):
    """Same 3x3 convolutional trunk, with flattening instead of spatial softmax."""

    def __init__(self, sample_obs):
        super().__init__(sample_obs)
        del self.spatial_softmax
        height, width = self.rgb_to_channels(sample_obs['rgb']).shape[-2:]
        # Each padded stride-2 convolution rounds the spatial size upward.
        height, width = (height + 3) // 4, (width + 3) // 4
        self.projection = nn.Sequential(
            nn.Linear(64 * height * width, 256), nn.ReLU()
        )

    def reduce_features(self, features):
        return features.flatten(start_dim=1)
