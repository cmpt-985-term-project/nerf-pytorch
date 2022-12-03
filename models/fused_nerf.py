#
# NeRF based on NVIDIA's FusedMLP
#
# From Instant NGP, section 5.4:
# "Informed by the analysis in Figure 10, our results were generated with a 
#  1-hidden-layer density MLP and a 2-hidden-layer color MLP, both 64 neurons wide."
#
# For Neural Scene Flow Fields, we will have two NeRFs:
#   - A Dynamic NeRF, which uses 4 position channels (3-space, 1-time)
#   - A Static NeRF, which uses 3 position channels (3-space)
#
# Each NeRF will have a "density" network, which will hash-encode the position channels and produce the following:
#   - For the Dynamic NeRF: the forward and backward scene flow vectors, forward and backward
#     "disocclusion" weights, and a 16-dimensional feature vector with the first dimension as density
#   - For the Static NeRF: a blending weight to blend the static and dynamic RGB and density values,
#     and a 16-dimensional feature vector with the first dimension as density
#
# Both NeRFs will also concatenate the 16-dimensional feature vector with a viewing direction encoded onto a
# spherical harmonic basis of degree 4 (so, also 16-dimensional). This 32-dimensional input will be fed into
# a 4-layer "color" MLP which will produce an RGB color value for the dynamic and static portions of the scene.

import torch
import torch.nn as nn

import tinycudann as tcnn
import json
import nvtx

# A "Density" (not view-angle dependent) MLP
class FusedDensityMLP(nn.Module):
    def __init__(self, position_channels, out_channels):
        super(FusedDensityMLP, self).__init__()

        # Network parameters
        self.W = 128

        network_config1 = json.loads(f'''
            {{"otype":"FullyFusedMLP", "activation":"ReLU", "output_activation":"ReLU", "n_neurons":{self.W},
              "n_hidden_layers":2, "feedback_alignment":false}}''')
        self.model_part1 = tcnn.Network(n_input_dims=position_channels, n_output_dims=self.W, network_config=network_config1)

        network_config2 = json.loads(f'''
            {{"otype":"FullyFusedMLP", "activation":"ReLU", "output_activation":"None", "n_neurons":{self.W},
              "n_hidden_layers":3, "feedback_alignment":false}}''')
        self.model_part2 = tcnn.Network(n_input_dims=self.W + position_channels, n_output_dims=out_channels, network_config=network_config2)

    def forward(self, x):
        part1 = self.model_part1(x)
        part2 = self.model_part2(torch.cat([x, part1], dim=-1))
        return part2


# A "Color" (view-angle dependent) MLP
class FusedColorMLP(nn.Module):
    def __init__(self, view_channels):
        super(FusedColorMLP, self).__init__()
        self.W = 128

        network_config = json.loads(f'''
            {{"otype":"FullyFusedMLP", "activation":"ReLU", "output_activation":"None", "n_neurons":{self.W},
              "n_hidden_layers":1, "feedback_alignment":false}}''')
        self.model = tcnn.Network(n_input_dims=self.W + view_channels, n_output_dims=3, network_config=network_config)

    def forward(self, x):
        return self.model(x)


#  NeRF model
class FusedNeRF(nn.Module):
    def __init__(self, position_channels, view_channels):
        super(FusedNeRF, self).__init__()
        self.W = 128
        self.position_channels = position_channels
        self.view_channels = view_channels

        # 1-dim density and 128-dim feature vector
        self.density_mlp = FusedDensityMLP(position_channels=position_channels, out_channels=self.W + 1)
        self.color_mlp = FusedColorMLP(view_channels=view_channels)

    @nvtx.annotate("Fused Static NeRF forward")
    def forward(self, x):
        input_position, input_view = torch.split(x, [self.position_channels, self.view_channels], dim=-1)
        x = self.density_mlp(input_position)

        # 1-dim density, 256-dim feature vector
        density, feature_vector = x.split([1, self.W], dim=-1)

        rgb = self.color_mlp(torch.cat([input_view, feature_vector], dim=-1))

        return torch.cat([rgb, density], dim=-1)
