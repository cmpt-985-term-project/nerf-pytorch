#
# Fully Fused MLP NeRF
#  
import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn
import json
import nvtx

class FusedNeRF(nn.Module):
    def __init__(self,
                 density_layers=3,
                 density_dim=64,
                 density_features=15,
                 color_layers=4,
                 color_dim=64,
                 position_input_channels=3,
                 viewangle_input_channels=3,
                 ):
        super(FusedNeRF, self).__init__()

        self.position_input_channels = position_input_channels
        self.viewangle_input_channels = viewangle_input_channels

        # Density network
        density_network_config = json.loads(f'''
        {{
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": {density_dim},
            "n_hidden_layers": {density_layers},
            "feedback_alignment": false
        }}
        ''')

        # Density network predicts a single density value given a position in 3-space,
        # and also outputs 15 density features which get used as input to the color network
        n_input_dims = position_input_channels
        n_output_dims = density_features + 1
        self.density_network = tcnn.Network(n_input_dims=n_input_dims, n_output_dims=n_output_dims, network_config=density_network_config)

        # Color network
        color_network_config = json.loads(f'''
        {{
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": {color_dim},
            "n_hidden_layers": {color_layers},
            "feedback_alignment": false
        }}
        ''')

        # Color network concatenates density features with viewing angle features to produce RGB color value
        n_input_dims = density_features + viewangle_input_channels
        n_output_dims = 3 # RGB values
        self.color_network = tcnn.Network(n_input_dims=n_input_dims, n_output_dims=n_output_dims, network_config=color_network_config)
    
    @nvtx.annotate("Fused NeRF Forward")
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.position_input_channels, self.viewangle_input_channels], dim=-1)

        # Density
        with nvtx.annotate("density network"):
            output = self.density_network(input_pts)
            density, density_features = output[..., 0], output[..., 1:]

        # Color
        with nvtx.annotate("rgb network"):
            color = self.color_network(torch.cat([input_views, density_features], dim=-1))

        return torch.cat([color, density.unsqueeze(dim=-1)], -1)