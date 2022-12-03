import torch
import torch.nn as nn

import nvtx


# A "Density" (not view-angle dependent) MLP
class DensityMLP(nn.Module):
    def __init__(self, position_channels, out_channels):
        super(DensityMLP, self).__init__()

        # Network parameters
        self.W = 256

        # "We follow the DeepSDF [32] architecture and include a skip connection that concatenates this input to the fifth layer’s activation"
        self.model_part1 = nn.Sequential(
            nn.Linear(position_channels, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, self.W), nn.ReLU(inplace=True)
        )
        self.model_part2 = nn.Sequential(
            nn.Linear(self.W + position_channels, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, out_channels)
        )

    def forward(self, x):
        part1 = self.model_part1(x)
        part2 = self.model_part2(torch.cat([x, part1], dim=-1))
        return part2

# A "Color" (view-angle dependent) MLP
class ColorMLP(nn.Module):
    def __init__(self, view_channels):
        super(ColorMLP, self).__init__()
        self.W = 256

        self.model = nn.Sequential(
            nn.Linear(view_channels + self.W, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, 3)
        )

    def forward(self, x):
        return self.model(x)


# Re-written NeRF model. Re-written to be as similar as possible to the CutlassNeRF and FusedNeRF models
class NeRF(nn.Module):
    def __init__(self, position_channels, view_channels):
        super(NeRF, self).__init__()
        self.W = 256
        self.position_channels = position_channels
        self.view_channels = view_channels

        # 1-dim density and 256-dim feature vector
        self.density_mlp = DensityMLP(position_channels=position_channels, out_channels=self.W+1)
        self.color_mlp = ColorMLP(view_channels=view_channels)

    @nvtx.annotate("NeRF forward")
    def forward(self, x):
        input_position, input_view = torch.split(x, [self.position_channels, self.view_channels], dim=-1)
        x = self.density_mlp(input_position)

        # 1-dim density, 256-dim feature vector
        density, feature_vector = x.split([1, self.W], dim=-1)

        rgb = self.color_mlp(torch.cat([input_view, feature_vector], dim=-1))

        return torch.cat([rgb, density], dim=-1)
