import torch
import torch.nn as nn

import nvtx

# Positional encoding (section 5.1)
class PositionalEncoder(nn.Module):
    def __init__(self, in_channels, degrees, include_inputs=True):
        super(PositionalEncoder, self).__init__()
        embed_fns = []
        out_channels = 0

        # Optionally include the inputs in the encoding.
        # The supplementary material in the NSFF paper says that this is set to False for the
        # static NeRF, but the code seems to keep it as True.
        if include_inputs:
            embed_fns.append(lambda x: x)
            out_channels += in_channels

        # Encoding is powers of 2 times sin and cos of input
        freq_bands = 2.**torch.linspace(0., degrees-1, steps=degrees)
        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_channels += in_channels

        self.embed_fns = embed_fns
        self.out_channels = out_channels

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# A "Density" (not view-angle dependent) MLP
class DensityMLP(nn.Module):
    def __init__(self, in_channels, out_channels, degrees=10):
        super(DensityMLP, self).__init__()

        # Network parameters
        self.W = 256

        self.position_encoder = PositionalEncoder(in_channels=in_channels, degrees=degrees)

        # "We follow the DeepSDF [32] architecture and include a skip connection that concatenates this input to the fifth layer’s activation"
        self.model_part1 = nn.Sequential(
            nn.Linear(self.position_encoder.out_channels, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, self.W), nn.ReLU(inplace=True)
        )
        self.model_part2 = nn.Sequential(
            nn.Linear(self.W + self.position_encoder.out_channels, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, out_channels)
        )

    def forward(self, x):
        encoded_position = self.position_encoder(x)
        part1 = self.model_part1(encoded_position)
        part2 = self.model_part2(torch.cat([encoded_position, part1], dim=-1))
        return part2

# An "Color" (view-angle dependent) MLP
class ColorMLP(nn.Module):
    def __init__(self, degrees=4):
        super(ColorMLP, self).__init__()
        self.W = 256

        # For consistency with original paper, we will use the position encoder on the viewing angle,
        # even though a spherical harmonic encoder makes more sense.
        self.view_encoder = PositionalEncoder(in_channels=3, degrees=degrees)

        self.model = nn.Sequential(
            nn.Linear(self.view_encoder.out_channels + self.W, self.W), nn.ReLU(inplace=True),
            nn.Linear(self.W, 3)
        )

    def forward(self, x):
        input_view, feature_vector = x.split([3, self.W], dim=-1)
        encoded_view = self.view_encoder(input_view)
        return self.model(torch.cat([encoded_view, feature_vector], dim=-1))


# Re-written NeRF model. Re-written to be as similar as possible to the CutlassNeRF and FusedNeRF models
class NeRF(nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()
        self.W = 256

        # 1-dim density and 256-dim feature vector
        self.density_mlp = DensityMLP(in_channels=3, out_channels=self.W+1)
        self.color_mlp = ColorMLP()

    @nvtx.annotate("NeRF forward")
    def forward(self, x):
        # 3 input channels, 3 view channels
        input_position, input_view = x.split([3, 3], dim=-1)
        x = self.density_mlp(input_position)

        # 1-dim density, 256-dim feature vector
        density, feature_vector = x.split([1, self.W], dim=-1)

        rgb = self.color_mlp(torch.cat([input_view, feature_vector], dim=-1))

        return torch.cat([rgb, density], dim=-1)
