#
# tiny-cuda-nn implementation of Spherical Harmonic encoder.
# Useful for encoding viewing direction vectors
#
import torch.nn as nn
import tinycudann as tcnn
import json
import nvtx

class CUDASHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
        super(CUDASHEncoder, self).__init__()

        encoding_config = json.loads(f'''
        {{
            "otype": "SphericalHarmonics",
            "degree": {degree}
        }}
        ''')
        self.encoder = tcnn.Encoding(n_input_dims=input_dim, encoding_config=encoding_config)

    @nvtx.annotate("SH Encoding")
    def forward(self, x):
        return self.encoder(x)
