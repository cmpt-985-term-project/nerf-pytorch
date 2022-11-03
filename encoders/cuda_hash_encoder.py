#
# tiny-cuda-nn implementation of grid encoder backed by hashtables
#
import torch.nn as nn
import tinycudann as tcnn
import json
import numpy as np
import nvtx

class CUDAHashEncoder(nn.Module):
    def __init__(self, input_dim=3, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(CUDAHashEncoder, self).__init__()

        per_level_scale = np.exp(np.log(finest_resolution / base_resolution) / (n_levels-1))

        encoding_config = json.loads(f'''
        {{
            "otype": "HashGrid",
            "n_levels": {n_levels},
            "n_features_per_level": {n_features_per_level},
            "log2_hashmap_size": {log2_hashmap_size},
            "base_resolution": {base_resolution},
            "per_level_scale": {per_level_scale}
        }}
        ''')
        self.encoder = tcnn.Encoding(n_input_dims=input_dim, encoding_config=encoding_config)

    @nvtx.annotate("Hash Encoding")
    def forward(self, x):
        return self.encoder(x)
