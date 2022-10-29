#
# Fused Encoder
#
import torch
import torch.nn as nn

import tinycudann as tcnn
import json

class FusedEncoder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(FusedEncoder, self).__init__()

        encoding_config = json.loads(f'''
            "otype": "HashGrid",
            "n_levels": {n_levels},
            "n_features_per_level": {n_features_per_level},
            "log2_hashmap_size": {log2_hashmap_size},
            "base_resolution": {base_resolution},
            "per_level_scale": 1.5
        ''')
        self.encoder = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_config)

    def forward(self, x):
        return self.encoder(x)
