from vitapp.models.MultiHeadAttention import MultiHeadAttentionOptimized

import torch

from vitapp.config.core import config

def test_mha():
    config_data = config.model_conf.ViT

    mha = MultiHeadAttentionOptimized(config_data)

    input_x = torch.rand(
        [
            8,
            config_data["patch_size"],
            config_data["hidden_size"]
        ]
    )

    res = mha(input_x)

    assert res[0].shape == (8, config_data["patch_size"], config_data["hidden_size"]), "Shape does not match"