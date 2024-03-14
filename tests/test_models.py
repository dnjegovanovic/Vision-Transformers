from vitapp.models.PatchEmbeddings import PatchEmbeddings
from vitapp.models.PositionEmbeddings import PositionEmbedding

import torch

from vitapp.config.core import config

print(config)


def test_patch_embeddings():
    config_data = config.model_conf.ViT

    patch_emb = PatchEmbeddings(config_data)

    input_x = torch.rand(
        [
            8,
            config_data["num_channels"],
            config_data["image_size"],
            config_data["image_size"],
        ]
    )

    print(f"Input tensor shape: {input_x.shape}")

    res = patch_emb(input_x)
    print(f"Output tensor: {res.shape}")
    assert res.shape == (8, patch_emb.num_patches, config_data["hidden_size"])


def test_position_emb():
    config_data = config.model_conf.ViT

    pos_emb = PositionEmbedding(config_data)

    input_x = torch.rand(
        [
            8,
            config_data["num_channels"],
            config_data["image_size"],
            config_data["image_size"],
        ]
    )

    print(f"Input tensor shape: {input_x.shape}")

    res = pos_emb(input_x)
    print(f"Output tensor: {res.shape}")
    assert res.shape == (
        8,
        pos_emb.patch_emb.num_patches + 1,
        config_data["hidden_size"],
    )
