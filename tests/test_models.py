from vitapp.models.PatchEmbeddings import PatchEmbeddings

import torch


def test_patch_embeddings():
    config = {"image_size": 32, "patch_size": 4, "num_channels": 3, "hidden_size": 48}

    patch_emb = PatchEmbeddings(config)

    input_x = torch.rand(
        [8, config["num_channels"], config["image_size"], config["image_size"]]
    )

    print(f"Input tensor shape: {input_x.shape}")

    res = patch_emb(input_x)
    print(f"Output tensor: {res.shape}")
    assert res.shape == (8, patch_emb.num_patches, config["hidden_size"])
