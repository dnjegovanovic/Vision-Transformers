import torch
import torch.nn as nn

from .PatchEmbeddings import PatchEmbeddings


class PositionEmbedding(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.patch_emb = PatchEmbeddings(config)
        # Create a learnable classification token
        # The classification token is added to the beginning of the input sequence
        # and is used to classify the entire sequence
        self.cls_token = nn.Parameter(torch.rand(1, 1, config["hidden_size"]))

        # Create position embeddings for the classification token and the patch embeddings
        # Add 1 to the sequence length for the classification token
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.patch_emb.num_patches + 1, config["hidden_size"])
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_channels, image_size, image_size)
        x = self.patch_emb(x)
        batch_size, _, _ = x.size()
        # Expand the classification token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the classification token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_emb
        x = self.dropout(x)

        return x
