import torch
import torch.nn as nn

from .MultiHeadAttention import MultiHeadAttentionOptimized, MultiHeadAttention
from .MLP import MLP


class Block(nn.Module):
    """
    Single Block
    """

    def __init__(self, config):
        super().__init__()
        self.use_fast_attn = config["use_optimized_attn"]
        if self.use_fast_attn:
            self.multi_head_attn = MultiHeadAttentionOptimized(config)
        else:
            self.multi_head_attn = MultiHeadAttention(config)

        self.layer_norm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layer_norm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x: torch.Tensor, output_attn=False):
        """
        :param x: input tensor
        :param output_attn: flag to output attention probability
        :return: (x, attn_prob)
        """
        x_norm = self.layer_norm_1(x)
        attn_output, attn_prob = self.multi_head_attn(x_norm, output_attn=output_attn)

        # Skip connection
        x = x + attn_output
        # Feed-forward network
        mlp_output = self.mlp(self.layer_norm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attn:
            return (x, None)
        else:
            return (x, attn_prob)

class Encoder(nn.Module):
    """
    Encoder Block
    """
    def __init__(self, config):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x: torch.Tensor, output_attn=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attn=output_attn)
            if output_attn:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attn:
            return (x, None)
        else:
            return (x, all_attentions)