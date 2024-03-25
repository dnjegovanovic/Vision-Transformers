import math

import torch
import torch.nn as nn
from einops import rearrange

from .SingleAttentionHead import AttentionHead


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    """

    def __int__(self, config):
        super().__int__()

        self.hidden_size = config["hidden_size"]
        self.num_attn_heads = config["num_attn_heads"]

        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attn_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]

        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attn_probs_dropout_prob"],
                self.qkv_bias,
            )
            self.heads.append(head)

        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x: torch.Tensor, output_attentions=False):
        # Calculate the attention output for each attention head
        attn_outs = [head(x) for head in self.heads]

        # Concatenate the attention outputs from each attention head
        attn_out = torch.cat([attn_out for attn_out, _ in attn_outs], dim=-1)

        # Project the concatenated attention output back to the hidden size
        attn_out = self.output_projection(attn_out)
        attn_out = self.output_dropout(attn_out)

        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attn_out, None)
        else:
            attn_probs = torch.stack([attn_probs for _, attn_probs in attn_outs], dim=1)
            return (attn_out, attn_probs)


class MultiHeadAttentionOptimized(nn.Module):

    """
    Slightly optimized model. All the heads are processed simultaneously with merged q,k,v proj.
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config["hidden_size"]
        self.num_attn_heads = config["num_attention_heads"]

        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attn_heads
        self.all_head_size = self.num_attn_heads * self.attention_head_size

        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]

        # Create a linear layer to project the query, key, and value
        # We created Linear layer witch project qkv in same time, so we increase all_head_size three times
        self.qkv_proj = nn.Linear(
            self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias
        )
        self.attn_dropout = nn.Dropout(config["attn_probs_dropout_prob"])

        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Project the query, key, and value
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        qkv = self.qkv_proj(x)

        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)

        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)

        Q = rearrange(
            query,
            "B S (H Z) -> B H S Z",
            H=self.num_attn_heads,
            Z=self.attention_head_size,
        )
        K = rearrange(
            key,
            "B S (H Z) -> B H S Z",
            H=self.num_attn_heads,
            Z=self.attention_head_size,
        )
        V = rearrange(
            value,
            "B S (H Z) -> B H S Z",
            H=self.num_attn_heads,
            Z=self.attention_head_size,
        )

        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(Q, K.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attn_probs = nn.functional.softmax(attention_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Calculate the attention output
        attn_out = torch.matmul(attn_probs, V)
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)

        attn_out = rearrange(attn_out, "B H S Z -> B S (H Z)")

        # Project the attention output back to the hidden size
        attn_out = self.output_projection(attn_out)
        attn_out = self.output_dropout(attn_out)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attn_out, None)
        else:
            return (attn_out, attn_probs)
