import torch
import torch.nn as nn
from models.mh_attention import MultiHeadAttention
from models.mlp import Mlp
from torch.nn import LayerNorm


class Block(nn.Module):
    """Multi-head attention and MLP block."""

    def __init__(self, num_of_heads, dim_of_model, dim_of_mlp, atten_dropout=0.1, mlp_dropout=0.1):
        super().__init__()

        self.hidden_size = dim_of_model
        # Multi-head attention norma layer
        self.mh_attention_norm = LayerNorm(dim_of_model, eps=1e-8)
        # Mlp norma layer
        self.mlp_norm = LayerNorm(dim_of_model, eps=1e-8)
        # Mlp
        self.mlp = Mlp(dim_of_model, dim_of_mlp, mlp_dropout)
        # Multi-head attention
        self.mh_attention = MultiHeadAttention(num_of_heads, dim_of_model, atten_dropout)

    def forward(self, x):
        residual = x
        x = self.mh_attention_norm(x)

        x, weights = self.mh_attention(x, x, x)
        x = x + residual

        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + residual
        return x, weights


def test():
    b = Block(12, 768, 3072)
    q = torch.Tensor(32, 197, 768)
    output, attention_weights = b(q)
    print('shape of output:"{0}", shape of attention weight:"{1}"'.format(output.shape, attention_weights.shape))