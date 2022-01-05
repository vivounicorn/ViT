import torch
import torch.nn as nn
from torch.nn import LayerNorm
from models.block import Block
import copy


class Encoder(nn.Module):
    """Encoder with n blocks."""

    def __init__(self, num_of_heads, dim_of_model, dim_of_mlp, num_layers, atten_dropout=0.1, mlp_dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList()

        self.encoder_norm = LayerNorm(dim_of_model, eps=1e-8)

        for _ in range(num_layers):
            layer = Block(num_of_heads, dim_of_model, dim_of_mlp, atten_dropout, mlp_dropout)
            self.layers.append(copy.deepcopy(layer))

    def forward(self, x):

        # 此时输入还正确
        attn_weights_list = []
        for layer_block in self.layers:
            x, weights = layer_block(x)
            attn_weights_list.append(weights)

        encoded = self.encoder_norm(x)

        return encoded, attn_weights_list


def unit_test():
    encoder = Encoder(12, 768, 3072, 1)
    q = torch.Tensor(32, 197, 768)
    output, attention_weights = encoder(q)
    print('shape of output:"{0}", shape of encoder attention weight:"({1},{2})"'.format(output.shape,
                                                                                        len(attention_weights),
                                                                                        attention_weights[0].shape))
