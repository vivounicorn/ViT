import torch
import torch.nn as nn
from torch.nn import Dropout, Linear


class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(self, dim_of_model, dim_of_mlp, dropout=0.1):
        super().__init__()

        self.fc1 = Linear(dim_of_model, dim_of_mlp)
        self.fc2 = Linear(dim_of_mlp, dim_of_model)
        self.active_fn = torch.nn.functional.gelu
        self.dropout = Dropout(dropout)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-8)
        nn.init.normal_(self.fc2.bias, std=1e-8)

    def forward(self, x):
        x = self.fc1(x)
        x = self.active_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def test():
    from mh_attention import MultiHeadAttention
    mha = MultiHeadAttention(12, 768)

    # (batch_size, sequence_len_q, dim_of_model)
    q = torch.Tensor(32, 197, 768)

    output, attention_weights = mha(q, k=q, v=q, mask=None)

    print('shape of output:"{0}", shape of attention weight:"{1}"'.format(output.shape, attention_weights.shape))
    mp = Mlp(768, 3072)
    r = mp(output)
    print('shape of mlp:"{0}""'.format(r.shape))
