import torch
import torch.nn as nn
from models.embeddings import Embeddings
from models.encoder import Encoder


class TransformerEncoder(nn.Module):
    """Embedding layer and Encoder Layer."""

    def __init__(self, num_of_heads, dim_of_model, dim_of_mlp, num_layers,
                 image_hw, channels=3, patch_size=16,
                 em_dropout=0.1, atten_dropout=0.1, mlp_dropout=0.1):
        super().__init__()

        self.embeddings = Embeddings(image_hw, dim_of_model, channels, patch_size, em_dropout)
        self.transformer_encoder = Encoder(num_of_heads, dim_of_model, dim_of_mlp, num_layers, atten_dropout,
                                           mlp_dropout)

    def forward(self, x):
        embedded = self.embeddings(x)
        encoded, attention_weights = self.transformer_encoder(embedded)
        return encoded, attention_weights


def test():
    trans = TransformerEncoder(12, 768, 3072, 2, (224, 224))
    q = torch.Tensor(32, 3, 224, 224)
    r, _ = trans(q)
    print('shape of transformer encoder:"{0}""'.format(r.shape))