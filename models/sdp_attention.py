import numpy as np
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_k, attn_dropout=0.1):
        super().__init__()
        # 缩放因子
        self.scalar = 1 / np.power(d_k, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # 计算q∙k
        attn = torch.bmm(q, k.transpose(1, 2))
        # 计算q∙k/sqr(d_k)
        attn = attn * self.scalar

        # attention masked
        if mask is not None:
            attn = attn.masked_fill(mask.bool(), -np.inf)

        # 计算softmax(q∙k/sqr(d_k))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # 计算softmax(q∙k/sqr(d_k))∙v
        sdp_output = torch.bmm(attn, v)

        return sdp_output, attn
