import torch
import torch.nn as nn
from models.sdp_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, num_of_heads, dim_of_model, dropout=0.1):
        super().__init__()

        # 拆分出的attention head数量.
        self.num_of_heads = num_of_heads
        # 模型维度，例如：embedding层为词向量维度.
        self.dim_of_model = dim_of_model

        # 模型维度的设置要保证能使拆分出来的所有head维度相同
        if self.dim_of_model % self.num_of_heads != 0:
            raise RuntimeError('Dimensions of the model must be divisible by number of attention heads.')

        # 拆分出来的每个head向量的维度
        self.depth = self.dim_of_model // self.num_of_heads

        # 保持输入输出维度的仿射变换
        self.w_qs = nn.Linear(self.dim_of_model, self.dim_of_model)
        self.w_ks = nn.Linear(self.dim_of_model, self.dim_of_model)
        self.w_vs = nn.Linear(self.dim_of_model, self.dim_of_model)

        self.attention = ScaledDotProductAttention(self.depth,
                                                   attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(self.dim_of_model)

        # 最终输出层
        self.fc = nn.Linear(self.dim_of_model, self.dim_of_model)

    def forward(self, q, k, v, mask=None):

        # q.shape=(batch_size, sequence_len_q, dim_of_model)，其中dim_of_model = num_of_heads * depth
        batch_size, sequence_len_q, _ = q.size()
        batch_size, sequence_len_k, _ = k.size()
        batch_size, sequence_len_v, _ = v.size()

        # 类似ResNet，对query保留输入信息
        # residual = q

        # q.shape=(batch_size, num_of_heads, sequence_len_q, depth)
        q = self.w_qs(q).view(batch_size, -1, self.num_of_heads, self.depth).permute(0, 2, 1, 3)
        k = self.w_qs(k).view(batch_size, -1, self.num_of_heads, self.depth).permute(0, 2, 1, 3)
        v = self.w_qs(v).view(batch_size, -1, self.num_of_heads, self.depth).permute(0, 2, 1, 3)
        # print('q.shape=', q.shape)#([32, 12, 197, 64])

        # q.shape=(batch_size * num_of_heads, sequence_len_q, depth)
        q = q.reshape(batch_size * self.num_of_heads, -1, self.depth)
        k = k.reshape(batch_size * self.num_of_heads, -1, self.depth)
        v = v.reshape(batch_size * self.num_of_heads, -1, self.depth)

        # mask操作
        if mask is not None:
            mask = mask.repeat(self.num_of_heads, 1, 1)

        scaled_attention, attention_weights = self.attention(q, k, v, mask=mask)

        # scaled_attention.shape=(batch_size, sequence_len_q, num_of_heads, depth)
        scaled_attention = scaled_attention.view(batch_size, self.num_of_heads, sequence_len_q, self.depth).permute(0,
                                                                                                                    2,
                                                                                                                    1,
                                                                                                                    3)
        # attention_weights.shape=(batch_size, num_of_heads, sequence_len_q, sequence_len_k)
        attention_weights = attention_weights.view(batch_size, self.num_of_heads, sequence_len_q, sequence_len_k)

        # 拼接所有head
        # concat_attention.shape=(batch_size, sequence_len_q, dim_of_model)，其中dim_of_model = num_of_heads * depth
        concat_attention = scaled_attention.reshape(batch_size, sequence_len_q, -1)

        # 全连接层做线性输出
        linear_output = self.fc(concat_attention)

        return linear_output, attention_weights


def unit_test():
    # (num_of_heads, dim_of_model)
    mha = MultiHeadAttention(12, 768)

    # (batch_size, sequence_len_q, dim_of_model)
    q = torch.Tensor(32, 197, 768)

    output, attention_weights = mha(q, k=q, v=q, mask=None)

    print('shape of output:"{0}", shape of attention weight:"{1}"'.format(output.shape, attention_weights.shape))
