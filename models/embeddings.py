import torch
import torch.nn as nn
from torch.nn import Conv2d, Dropout


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, image_hw, dim_of_model, channels=3, patch_size=16, dropout=0.1):
        super().__init__()

        height = image_hw[0]
        width = image_hw[1]

        assert height % patch_size == 0 and width % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        n_patches = (height * width) // (patch_size ** 2)

        # 对原图做CNN卷积，提取特征，卷积核大小和卷积步长为切片大小，所以输出向量的后两个维度等于n_patches开根号
        self.patch_embeddings = Conv2d(in_channels=channels,
                                       out_channels=dim_of_model,
                                       kernel_size=(patch_size, patch_size),
                                       stride=(patch_size, patch_size))

        # shape=(1, n_patches+1, dim_of_model)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, dim_of_model))

        self.class_token = nn.Parameter(torch.zeros(1, 1, dim_of_model))

        self.dropout = Dropout(dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.class_token.expand(batch_size, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class DistillationEmbeddings(Embeddings):
    def __init__(self, image_hw, dim_of_model, channels=3, patch_size=16, dropout=0.1):
        super().__init__(image_hw, dim_of_model, channels, patch_size, dropout)

        self.distillation_token = nn.Parameter(torch.zeros(1, 1, dim_of_model))

    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.class_token.expand(batch_size, -1, -1)
        dis_tokens = self.distillation_token.expand(batch_size, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = torch.cat((embeddings, dis_tokens), dim=1)
        embeddings = self.dropout(embeddings)
        return embeddings


def unit_test():
    emb = Embeddings((224, 224), 768)
    q = torch.Tensor(32, 3, 224, 224)
    r = emb(q)
    print('shape of embedding:"{0}""'.format(r.shape)) #[32, 197, 768]
    exit(0)