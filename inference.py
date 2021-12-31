import os

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from utils.config_utils import Config

from models.vi_transformer import VisionTransformer


class Cifar10Inference(object):
    """
    inference class for cifar10 dataset.
    """
    def __init__(self, model_path):
        self.cifar10_labels = dict(enumerate(open(os.path.join(os.path.dirname(__file__), 'config/cifar10_labels.txt'))))

        self.cfg = Config()
        self.cfg.load_config(os.path.join(os.path.dirname(__file__), 'config/config.ini'))

        self.model = VisionTransformer(num_of_heads=self.cfg.items.num_of_heads,
                                       dim_of_model=self.cfg.items.dim_of_model,
                                       dim_of_mlp=self.cfg.items.dim_of_mlp,
                                       num_layers=self.cfg.items.num_layers,
                                       image_hw=self.cfg.items.img_size,
                                       channels=self.cfg.items.channels,
                                       patch_size=self.cfg.items.patch_size,
                                       em_dropout=self.cfg.items.em_dropout,
                                       atten_dropout=self.cfg.items.atten_dropout,
                                       mlp_dropout=self.cfg.items.mlp_dropout,
                                       num_classes=self.cfg.items.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location={'cuda:0':'cuda:1'}))

    def inference(self, img_path):
        self.model.eval()

        transform = transforms.Compose([
            transforms.Resize(self.cfg.items.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        im = Image.open(img_path).convert('RGB')
        x = transform(im)

        obj_res, attention_matrix = self.model(x.unsqueeze(0))
        attention_matrix = torch.stack(attention_matrix).squeeze(1)

        # 对所有header的attention map求均值
        attention_matrix = torch.mean(attention_matrix, dim=1)

        # 增加残差信息并归一化
        residual_att = torch.eye(attention_matrix.size(1))
        augmentation_att_matrix = attention_matrix + residual_att
        augmentation_att_matrix = augmentation_att_matrix / augmentation_att_matrix.sum(dim=-1).unsqueeze(-1)

        # 循环连乘权重矩阵
        joint_attentions = torch.zeros(augmentation_att_matrix.size())
        joint_attentions[0] = augmentation_att_matrix[0]

        for n in range(1, augmentation_att_matrix.size(0)):
            joint_attentions[n] = torch.matmul(augmentation_att_matrix[n], joint_attentions[n - 1])

        v = joint_attentions[-1]

        grid_size = int(np.sqrt(augmentation_att_matrix.size(-1)))
        attention_mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        attention_mask = cv2.resize(attention_mask / attention_mask.max(), im.size)[..., np.newaxis]
        result = (attention_mask * im).astype("uint8")

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 16))

        ax1.set_title('Original')
        ax2.set_title('Attention Mask')
        ax3.set_title('Attention Map')
        _ = ax1.imshow(im)
        _ = ax2.imshow(attention_mask)
        _ = ax3.imshow(result)

        plt.show()

        probs = torch.nn.Softmax(dim=-1)(obj_res)
        top5 = torch.argsort(probs, dim=-1, descending=True)
        print("Prediction Label and Attention Map!\n")
        for idx in top5[0, :5]:
            print(f'{probs[0, idx.item()]:.5f} : {self.cifar10_labels[idx.item()]}', end='')
