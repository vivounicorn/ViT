import os

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from torch import nn
from torchvision import transforms


class Cifar10Inference(object):
    """
    inference class for cifar10 dataset.
    """

    def __init__(self, model, model_path, cfg):
        self.cifar10_labels = dict(
            enumerate(open(os.path.join(os.path.dirname(__file__), 'config/cifar10_labels.txt'))))

        self.cfg = cfg
        self.model = model
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location={'cuda:0': 'cuda:1'}))

    def inference(self, img_path):
        self.model.eval()

        transform = transforms.Compose([
            transforms.Resize(self.cfg.items.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        im = Image.open(img_path).convert('RGB')
        x = transform(im)
        x = x.to(self.cfg.items.device)

        obj_res, attention_matrix = self.model(x.unsqueeze(0))
        attention_matrix = torch.stack(attention_matrix).squeeze(1)

        # 对所有header的attention map求均值
        attention_matrix = torch.mean(attention_matrix, dim=1)

        # 增加残差信息并归一化
        residual_att = torch.eye(attention_matrix.size(1)).to(self.cfg.items.device)
        augmentation_att_matrix = attention_matrix + residual_att
        augmentation_att_matrix = augmentation_att_matrix / augmentation_att_matrix.sum(dim=-1).unsqueeze(-1)

        # 循环连乘权重矩阵
        joint_attentions = torch.zeros(augmentation_att_matrix.size()).to(self.cfg.items.device)
        joint_attentions[0] = augmentation_att_matrix[0]

        for n in range(1, augmentation_att_matrix.size(0)):
            joint_attentions[n] = torch.matmul(augmentation_att_matrix[n], joint_attentions[n - 1])

        v = joint_attentions[-1]

        grid_size = int(np.sqrt(augmentation_att_matrix.size(-1)))
        attention_mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
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

    def classify(self, img_path):
        self.model.eval()

        transform = transforms.Compose([
            transforms.Resize(self.cfg.items.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        im = Image.open(img_path).convert('RGB')
        x = transform(im)
        x = x.to(self.cfg.items.device)

        obj_res = self.model(x.unsqueeze(0))

        model_weights = []
        conv_layers = []
        model_children = list(self.model.resnet.children())
        counter = 0
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            counter += 1
                            model_weights.append(child.weight)
                            conv_layers.append(child)

        outputs = []
        names = []
        image = x.unsqueeze(0)
        for layer in conv_layers[0:]:
            image = layer(image)
            outputs.append(image)
            names.append(str(layer))

        processed = []
        for feature_map in outputs:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(30,50))

        ax1.set_title('Original')
        ax2.set_title('First Feature Map')
        ax3.set_title('Last Feature Map')
        _ = ax1.imshow(im)
        _ = ax2.imshow(processed[0])
        _ = ax3.imshow(processed[len(processed)-1])

        plt.show()

        probs = torch.nn.Softmax(dim=-1)(obj_res)
        top5 = torch.argsort(probs, dim=-1, descending=True)
        print("Prediction Label and Attention Map!\n")
        for idx in top5[0, :5]:
            print(f'{probs[0, idx.item()]:.5f} : {self.cifar10_labels[idx.item()]}', end='')
