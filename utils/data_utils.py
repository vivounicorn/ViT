import logging

import os

import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)


def data_loader(img_size, train_batch_size, test_batch_size, dt_type='cifar10'):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if dt_type == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test)

    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test)

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=test_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader


def find_newest_model(models_dir):
    model_file_lists = os.listdir(models_dir)
    model_file_lists.sort(key=lambda fn: os.path.getmtime(models_dir + "/" + fn)
    if not os.path.isdir(models_dir + "/" + fn) else 0)

    model_name = model_file_lists[-1]
    model_file = os.path.join(models_dir, model_file_lists[-1])
    if model_name.find(".bin") == -1:
        return None, None

    return model_name, model_file


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def load_pretrained(model, path):
    weights = np.load(path)
    with torch.no_grad():
        nn.init.zeros_(model.vit_head.weight)
        nn.init.zeros_(model.vit_head.bias)
        # model.vit_head.weight.copy_(np2th(weights["head/kernel"]).t())
        # model.vit_head.bias.copy_(np2th(weights["head/bias"]).t())

        model.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
        model.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
        model.transformer.embeddings.class_token.copy_(np2th(weights["cls"]))
        model.transformer.transformer_encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
        model.transformer.transformer_encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

        posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
        posemb_new = model.transformer.embeddings.position_embeddings
        if posemb.size() == posemb_new.size():
            model.transformer.embeddings.position_embeddings.copy_(posemb)
        else:
            ntok_new = posemb_new.size(1)

            posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

            gs_old = int(np.sqrt(len(posemb_grid)))
            gs_new = int(np.sqrt(ntok_new))
            print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
            posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

            zoom = (gs_new / gs_old, gs_new / gs_old, 1)
            posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
            posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
            posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
            model.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

        for bname, block in model.transformer.transformer_encoder.named_children():
            for uname, unit in block.named_children():
                block_load(unit, weights, uname, unit.hidden_size)


from os.path import join as pjoin
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def block_load(block, weights, n_block, hidden_size):
    ROOT = f"Transformer/encoderblock_{n_block}"
    with torch.no_grad():
        query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(hidden_size, hidden_size).t()
        key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(hidden_size, hidden_size).t()
        value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(hidden_size, hidden_size).t()
        out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(hidden_size, hidden_size).t()

        query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
        key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
        value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
        out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

        block.mh_attention.w_qs.weight.copy_(query_weight)
        block.mh_attention.w_ks.weight.copy_(key_weight)
        block.mh_attention.w_vs.weight.copy_(value_weight)
        block.mh_attention.fc.weight.copy_(out_weight)
        block.mh_attention.w_qs.bias.copy_(query_bias)
        block.mh_attention.w_ks.bias.copy_(key_bias)
        block.mh_attention.w_vs.bias.copy_(value_bias)
        block.mh_attention.fc.bias.copy_(out_bias)

        mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
        mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
        mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
        mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

        block.mlp.fc1.weight.copy_(mlp_weight_0)
        block.mlp.fc2.weight.copy_(mlp_weight_1)
        block.mlp.fc1.bias.copy_(mlp_bias_0)
        block.mlp.fc2.bias.copy_(mlp_bias_1)

        block.mh_attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
        block.mh_attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
        block.mlp_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
        block.mlp_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))