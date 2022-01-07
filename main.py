import os

import torch

from inference import Cifar10Inference
from models.teachers.resnet import ResNet50
from models.vi_transformer import VisionTransformer, DistillationVisionTransformer
from trainer import Trainer
from utils.config_utils import Config


def setup(t_model_path=None):
    models_cfg = Config()
    models_cfg.load_config(os.path.join(os.path.dirname(__file__), 'config/config.ini'))
    if models_cfg.items.model_type == 'vit':
        models = VisionTransformer(num_of_heads=models_cfg.items.num_of_heads,
                                   dim_of_model=models_cfg.items.dim_of_model,
                                   dim_of_mlp=models_cfg.items.dim_of_mlp,
                                   num_layers=models_cfg.items.num_layers,
                                   image_hw=models_cfg.items.img_size,
                                   channels=models_cfg.items.channels,
                                   patch_size=models_cfg.items.patch_size,
                                   em_dropout=models_cfg.items.em_dropout,
                                   atten_dropout=models_cfg.items.atten_dropout,
                                   mlp_dropout=models_cfg.items.mlp_dropout,
                                   num_classes=models_cfg.items.num_classes)
    else:
        if models_cfg.items.teacher == 'resnet50':
            teacher = ResNet50()
            if t_model_path is not None:
                teacher.load_state_dict(torch.load(t_model_path, map_location={'cuda:0': 'cuda:1'}))
        elif models_cfg.items.teacher == 'resnet101':
            teacher = None
        else:
            teacher = None

        models = DistillationVisionTransformer(num_of_heads=models_cfg.items.num_of_heads,
                                               dim_of_model=models_cfg.items.dim_of_model,
                                               dim_of_mlp=models_cfg.items.dim_of_mlp,
                                               num_layers=models_cfg.items.num_layers,
                                               image_hw=models_cfg.items.img_size,
                                               channels=models_cfg.items.channels,
                                               patch_size=models_cfg.items.patch_size,
                                               em_dropout=models_cfg.items.em_dropout,
                                               atten_dropout=models_cfg.items.atten_dropout,
                                               mlp_dropout=models_cfg.items.mlp_dropout,
                                               num_classes=models_cfg.items.num_classes,
                                               teacher=teacher,
                                               is_hard=models_cfg.items.is_hard,
                                               temperature=models_cfg.items.temperature,
                                               balancing=models_cfg.items.balancing)

    models.to(models_cfg.items.device)
    return models, models_cfg


def run_vit(t_name, models, models_cfg):
    trainer = Trainer(t_name, models, models_cfg)
    trainer.train()


def inference(models, models_path, image_path):
    ci = Cifar10Inference(models, models_path, cfg)
    ci.inference(image_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    t_model_path = '/home/dell/PycharmProjects/ViT/checkpoint/resnet50_ckpt.pth'
    model_path = '/home/dell/PycharmProjects/ViT/checkpoint/vit_teacher/vit_teacher_chkpt.bin'
    # model_path = '/home/dell/PycharmProjects/ViT/checkpoint/vit_fine_tuning/vit_fine_tuning_chkpt.bin'
    img_path = '/home/dell/PycharmProjects/ViT/data/demo/cat.jpeg'

    model, cfg = setup(t_model_path)

    trainer_name = "vit_teacher"
    run_vit(trainer_name, model, cfg)
    inference(model, model_path, img_path)
