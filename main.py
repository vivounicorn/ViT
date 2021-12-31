import os

from inference import Cifar10Inference
from models.vi_transformer import VisionTransformer
from trainer import Trainer
from utils.config_utils import Config
from utils.data_utils import load_pretrained


def run():
    cfg = Config()
    cfg.load_config(os.path.join(os.path.dirname(__file__), 'config/config.ini'))

    model = VisionTransformer(num_of_heads=cfg.items.num_of_heads, dim_of_model=cfg.items.dim_of_model,
                              dim_of_mlp=cfg.items.dim_of_mlp, num_layers=cfg.items.num_layers,
                              image_hw=cfg.items.img_size, channels=cfg.items.channels,
                              patch_size=cfg.items.patch_size, em_dropout=cfg.items.em_dropout,
                              atten_dropout=cfg.items.atten_dropout, mlp_dropout=cfg.items.mlp_dropout,
                              num_classes=cfg.items.num_classes)

    load_pretrained(model, '/home/dell/PycharmProjects/ViT/finetuning/ViT-B_32.npz')
    model.to(cfg.items.device)

    trainer = Trainer("vit_fine_tuning", model, cfg)
    trainer.train()


def inference(model_path, img_path):
    ci = Cifar10Inference(model_path)
    ci.inference(img_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()
    model_path = '/home/dell/PycharmProjects/ViT/checkpoint/vit_fine_tuning/vit_fine_tuning_chkpt.bin'
    img_path = '/home/dell/PycharmProjects/ViT/data/demo/car.jpeg'
    inference(model_path, img_path)

