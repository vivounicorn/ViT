import configparser

import torch
from torch.nn.modules.utils import _pair
import ml_collections


class Config(object):
    def __init__(self):
        self.items = ml_collections.ConfigDict()

    def load_config(self, conf_path):
        cfg = configparser.ConfigParser()
        cfg.read(conf_path)

        # training configuration
        self.items.train_batch_size = cfg.getint('training', 'train_batch_size')
        self.items.test_batch_size = cfg.getint('training', 'test_batch_size')
        self.items.img_size = _pair(cfg.getint('training', 'img_size'))
        self.items.learning_rate = cfg.getfloat('training', 'learning_rate')
        self.items.num_steps = cfg.getint('training', 'num_steps')
        self.items.warmup_steps = cfg.getint('training', 'warmup_steps')
        self.items.warmup_steps = self.items.warmup_steps if self.items.warmup_steps > 0 else 1.0
        self.items.test_epoch = cfg.getint('training', 'test_epoch')
        self.items.pretrained_model = cfg.get('training', 'pretrained_model')
        self.items.optimizer = cfg.get('training', 'optimizer')

        # modeling configuration
        self.items.num_classes = cfg.getint('modeling', 'num_classes')
        self.items.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.items.num_of_heads = cfg.getint('modeling', 'num_of_heads')
        self.items.dim_of_model = cfg.getint('modeling', 'dim_of_model')
        self.items.dim_of_mlp = cfg.getint('modeling', 'dim_of_mlp')
        self.items.num_layers = cfg.getint('modeling', 'num_layers')
        self.items.channels = cfg.getint('modeling', 'channels')
        self.items.patch_size = cfg.getint('modeling', 'patch_size')
        self.items.em_dropout = cfg.getfloat('modeling', 'em_dropout')
        self.items.atten_dropout = cfg.getfloat('modeling', 'atten_dropout')
        self.items.mlp_dropout = cfg.getfloat('modeling', 'mlp_dropout')