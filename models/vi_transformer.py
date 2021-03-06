import torch
import torch.nn as nn
from models.tf_encoder import TransformerEncoder, DistillationTransformerEncoder
from torch.nn import Linear
from torchvision import transforms
from PIL import Image

from utils.func_utils import soft_distillation, hard_distillation


class VisionTransformer(nn.Module):
    """Vit."""

    def __init__(self, num_of_heads, dim_of_model, dim_of_mlp, num_layers,
                 image_hw=(224, 224), channels=3, patch_size=16,
                 em_dropout=0.1, atten_dropout=0.1, mlp_dropout=0.1, num_classes=10):
        super().__init__()

        self.num_classes = num_classes
        self.num_of_heads = num_of_heads
        self.dim_of_model = dim_of_model
        self.dim_of_mlp = dim_of_mlp
        self.num_layers = num_layers
        self.image_hw = image_hw
        self.channels = channels
        self.patch_size = patch_size
        self.em_dropout = em_dropout
        self.atten_dropout = atten_dropout
        self.mlp_dropout = mlp_dropout

        self.transformer = TransformerEncoder(self.num_of_heads, self.dim_of_model, self.dim_of_mlp, self.num_layers,
                                              self.image_hw, self.channels, self.patch_size,
                                              self.em_dropout, self.atten_dropout, self.mlp_dropout)

        self.vit_head = Linear(self.dim_of_model, self.num_classes)

        nn.init.zeros_(self.vit_head.weight)
        nn.init.zeros_(self.vit_head.bias)
        self.model_paras_summary()

    def model_paras_summary(self):
        print("\033[32m***** ViT Model Parameters *****\033[0m")
        print("  heads number of multi-head attention:\033[31m%d\033[0m" % self.num_of_heads)
        print("  dimension of hidden states:\033[31m%d\033[0m" % self.dim_of_model)
        print("  dimension of mlp:\033[31m%d\033[0m" % self.dim_of_mlp)
        print("  number of block layers:\033[31m%d\033[0m" % self.num_layers)
        print("  height and width of image:", self.image_hw)
        print("  image channels:\033[31m%d\033[0m" % self.channels)
        print("  size of patches(it can be divided by image height*width):\033[31m%d\033[0m" % self.patch_size)
        print("  dropout of embedding layer:\033[31m%f\033[0m" % self.em_dropout)
        print("  dropout of attention layer:\033[31m%f\033[0m" % self.atten_dropout)
        print("  dropout of mlp:\033[31m%f\033[0m" % self.mlp_dropout)
        print("\033[32m***** End *****\033[0m")

    def forward(self, x, labels=None):
        encoded, attention_weights = self.transformer(x)
        obj_res = self.vit_head(encoded[:, 0])

        if labels is not None:
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(obj_res.view(-1, self.num_classes), labels.view(-1))
            return loss, obj_res
        else:
            return obj_res, attention_weights


class DistillationVisionTransformer(VisionTransformer):
    def __init__(self, num_of_heads, dim_of_model, dim_of_mlp, num_layers,
                 image_hw=(224, 224), channels=3, patch_size=16,
                 em_dropout=0.1, atten_dropout=0.1, mlp_dropout=0.1, num_classes=10,
                 teacher=None, is_hard=False, temperature=3.0, balancing=0.1):

        super().__init__(num_of_heads, dim_of_model, dim_of_mlp, num_layers,
                         image_hw, channels, patch_size,
                         em_dropout, atten_dropout, mlp_dropout, num_classes)

        self.transformer = DistillationTransformerEncoder(self.num_of_heads, self.dim_of_model, self.dim_of_mlp,
                                                          self.num_layers,
                                                          self.image_hw, self.channels, self.patch_size,
                                                          self.em_dropout, self.atten_dropout, self.mlp_dropout)
        self.teacher = teacher
        self.is_hard = is_hard
        self.temperature = temperature
        self.balancing = balancing

    def forward(self, x, labels=None):
        encoded, attention_weights = self.transformer(x)
        student_logits = self.vit_head(encoded[:, 0])
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        if labels is not None:
            if not self.is_hard:
                loss = soft_distillation(teacher_logits, student_logits,
                                         self.temperature, self.balancing, labels, self.num_classes)
            else:
                loss = hard_distillation(teacher_logits, student_logits,
                                         self.balancing, labels, self.num_classes)

            return loss, student_logits
        else:
            return student_logits, attention_weights


def unit_test(img, vit_type='vit'):
    resolution = (224, 224)
    num_of_heads = 12
    dim_of_model = 768
    dim_of_mlp = 3072
    num_layers = 2

    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor()])

    im = Image.open(img).convert('RGB')
    x = transform(im)

    if vit_type == 'vit':
        vit = VisionTransformer(num_of_heads, dim_of_model, dim_of_mlp, num_layers, resolution)
    else:
        vit = unit_test_build_teacher(num_of_heads, dim_of_model, dim_of_mlp, num_layers, resolution)

    q = x.reshape(1, 3, 224, 224)
    # attention_matrix is a list.
    result, attention_matrix = vit(q)
    # attention_matrix size: (2, 197, 197)
    attention_matrix = torch.mean(torch.stack(attention_matrix).squeeze(1), dim=1)

    print('shape of vision transformer:"{0}""'.format(attention_matrix.size()))

    from torchviz import make_dot
    dis_net = make_dot(result, params=dict(list(vit.named_parameters())))
    dis_net.format = "png"
    # ??????????????????????????????
    dis_net.directory = "data"
    # ????????????
    dis_net.view()


def unit_test_build_teacher(num_of_heads, dim_of_model, dim_of_mlp, num_layers, resolution):
    from torchvision.models import resnet50
    teacher = resnet50(pretrained=True)
    vit = DistillationVisionTransformer(num_of_heads, dim_of_model, dim_of_mlp, num_layers, resolution, teacher=teacher)
    return vit

# unit_test('/home/dell/PycharmProjects/ViT/data/demo/ship.jpeg','vvv')
