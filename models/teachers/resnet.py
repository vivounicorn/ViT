import torchvision
import torch.nn as nn


class ResNet50(nn.Module):
    def __init__(self, num_classes=10, num_hidden=1000, pretrained=True):
        super().__init__()

        self.resnet = torchvision.models.resnet50(pretrained)
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self._make_layer()

    def _make_layer(self):
        num_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, self.num_hidden),
            nn.ReLU(),
            nn.Linear(self.num_hidden, self.num_classes))

    def forward(self, x):
        out = self.resnet(x)
        return out
