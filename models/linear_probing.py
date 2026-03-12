from spikingjelly.activation_based import layer, functional
from torch import nn


class LinearProbingModule(nn.Module):
    def __init__(self, backbone: nn.Module, n_features: int, n_classes: int):
        super().__init__()

        self.backbone = backbone

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classification_head = nn.Sequential(
            layer.Linear(n_features, n_classes),
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        h = self.backbone(x)
        out = self.classification_head(h)

        return out