from spikingjelly.activation_based import functional, layer
from torch import nn


class FinetuningModule(nn.Module):
    def __init__(self, backbone: nn.Module, n_features: int, n_classes: int):
        super().__init__()

        self.backbone = backbone

        self.classification_head = nn.Sequential(
            layer.Linear(n_features, n_classes),
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        h = self.backbone(x)
        out = self.classification_head(h)

        return out