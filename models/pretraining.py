from spikingjelly.activation_based import layer, functional
from torch import nn

from utils.spiking_neuron import get_spiking_neuron


class SpikeCLR(nn.Module):
    def __init__(self, backbone: nn.Module, n_features: int, projection_dim: int, neuron_type='LIF', **kwargs):
        super().__init__()

        self.backbone = backbone

        self.projection_head = nn.Sequential(
            layer.Linear(n_features, 2048),
            layer.BatchNorm1d(2048),
            get_spiking_neuron(neuron_type=neuron_type, **kwargs),
            layer.Linear(2048, projection_dim),
            layer.BatchNorm1d(projection_dim),
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)

        return z