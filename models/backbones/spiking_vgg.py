import torch.nn as nn
from spikingjelly.activation_based import layer, functional

from utils.spiking_neuron import get_spiking_neuron


class SpikingConvVGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool, neuron_type='LIF', **kwargs):
        super().__init__()
        self.conv = layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = layer.BatchNorm2d(num_features=out_channels)
        self.neuron = get_spiking_neuron(neuron_type, **kwargs)

        self.pool = None
        if pool:
            self.pool = layer.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.neuron(x)

        if self.pool is not None:
            x = self.pool(x)

        return x


class SpikingVGG9(nn.Module):
    def __init__(self, in_channels=2, width=1.0, neuron_type='LIF', **kwargs):
        super().__init__()

        def c(x):
            return int(x * width)

        # VGG11 configuration: [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P']
        # where 'M' stands for MaxPool

        self.network = nn.Sequential(
            SpikingConvVGGBlock(in_channels, c(64), pool=True, neuron_type=neuron_type, **kwargs),
            SpikingConvVGGBlock(c(64), c(128), pool=True, neuron_type=neuron_type, **kwargs),

            SpikingConvVGGBlock(c(128), c(256), pool=False, neuron_type=neuron_type, **kwargs),
            SpikingConvVGGBlock(c(256), c(256), pool=True, neuron_type=neuron_type, **kwargs),

            SpikingConvVGGBlock(c(256), c(512), pool=False, neuron_type=neuron_type, **kwargs),
            SpikingConvVGGBlock(c(512), c(512), pool=True, neuron_type=neuron_type, **kwargs),

            SpikingConvVGGBlock(c(512), c(512), pool=False, neuron_type=neuron_type, **kwargs),
            SpikingConvVGGBlock(c(512), c(512), pool=False, neuron_type=neuron_type, **kwargs),

            layer.AdaptiveAvgPool2d(output_size=(1, 1)),

            layer.Flatten(start_dim=1)
        )

        # Set step mode for temporal processing
        functional.set_step_mode(self, step_mode='m')

        # Initialize weights
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.network(x)

        return x

