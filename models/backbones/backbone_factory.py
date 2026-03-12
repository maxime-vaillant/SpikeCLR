import mlflow
import torch

from models.backbones.sew_resnet import SEWResNet18, SEWResNet18Sep
from models.backbones.spiking_vgg import SpikingVGG9
from utils.config import ExperimentConfig


class BackboneFactory:
    """Factory for creating backbone models."""

    @staticmethod
    def create(
            backbone_name: str,
            in_channels: int = 2,
            width: float = 1.0,
            cnf: 'str' = 'ADD',
            neuron_type: 'str' = 'LIF',
            **snn_kwargs
    ):
        """Create a backbone model.

        Args:
            backbone_name: Name of the backbone architecture
            in_channels: Number of input channels
            width: Width multiplier for the backbone
            cnf: Connection function for SNN (ADD, AND, etc.)
            neuron_type: Type of spiking neuron (LIF, IF, etc.)
            **snn_kwargs: Additional arguments for SNN (neuron_type, cnf, etc.)

        Returns:
            Backbone model and number of output features
        """
        if backbone_name.lower() == 'resnet18':
            backbone = SEWResNet18(
                in_channels=in_channels,
                width=width,
                cnf=cnf,
                neuron_type=neuron_type,
                **snn_kwargs
            )
            n_features = int(512 * width)
        elif backbone_name.lower() == 'resnet18sep':
            backbone = SEWResNet18Sep(
                in_channels=in_channels,
                width=width,
                cnf=cnf,
                neuron_type=neuron_type,
                **snn_kwargs
            )
            n_features = int(512 * width)
        elif backbone_name.lower() == 'vgg9':
            backbone = SpikingVGG9(
                in_channels=in_channels,
                width=width,
                neuron_type=neuron_type,
                **snn_kwargs
            )
            n_features = int(512 * width)
        else:
            raise ValueError(f"Unsupported SNN backbone: {backbone_name}")

        return backbone, n_features


    @staticmethod
    def load_pretrained_from_mlflow(run_id: str, config: ExperimentConfig):
        """Load a pretrained backbone from an MLflow run.

        Args:
            run_id: MLflow run ID containing the pretrained weights
            config: ExperimentConfig with model specifications

        Returns:
            Tuple of (backbone, n_features)
        """
        import tempfile

        print(f"\n{'=' * 60}")
        print(f"Loading pretrained weights from MLflow run: {run_id}")
        print(f"{'=' * 60}\n")

        # Create a fresh backbone with the same architecture
        backbone, n_features = BackboneFactory.create(
            config.backbone_name,
            in_channels=2,
            width=config.backbone_width,
            neuron_type=config.neuron_type,
            cnf=config.cnf,
            **config.neuron_kwargs
        )

        # Download the weights from MLflow
        client = mlflow.tracking.MlflowClient()

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = client.download_artifacts(run_id, "pretrained_weights/backbone_weights.pth", tmp_dir)

            # Load the state dict
            state_dict = torch.load(local_path, map_location='cpu')
            backbone.load_state_dict(state_dict)

            print(f"✓ Successfully loaded pretrained weights from run {run_id}\n")

        return backbone, n_features
