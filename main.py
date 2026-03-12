import argparse

from spikingjelly.activation_based import surrogate

from datasets.dataset_factory import DatasetFactory
from pipelines.experiment import run_self_supervised_experiment
from utils.config import ExperimentConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SpikeCLR self-supervised learning experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-dataset", default="cifar10dvs",
                        choices=list(DatasetFactory.DATASETS.keys()),
                        help="Dataset to evaluate on")
    parser.add_argument("--ssl-datasets", nargs="+", default=None,
                        help="Datasets for SSL pretraining (defaults to target-dataset)")
    parser.add_argument("--backbone", dest="backbone_name", default="resnet18",
                        help="Backbone architecture (resnet18, resnet18sep, vgg9)")
    parser.add_argument("--backbone-width", type=float, default=1.0,
                        help="Width multiplier for the backbone")
    parser.add_argument("--projection-dim", type=int, default=128,
                        help="Output dimension of the projection head")
    parser.add_argument("--pretrain-epochs", type=int, default=500)
    parser.add_argument("--eval-epochs", type=int, default=150)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--eval-lr-linear", type=float, default=1e-3)
    parser.add_argument("--eval-lr-finetune", type=float, default=1e-3)
    parser.add_argument("--n-time-bins", type=int, default=4)
    parser.add_argument("--resize-size", type=int, nargs=2, default=[48, 48],
                        metavar=("H", "W"))
    parser.add_argument("--representation", default="frame", choices=["frame", "voxel"],
                        help="Event representation: 'frame' (ToFrame) or 'voxel' (ToVoxelGrid)")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="Normalize representation tensor to [0, 1]")
    parser.add_argument("--samples-per-class", type=float, nargs="+",
                        default=[0.00, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
                        dest="samples_per_class_list")
    parser.add_argument("--samples-mode", default="percent", choices=["percent", "count"])
    parser.add_argument("--n-subset-runs", type=int, default=3)
    parser.add_argument("--pretrain-samples", type=float, default=1.0,
                        dest="pretrain_samples_percentage")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU index (passed to PyTorch Lightning)")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-run-id", default=None,
                        help="MLflow run ID to load pretrained backbone from (skips pretraining)")
    parser.add_argument("--eval-types", nargs="+", default=["lp", "ft", "sup"],
                        choices=["lp", "ft", "sup"],
                        help="Evaluation types to run: lp (linear probing), ft (finetuning), sup (supervised)")
    parser.add_argument("--pretrain-loss-strategy", default="naive", choices=["naive", "temporal"],
                        dest="pretrain_loss_strategy",
                        help="Temporal NT-Xent loss strategy: 'naive' (mean then loss) or 'temporal' (loss then mean)")
    parser.add_argument("--supervised-loss-strategy", default="naive", choices=["naive", "temporal"],
                        dest="supervised_loss_strategy",
                        help="Cross entropy loss strategy: 'naive' (mean then loss) or 'temporal' (loss then mean)")
    parser.add_argument("--use-cutmix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cutmix-prob", type=float, default=0.5)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    print()

    config = ExperimentConfig(
        target_dataset=args.target_dataset,
        ssl_datasets=args.ssl_datasets,
        backbone_name=args.backbone_name,
        backbone_width=args.backbone_width,
        projection_dim=args.projection_dim,
        pretrain_epochs=args.pretrain_epochs,
        eval_epochs=args.eval_epochs,
        pretrain_batch_size=args.pretrain_batch_size,
        eval_batch_size=args.eval_batch_size,
        pretrain_lr=args.pretrain_lr,
        eval_lr_linear=args.eval_lr_linear,
        eval_lr_finetune=args.eval_lr_finetune,
        n_time_bins=args.n_time_bins,
        resize_size=tuple(args.resize_size),
        representation=args.representation,
        normalize=args.normalize,
        samples_per_class_list=args.samples_per_class_list,
        samples_mode=args.samples_mode,
        n_subset_runs=args.n_subset_runs,
        pretrain_samples_percentage=args.pretrain_samples_percentage,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        pretrained_run_id=args.pretrained_run_id,
        use_cutmix=args.use_cutmix,
        cutmix_prob=args.cutmix_prob,
        eval_types=args.eval_types,
        pretrain_loss_strategy=args.pretrain_loss_strategy,
        supervised_loss_strategy=args.supervised_loss_strategy,
        **{'decay_input': False, 'surrogate_function': surrogate.ATan(), 'backend': 'cupy'}
    )

    run_self_supervised_experiment(config)


if __name__ == "__main__":
    main()
