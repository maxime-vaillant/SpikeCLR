import mlflow
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from spikingjelly.activation_based import functional
from torch import optim

from augmentations.batch_transforms import BatchCutMix
from losses.tet import TemporalEfficientTrainingLoss
from utils.metrics import TopKAccuracy


class LitSupervised(LightningModule):
    def __init__(
            self,
            model,
            lr=1e-3,
            weight_decay=0.0,
            max_epochs=100,
            use_cutmix=True,
            cutmix_prob=0.5,
            strategy='naive'
    ):
        super().__init__()

        self.model = model

        functional.set_step_mode(self.model, 'm')

        self.criterion = TemporalEfficientTrainingLoss(alpha=0.0, strategy=strategy)
        self.metric = TopKAccuracy(k=1)

        self.lr = lr
        self.weight_decay = weight_decay
        self.eta_min = lr * 0.01
        self.max_epochs = max_epochs

        self.best_val_acc = 0.0

        # CutMix configuration
        self.use_cutmix = use_cutmix
        self.cutmix = BatchCutMix(prob=cutmix_prob) if use_cutmix else None

    def forward(self, x):
        y = self.model(x)

        functional.reset_net(self.model)

        return y

    def _shared_step(self, batch, mode):
        x, target = batch

        # Apply CutMix only during training
        if mode == 'train' and self.use_cutmix and self.cutmix is not None:
            x, targets_a, targets_b, lambdas = self.cutmix(x, target)
        else:
            targets_a = target
            targets_b = None
            lambdas = None

        # (BxS, T, C, H, W) -> (T, BxS, C, H, W)
        x = torch.permute(x, (1, 0, 2, 3, 4))

        # Forward returns (T, BxS, C)
        y = self.forward(x)

        # Compute loss
        loss = self.criterion(y, targets_a, targets_b, lambdas)

        self.log(f'{mode}_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # Accuracy metric (using original targets)
        value = self.metric(F.softmax(y.mean(0), dim=1), target)
        self.log(f'{mode}_{self.metric}', value, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')

    def on_train_epoch_end(self):
        mlflow.log_metric("train_loss", self.trainer.callback_metrics['train_loss_epoch'].item(), step=self.current_epoch)
        mlflow.log_metric("train_top_1_acc", self.trainer.callback_metrics['train_top_1_acc_epoch'].item(), step=self.current_epoch)

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics['val_loss'].item()
        val_acc = self.trainer.callback_metrics['val_top_1_acc'].item()

        # Update best accuracy
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

        mlflow.log_metric("val_loss", val_loss, step=self.current_epoch)
        mlflow.log_metric("val_top_1_acc", val_acc, step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=self.eta_min, T_max=self.max_epochs)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}