import mlflow
import torch
from pytorch_lightning import LightningModule
from spikingjelly.activation_based import functional
from torch import optim

from losses.ntxent import TemporalNTXentLoss


class LitSelfSupervised(LightningModule):
    def __init__(
            self,
            model,
            lr=1e-4,
            weight_decay=0.0,
            max_epochs=100,
            ntxent_temp=0.2,
            strategy='naive'
    ):
        super().__init__()

        self.model = model

        self.criterion = TemporalNTXentLoss(temperature=ntxent_temp, strategy=strategy)

        self.lr = lr
        self.weight_decay = weight_decay
        self.eta_min = lr * 0.01
        self.max_epochs = max_epochs

    def forward(self, x_i, x_j):
        z_i = self.model(x_i)
        functional.reset_net(self.model)

        z_j = self.model(x_j)
        functional.reset_net(self.model)

        return z_i, z_j

    def _shared_step(self, batch, batch_idx, mode):
        (x_i, x_j), target = batch

        # (B, T, C, H, W) -> (T, B, C, H, W)
        x_i = torch.permute(x_i, (1, 0, 2, 3, 4))
        x_j = torch.permute(x_j, (1, 0, 2, 3, 4))

        z_i, z_j = self.forward(x_i, x_j)

        # Average over time dimension for contrastive loss
        pretrain_loss = self.criterion(z_i, z_j)

        self.log(f'{mode}_pretrain_loss', pretrain_loss, prog_bar=True, on_epoch=True, sync_dist=True)

        return pretrain_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, mode="train")

    def on_train_epoch_end(self) -> None:
        mlflow.log_metric("pretrain_loss", self.trainer.callback_metrics['train_pretrain_loss_epoch'].item(),
                          step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=self.eta_min, T_max=self.max_epochs)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

