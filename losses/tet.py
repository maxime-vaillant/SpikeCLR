import torch
from torch import nn


class TemporalEfficientTrainingLoss(nn.Module):
    """
    Temporal Efficient Training (TET) loss for spiking neural networks.

    Reference: "Training High-Performance Low-Latency Spiking Neural Networks
    by Differentiation on Spike Representation" (Deng & Gu, NeurIPS 2022).

    Args:
        alpha:    Weight of the MSE regularization term (default: 0, i.e. disabled).
        sigma:    Target mean firing rate for the regularization (default: 1.0).
        strategy: Reduction strategy over the time dimension.
                  - ``'naive'``:    average logits across T, then compute one CE loss.
                  - ``'temporal'``: compute CE at each time step and average.
    """

    STRATEGIES = ("naive", "temporal")

    def __init__(self, alpha=0, sigma=1.0, strategy: str = "naive"):
        super().__init__()

        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}."
            )

        self.alpha = alpha
        self.sigma = sigma
        self.strategy = strategy

        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.mse = nn.MSELoss()

    def _ce_loss(self, logits, targets, targets_b=None, lam=None):
        """Compute (optionally mixed) cross-entropy for a single (B, C) logit tensor."""
        if targets_b is not None and lam is not None:
            loss = lam * self.cross_entropy(logits, targets) + (1 - lam) * self.cross_entropy(logits, targets_b)
        else:
            loss = self.cross_entropy(logits, targets)
        return loss.mean()

    def forward(self, inputs, targets, targets_b=None, lam=None):
        """
        Args:
            inputs:   Model outputs. Either shape ``(T, B, C)``
                      for temporal (spiking) models.
            targets:  Ground-truth class indices, shape ``(B,)``.
            targets_b: Second set of targets for Mixup / CutMix, shape ``(B,)``.
            lam:      Mixing coefficient for Mixup / CutMix, scalar or shape ``(B,)``.

        Returns:
            loss: Scalar loss value.
        """
        # inputs shape: (T, B, C)
        T = inputs.shape[0]

        if self.strategy == "naive":
            # Average over time, then compute a single CE loss
            out = inputs.mean(dim=0)  # (B, C)
            ce = self._ce_loss(out, targets, targets_b, lam)
        else:
            # Compute CE at each time step and average
            ce = sum(self._ce_loss(inputs[t], targets, targets_b, lam) for t in range(T)) / T

        # Optional MSE regularization: encourage uniform firing rate across time steps
        if self.alpha > 0:
            sigmas = torch.full_like(inputs, self.sigma)
            mse = self.mse(inputs, sigmas)
            return ce + self.alpha * mse

        return ce
