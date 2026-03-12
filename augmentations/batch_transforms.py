import numpy as np
import torch


class BatchCutMix:
    """
    Applies CutMix augmentation at the batch level.

    CutMix is a data augmentation technique that cuts and pastes patches between training images,
    where the ground truth labels are mixed proportionally to the area of the patches.

    This implementation uses sequential circular pairing instead of random permutation and applies
    different random box positions and sizes for each sample in the batch.

    Attributes:
        prob (float): Probability of applying CutMix to a batch. Default: 0.5
    """

    def __init__(self, prob=0.5):
        """
        Initialize BatchCutMix.

        Args:
            prob (float, optional): Probability of applying CutMix. Default: 0.5
        """
        self.prob = prob

    def __call__(self, batch, targets):
        """
        Apply CutMix at batch level with sequential pairing and per-sample random box positions.

        For each sample in the batch, a random rectangular region is cut from the paired sample
        and pasted onto the current sample. The pairing follows a sequential circular pattern
        where sample i is paired with sample (i+1) % batch_size.

        Args:
            batch (torch.Tensor): Input batch of shape (B, T, C, H, W)
                                 where B is batch size, T is temporal dimension (optional),
                                 C is channels, H is height, W is width.
            targets (torch.Tensor): Class labels of shape (B,) where B is batch size.

        Returns:
            tuple: A tuple containing:
                - mixed_batch (torch.Tensor): The batch with CutMix applied, same shape as input.
                - targets_a (torch.Tensor): Original targets, shape (B,).
                - targets_b (torch.Tensor): Targets of the paired samples, shape (B,).
                - lambdas (torch.Tensor): Mixing coefficients for each sample, shape (B,).
                                         Lambda represents the proportion of the original image.

        Note:
            If CutMix is not applied (based on prob), returns the original batch with
            targets_a = targets_b = targets and lambdas = all ones.
        """
        if np.random.rand() > self.prob:
            return batch, targets, targets, torch.ones(targets.shape[0], device=batch.device)

        batch_size = targets.shape[0]

        H, W = batch.shape[-2], batch.shape[-1]

        # Sequential circular pairing instead of random permutation
        index = torch.arange(1, batch_size + 1, device=batch.device) % batch_size

        targets_b = targets[index]
        lambdas = torch.zeros(batch_size, device=batch.device)

        # Apply different random ratio and box position for each sample
        for i in range(batch_size):
            # Sample random ratio for this specific sample
            ratio = np.random.rand() * 0.9 + 0.1

            cut_h = int(ratio * H)
            cut_w = int(ratio * W)

            # Sample random position for this specific pair
            bbx1 = np.random.randint(0, W - cut_w + 1)
            bby1 = np.random.randint(0, H - cut_h + 1)
            bbx2 = bbx1 + cut_w
            bby2 = bby1 + cut_h

            # Apply CutMix based on batch dimension
            batch[i, ..., bby1:bby2, bbx1:bbx2] = batch[index[i], ..., bby1:bby2, bbx1:bbx2]

            # Compute lambda as 1 - ratio^2
            lambdas[i] = 1 - ratio ** 2

        return batch, targets, targets_b, lambdas