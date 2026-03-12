import torch

class TopKAccuracy:
    def __init__(self, k: int = 1):
        self.k = k

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        top_k = torch.topk(y_pred, self.k, dim=1).indices
        correct = top_k.eq(y_true.unsqueeze(1)).any(dim=1)

        return correct.float().mean().item()

    def __str__(self):
        return f'top_{self.k}_acc'