import numpy as np
import torch
import math
from torch import nn

def scoring_func(error_arr):
    pos_error_arr = error_arr[error_arr >= 0]
    neg_error_arr = error_arr[error_arr < 0]
    score = 0
    for error in neg_error_arr:
        score = math.exp(-(error / 13)) - 1 + score
    for error in pos_error_arr:
        score = math.exp(error / 10) - 1 + score
    return score


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def late_penalize_loss(y_pred, y_true):
    diff = y_pred - y_true
    diff2 = diff ** 2
    mask = torch.sigmoid(diff)
    losses = diff2 * (1 - mask) + diff2 * mask * 2
    loss = losses.mean()
    return torch.sqrt(loss)
# Qunatile Loss
class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
# Single Qunatile Loss
def compute_quantile_loss(y_pred, y_true, quantile):
    """
    Parameters
    ----------
    y_true : 1d ndarray
        Target value.

    y_pred : 1d ndarray
        Predicted value.

    quantile : float, 0. ~ 1.
        Quantile to be evaluated, e.g., 0.5 for median.
    """
    residual = y_pred - y_true
    return np.maximum(quantile * residual, (quantile - 1) * residual)