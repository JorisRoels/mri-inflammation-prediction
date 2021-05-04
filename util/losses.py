import torch
import torch.nn as nn
from torch import lgamma

from util.constants import *


def _gamma_pdf(x, alpha):
    g = torch.exp(lgamma(torch.ones_like(x) * alpha))
    g = (torch.pow(x, alpha - 1) * torch.exp(- x)) / g
    return g


class SPARCCSimilarityLoss(nn.Module):
    """
    SPARCC similarity loss function

    :param initalization alpha: parameter that offers trade-off between two desidered properties:
            - the loss increases as the absolute difference between the prediction and target increases
            - the loss increases as the target decreases (i.e. mistakes with low target scores are more important)
    :param forward pred: SPARCC prediction (B)
    :param forward target: SPARCC target (B)
    :return: SPARCC similarity loss
    """

    def __init__(self, alpha=1.3, beta=5, w_sparcc=None):
        super(SPARCCSimilarityLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.w_sparcc = w_sparcc

    def forward(self, pred, target):

        w = torch.Tensor([1] * len(target)).to(target.device)
        if self.w_sparcc is not None:
            w = torch.Tensor([self.w_sparcc[int(t*BINS)] for t in target]).to(target.device)

        l1 = torch.abs(pred - target)
        reg = _gamma_pdf(l1, self.alpha)
        loss = l1 + torch.exp(- self.beta * target) * reg

        return torch.sum(w * loss)