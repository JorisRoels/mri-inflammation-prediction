import torch
import torch.nn as nn

from util.constants import *

eps = 1e-5


def inv_gaussian(x, l=0.2, m=5):
    return torch.sqrt(l / (2 * np.pi * (x+eps)**3)) * torch.exp(- (l * (x - m)**2) / (2 * m**2 * (x+eps)))


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

    def __init__(self, alpha=0.05, beta=10, w_sparcc=None):
        super(SPARCCSimilarityLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.w_sparcc = w_sparcc

    def forward(self, pred, target):

        w = torch.Tensor([1] * len(target)).to(target.device)
        if self.w_sparcc is not None:
            w = torch.Tensor([self.w_sparcc[int(t*BINS)] for t in target]).to(target.device)

        loss = torch.abs(pred - target)
        # reg = inv_gaussian(loss)
        # loss = loss + self.alpha * torch.exp(- self.beta * target) * reg

        return torch.sum(w * loss)