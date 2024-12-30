import pdb

import torch
from torch import nn


class LossBCEMonotonic(nn.BCELoss):
    def __init__(self, gamma=100):
        super(LossBCEMonotonic, self).__init__()
        self.gamma = gamma

    def forward(self, x, y_pred, y_true):
        lossBCE = super(LossBCEMonotonic, self).forward(y_pred, y_true)
        ix = torch.argsort(x, axis=0)
        # pdb.set_trace()
        y_pred = y_pred[ix.flatten(), :]
        first_diff = -torch.diff(y_pred, axis=0)
        lossMono = torch.sum(torch.relu(first_diff))
        loss = lossBCE + self.gamma * lossMono
        # loss = lossBCE
        return loss


class LossBCEMonotonicAlpha(nn.Module):
    def __init__(self, alpha, gamma=100):
        super(LossBCEMonotonicAlpha, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, y_pred, y_true):
        #lossBCE = super(LossBCEMonotonic, self).forward(y_pred, y_true)
        #pdb.set_trace()
        logit0 = torch.log(1-y_pred[(y_true==0).flatten(),:])
        logit1 = torch.log(y_pred[(y_true == 1).flatten(), :])
        lossBCE1 = -self.alpha * torch.mean(torch.clamp(logit1, min=-100), axis=0)
        lossBCE0 = -(1-self.alpha) * torch.mean(torch.clamp(logit0, min=-100), axis=0)
        lossBCE = lossBCE1 + lossBCE0
        ix = torch.argsort(x, dim=0)
        # pdb.set_trace()
        y_pred = y_pred[ix.flatten(), :]
        first_diff = -torch.diff(y_pred, dim=0)
        lossMono = torch.sum(torch.relu(first_diff))
        loss = lossBCE + self.gamma * lossMono
        # loss = lossBCE
        return loss


def lossFunc2(x, y_pred, y_true):
    # lossBCE = BCE(y_pred, y_true)
    # ix = torch.argsort(x, axis=0)
    # #pdb.set_trace()
    # y_pred = y_pred[ix.flatten(),:]
    # first_diff = -torch.diff(y_pred, axis=0)
    # lossMono = torch.sum(torch.relu(first_diff))
    # loss = lossBCE + 100*lossMono
    loss = lossBCE
    return loss