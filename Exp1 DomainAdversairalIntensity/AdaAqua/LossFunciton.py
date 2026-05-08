import torch
import torch.nn as nn

class RegressLoss(nn.Module):
    def __init__(self, args):
        super(RegressLoss, self).__init__()
        self.args = args

    def forward(self, inputs, targets):

        loss = torch.abs(inputs - targets)

        fold_from = self.args.fold * 7
        fold_to = self.args.fold * 7 + 7

        loss[(199+fold_from):(199+fold_to)] = 0
        loss = torch.mean(loss)
        
        return loss


class DomainLoss(nn.Module):
    def __init__(self, args):
        super(DomainLoss, self).__init__()
        self.args = args

    def forward(self, inputs, targets):

        mask_src = (targets == -1).float()
        mask_trg = (targets == 1).float()

        loss_src = torch.sum(mask_src * torch.abs(inputs - targets))
        loss_trg = torch.sum(mask_trg * torch.abs(inputs - targets))

        loss_ = 0.16*loss_src + 1.99*loss_trg

        return loss_/200