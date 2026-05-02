import torch
import torch.nn as nn

class RegressLoss(nn.Module):
    def __init__(self, args):
        super(RegressLoss, self).__init__()
        self.args = args

    def forward(self, inputs, targets):

        # loss = (inputs - targets)**2
        loss = torch.abs(inputs - targets)

        if self.args.fold != 2:
            fold_from = self.args.fold * 8
            fold_to = self.args.fold * 8 + 16
            loss[(199+fold_from):(199+fold_to)] = 0
        else:
            loss[(199+0):(199+8)] = 0
            loss[(199+16):(199+24)] = 0
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

        loss_ = 0.24*loss_src + 1.99*loss_trg

        return loss_/200