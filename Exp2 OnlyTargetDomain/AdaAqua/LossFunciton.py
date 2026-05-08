import torch
import torch.nn as nn

class RegressLoss(nn.Module):
    def __init__(self, args):
        super(RegressLoss, self).__init__()
        self.args = args

    def forward(self, inputs, targets):

        # loss = (inputs - targets)**2
        loss = torch.abs(inputs - targets)

        fold_from = self.args.fold * 7
        fold_to = self.args.fold * 7 + 7

        loss[fold_from:fold_to] = 0
        loss = torch.mean(loss)
        
        return loss