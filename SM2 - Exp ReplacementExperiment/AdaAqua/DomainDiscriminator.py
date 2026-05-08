import torch
import torch.nn as nn

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    # 更大的alpha值会导致更强的梯度反转,强制特征提取器学习更具领域不变性的特征
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def gradient_reversal(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_lens = args.sequence_len
        self.hidden_dims = args.hidden_dims
        self.modes = args.modes

        self.classifier_1 = nn.Sequential(
            nn.Linear(4*self.hidden_dims, self.hidden_dims),
            nn.Tanh(),
            nn.Linear(self.hidden_dims, 1),
            )
        self.classifier_2 = nn.Sequential(
            nn.Linear(self.seq_lens, 1),
            nn.Tanh()
            )

    def forward(self, f):
        
        f = self.classifier_1(f).squeeze(-1)
        f = self.classifier_2(f).squeeze(-1)

        return f   