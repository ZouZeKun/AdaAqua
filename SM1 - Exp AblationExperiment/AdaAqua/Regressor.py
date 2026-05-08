import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_lens = args.sequence_len
        self.hidden_dims = args.hidden_dims
        self.modes = args.modes

        self.regressor_1 = nn.Sequential(
            nn.Linear(4*self.hidden_dims, self.hidden_dims),
            nn.Tanh(),
            nn.Linear(self.hidden_dims, 4*self.hidden_dims),
            )

        self.regressor_2 = nn.Sequential(
            nn.Linear(4*self.hidden_dims, self.hidden_dims),
            nn.Linear(self.hidden_dims, 1),
            )

    def forward(self, f):
        f = self.regressor_1(f) + f
        f = self.regressor_2(f).squeeze(-1)
        
        return f