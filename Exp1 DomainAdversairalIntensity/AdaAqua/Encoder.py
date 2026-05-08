import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, args):
        super(PositionalEncoding, self).__init__()

        self.seq_lens = args.sequence_len
        self.hidden_dims = args.hidden_dims 
        self.numbers = args.num_buildings_source + args.num_buildings_target

        pe = torch.zeros(self.numbers, self.seq_lens)

        position = torch.arange(0, self.numbers, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.seq_lens, 2).float() * (-torch.log(torch.tensor(10000)) / self.seq_lens))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(-1)
        
        # 注册到缓冲区
        self.register_buffer('pe', pe) 

    def forward(self, x):

        x = torch.concat([x, self.pe], dim=-1) 

        return x

class FeatureEncoder(nn.Module):
    def __init__(self, args):
        super(FeatureEncoder, self).__init__()
        
        self.input_dims = args.input_dims
        self.hidden_dims = args.hidden_dims
        self.num_layers = args.layer_depth  
        def make_block():
            return nn.Sequential(
                nn.Linear(self.input_dims + 1, self.hidden_dims),
                nn.Tanh(),
                nn.Linear(self.hidden_dims, self.input_dims + 1)
            )
        self.feature_blocks = nn.ModuleList([make_block() for _ in range(self.num_layers)])
        self.feature_out = nn.Sequential(
            nn.Linear(self.input_dims + 1, self.hidden_dims),
            nn.Tanh(),
            nn.Linear(self.hidden_dims, self.hidden_dims),
        )

    def forward(self, feature):

        residual = feature.clone()

        for block in self.feature_blocks:
            feature = block(feature) + residual
        feature = self.feature_out(feature)

        return feature

# 傅里叶基函数
class FourierProjection(nn.Module):
    def __init__(self, args):
        super(FourierProjection, self).__init__()
        
        self.gridsize = args.modes // 2
        self.hidden_dims = args.hidden_dims
        self.num_layers = args.layer_depth 

        # 共享结构的构建函数
        def make_block():
            return nn.Sequential(
                nn.Linear(self.gridsize, self.hidden_dims),
                nn.Tanh(),
                nn.Linear(self.hidden_dims, self.gridsize)
            )
        
        # 使用 ModuleList 创建 sin 和 cos 的共享块
        self.sin_blocks = nn.ModuleList([make_block() for _ in range(self.num_layers)])
        self.cos_blocks = nn.ModuleList([make_block() for _ in range(self.num_layers)])

        # 输出层（分别处理 sin 和 cos 的输出）
        self.sin_out = nn.Sequential(
            nn.Linear(self.gridsize, self.hidden_dims),
            nn.Tanh(),
            nn.Linear(self.hidden_dims, self.hidden_dims),
        )
        self.cos_out = nn.Sequential(
            nn.Linear(self.gridsize, self.hidden_dims),
            nn.Tanh(),
            nn.Linear(self.hidden_dims, self.hidden_dims),
        )

        self.fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dims, self.hidden_dims),
            nn.Tanh(),
            nn.Linear(self.hidden_dims, self.hidden_dims),
        )

    def forward(self, x):
        '''
        傅里叶特征投影：将输入 x 映射到由前 K 个正弦/余弦函数构成的频域空间。
        输出：融合后的高维傅里叶特征表示。
        '''

        k = torch.arange(1, self.gridsize + 1, device=x.device).reshape(1, 1, self.gridsize)
        x_expanded = x.unsqueeze(-1)     # (batch_size, input_dim, 1)
        s = torch.sin(k * x_expanded)    # (B, D, gridsize)
        c = torch.cos(k * x_expanded)    # (B, D, gridsize)

        residual_s = s.clone()
        residual_c = c.clone()

        for block in self.sin_blocks:
            s = block(s) + residual_s
        s = self.sin_out(s)  
        for block in self.cos_blocks:
            c = block(c) + residual_c
        c = self.cos_out(c) 

        y = torch.cat((s, c), dim=-1)  

        y = self.fusion(y)

        return y

# Multi-Head Attention for easy understanding
class AttentionTransfer(nn.Module):
    def __init__(self, args):
        super(AttentionTransfer, self).__init__()
        
        self.seq_lens = args.sequence_len
        self.hidden_dims = args.hidden_dims
        self.numbers = args.num_buildings_source + args.num_buildings_target

        self.fc_q1 = nn.Linear(2*self.hidden_dims+1, self.hidden_dims)
        self.fc_q2 = nn.Linear(2*self.hidden_dims+1, self.hidden_dims)
        self.fc_q3 = nn.Linear(2*self.hidden_dims+1, self.hidden_dims)
        self.fc_q4 = nn.Linear(2*self.hidden_dims+1, self.hidden_dims)
        
        self.fc_k1 = nn.Linear(2*self.hidden_dims+1, self.hidden_dims)
        self.fc_k2 = nn.Linear(2*self.hidden_dims+1, self.hidden_dims)
        self.fc_k3 = nn.Linear(2*self.hidden_dims+1, self.hidden_dims)
        self.fc_k4 = nn.Linear(2*self.hidden_dims+1, self.hidden_dims)
        
        self.fc_v1 = nn.Linear(2*self.hidden_dims+1, self.hidden_dims//4)
        self.fc_v2 = nn.Linear(2*self.hidden_dims+1, self.hidden_dims//4)
        self.fc_v3 = nn.Linear(2*self.hidden_dims+1, self.hidden_dims//4)
        self.fc_v4 = nn.Linear(2*self.hidden_dims+1, self.hidden_dims//4)
        
    def forward(self, x):

        x_q1 = self.fc_q1(x.transpose(0,1))
        x_q2 = self.fc_q2(x.transpose(0,1))
        x_q3 = self.fc_q3(x.transpose(0,1))
        x_q4 = self.fc_q4(x.transpose(0,1))

        x_k1 = self.fc_k1(x.transpose(0,1))
        x_k2 = self.fc_k2(x.transpose(0,1))
        x_k3 = self.fc_k3(x.transpose(0,1))
        x_k4 = self.fc_k4(x.transpose(0,1))

        x_v1 = self.fc_v1(x.transpose(0,1))
        x_v2 = self.fc_v2(x.transpose(0,1))
        x_v3 = self.fc_v3(x.transpose(0,1))
        x_v4 = self.fc_v4(x.transpose(0,1))

        attention1 = torch.matmul(x_q1, x_k1.transpose(1,2)) / math.sqrt(self.hidden_dims)
        attention2 = torch.matmul(x_q2, x_k2.transpose(1,2)) / math.sqrt(self.hidden_dims)
        attention3 = torch.matmul(x_q3, x_k3.transpose(1,2)) / math.sqrt(self.hidden_dims)
        attention4 = torch.matmul(x_q4, x_k4.transpose(1,2)) / math.sqrt(self.hidden_dims)

        attention1 = F.softmax(attention1, dim=-1)
        attention2 = F.softmax(attention2, dim=-1)
        attention3 = F.softmax(attention3, dim=-1)
        attention4 = F.softmax(attention4, dim=-1)

        x1 = torch.matmul(attention1, x_v1)
        x2 = torch.matmul(attention2, x_v2)
        x3 = torch.matmul(attention3, x_v3)
        x4 = torch.matmul(attention4, x_v4)

        x1 = x1.permute(1, 2, 0)
        x2 = x2.permute(1, 2, 0)
        x3 = x3.permute(1, 2, 0)
        x4 = x4.permute(1, 2, 0)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class TemporalAlignment(nn.Module):
    def __init__(self, args):
        super(TemporalAlignment, self).__init__()
        
        self.seq_lens = args.sequence_len
        self.hidden_dims = args.hidden_dims
        self.numbers = args.num_buildings_source + args.num_buildings_target

        self.Conv1d_3 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, 
                    kernel_size=3, padding=1, padding_mode='zeros'),
            nn.Tanh(),
            nn.Conv1d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, 
                    kernel_size=3, padding=1, padding_mode='zeros'),
            )
        
        self.Conv1d_5 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, 
                    kernel_size=5, padding=2, padding_mode='zeros'),
            nn.Tanh(),
            nn.Conv1d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, 
                    kernel_size=5, padding=2, padding_mode='zeros'),
            )
        
        self.Conv1d_7 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, 
                    kernel_size=7, padding=3, padding_mode='zeros'),
            nn.Tanh(),
            nn.Conv1d(in_channels=self.hidden_dims, out_channels=self.hidden_dims, 
                    kernel_size=7, padding=3, padding_mode='zeros'),
            )
        
    def forward(self, x):

        # 外部建筑环境特征与用水模式特征同时偏移
        x_3 = self.Conv1d_3(x)
        x_5 = self.Conv1d_5(x)
        x_7 = self.Conv1d_7(x)

        x = torch.cat((x, x_3, x_5, x_7), dim=1)

        return x.transpose(1,2) 


# 特征引导用水特征迁移
class FeatMigrator(nn.Module):
    def __init__(self, args):
        super(FeatMigrator, self).__init__()

        self.input_dims = args.input_dims
        self.hidden_dims = args.hidden_dims
        self.seq_len = args.sequence_len

        self.modes = args.modes

        # 位置编码
        self.Position_Embedding = PositionalEncoding(args)

        # 外部建筑环境特征
        self.Feature = FeatureEncoder(args)

        # Fourier KAN
        self.FP = FourierProjection(args)

        # 特征融合
        self.Transfer = AttentionTransfer(args) 

        # 特征对齐
        self.Alignment = TemporalAlignment(args)

    def forward(self, Features, Patterns, mask):

        Features = torch.cat([Features, mask], dim=-1)  

        # 特征自注意力层(迁移)
        Features = self.Feature(Features)      
        Features = self.Position_Embedding(Features)  
        # print("FeatMigrator —— Features:", Features.shape)
        
        # FourierProjection
        Time = self.FP(Patterns.squeeze(-1)) 
        # print("FeatMigrator —— Time:", Time.shape)

        # 基于注意力机制的特征迁移
        out = torch.cat([Features, Time], dim=-1)  
        # print("FeatMigrator —— out:", out.shape)

        # 特征融合
        out = self.Transfer(out) 
        # print("FeatMigrator —— out:", out.shape)

        # 时间对齐
        out = self.Alignment(out)
        # print("FeatMigrator —— out:", out.shape)

        return out 