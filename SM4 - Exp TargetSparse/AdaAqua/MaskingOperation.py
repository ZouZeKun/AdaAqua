import random
import torch

def apply_mask(args, labels, mask_ratio):
    # 创建掩码
    mask_ = torch.zeros(labels.shape)

    # 随机掩码(增强泛化能力-->计算损失)
    num_masked = int(mask_ratio * labels.shape[0])
    # 随机从labels.shape[0] 中选择num_masked个索引
    masked_indices = random.sample(range(labels.shape[0]), num_masked)
    # 第一维度进行随机掩码
    mask_[masked_indices] = 1
    if args.fold != 2:
        fold_from = args.fold * 7
        fold_to = args.fold * 7 + 14
        mask_[fold_from:fold_to] = 1 
    else:
        mask_[0:7] = 1
        mask_[14:28] = 1

    mask_ = torch.tensor(mask_).to(args.device)
    
    return mask_