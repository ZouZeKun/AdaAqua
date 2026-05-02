import os
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch

def test(args, IO, test_loader, output_dir):
    """测试模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 输出内容保存在之前的训练日志里
    IO.cprint('')
    IO.cprint('********** TEST START **********')
    IO.cprint('Reload Best Model')
    IO.cprint('The current best model is saved in: {}'.format('******** outputs/%s/PF_extractor.pth *********' % args.exp_name))
    IO.cprint('The current best model is saved in: {}'.format('******** outputs/%s/regressor.pth *********' % args.exp_name))

    PF_extractor = torch.load(os.path.join(output_dir, 'PF_extractor.pth')).to(device)
    regressor    = torch.load(os.path.join(output_dir, 'regressor.pth')).to(device)

    PF_extractor = PF_extractor.eval() 
    regressor = regressor.eval()

    ##############################
    ### Test For Generalization ##
    ##############################

    true_svae = []
    pred_svae = []

    for batch_idx, data in enumerate(tqdm(test_loader, total=int(96*8 - 95), desc="Test_Loader")):

        (x, y) = data
        x, y = x.squeeze(0), y.squeeze(0)

        # 获取账单信息
        ZB_Bill = pd.read_csv(args.target_average_billing, header=0, index_col=None)
        ZB_Bill = torch.tensor(ZB_Bill.values, dtype=torch.float)
        ZB_Bill = ZB_Bill.to(args.device)

        # 保留8个测试位置验证模型性能
        fold_from = args.fold * 8
        fold_to = args.fold * 8 + 8
        y_test = y[fold_from:fold_to, :, :].squeeze(-1) + ZB_Bill[fold_from:fold_to, :] 
        y_test = torch.abs(y_test)  
        y_test = y_test.detach().cpu().numpy() 

        # 目标域测试区域用水量真值置0  防止信息泄露
        y[fold_from:fold_to, :, :] = 0

        mask_target_ = torch.zeros_like(y)
        mask_target_[fold_from:fold_to, :, :] = 1

        feat_all = PF_extractor(x, y, mask_target_) 

        pred_all = regressor(feat_all)
        pred_all = pred_all[fold_from:fold_to, :] + ZB_Bill[fold_from:fold_to, :]
        pred_all = torch.abs(pred_all)
        pred_all = pred_all.detach().cpu().numpy()
        
        true_svae.append(y_test)
        pred_svae.append(pred_all)

    true_svae = np.concatenate(true_svae, axis=1) 
    pred_svae = np.concatenate(pred_svae, axis=1)

    return true_svae, pred_svae