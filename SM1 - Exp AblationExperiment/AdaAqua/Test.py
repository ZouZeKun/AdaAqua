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

    for batch_idx, data in enumerate(tqdm(test_loader, total=int(96*8*3 - 95*3), desc="Test_Loader")):

        # 加载数据(x为特征；y为预测值; z为源域1/目标域2标签)(1为源域，2为目标域)
        (x1, y1, z1, x2, y2, z2) = data
        x1, y1, z1, x2, y2, z2 = x1.squeeze(0), y1.squeeze(0), z1.squeeze(0), x2.squeeze(0), y2.squeeze(0), z2.squeeze(0)

        # 获取账单信息
        ZZ_Bill = pd.read_csv(args.source_average_billing, header = 0, index_col=None)
        ZB_Bill = pd.read_csv(args.target_average_billing, header = 0, index_col=None)
        ZZ_Bill = torch.tensor(ZZ_Bill.values, dtype=torch.float)
        ZB_Bill = torch.tensor(ZB_Bill.values, dtype=torch.float)
        ZZ_Bill = ZZ_Bill.to(args.device)
        ZB_Bill = ZB_Bill.to(args.device)
        Mean_Tensor = torch.cat((ZZ_Bill, ZB_Bill), dim=0)

        # 保留8个测试位置验证模型性能
        fold_from = args.fold * 7
        fold_to = args.fold * 7 + 7
        y_test = y2[fold_from:fold_to, :, :].squeeze(-1) + Mean_Tensor[(199+fold_from):(199+fold_to), :] 
        y_test = torch.abs(y_test)  
        y_test = y_test.detach().cpu().numpy() 

        # 目标域测试区域用水量真值置0  防止信息泄露
        y2[fold_from:fold_to, :, :] = 0

        y_all = torch.cat((y1, y2), dim=0)

        # 分别提取源域和目标域的特征(1:计算损失，0:计算特征)
        mask_source_ = torch.zeros_like(y1)
        mask_target_ = torch.zeros_like(y2)
        mask_target_[fold_from:fold_to, :, :] = 1
        mask_all = torch.cat((mask_source_, mask_target_), dim=0)

        x_all = torch.cat((x1, x2), dim=0)
        feat_all = PF_extractor(x_all, y_all, mask_all) 

        pred_all = regressor(feat_all)
        pred_all = pred_all[(199+fold_from):(199+fold_to), :] + Mean_Tensor[(199+fold_from):(199+fold_to), :]
        pred_all = torch.abs(pred_all)
        pred_all = pred_all.detach().cpu().numpy()
        
        true_svae.append(y_test)
        pred_svae.append(pred_all)
        # print("true_svae: ", true_svae)

    true_svae = np.concatenate(true_svae, axis=1) 
    pred_svae = np.concatenate(pred_svae, axis=1)

    return true_svae, pred_svae