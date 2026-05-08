import time, os
from tqdm import tqdm

import torch
import torch.optim as optim

from AdaAqua.MaskingOperation import apply_mask
from AdaAqua.Encoder import FeatMigrator
from AdaAqua.Regressor import Regressor
from AdaAqua.LossFunciton import RegressLoss


def train(args, IO,  train_loader, val_loader, output_dir):
    best_val_loss = float('inf')
    patience = args.patience  
    patience_counter = 0

    # 使用GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.manual_seed(args.seed)  

    # 加载三个模型及参数量统计
    PF_extractor = FeatMigrator(args).to(device)
    regressor = Regressor(args).to(device)

    IO.cprint(str(PF_extractor))
    PF_extractor_params = sum(p.numel() for p in PF_extractor.parameters() if p.requires_grad)
    IO.cprint(str(regressor))
    Regressor_params = sum(p.numel() for p in regressor.parameters() if p.requires_grad)
    IO.cprint('PF_extractor Model Parameter: {}'.format(PF_extractor_params))
    IO.cprint('Regressor Model Parameter: {}'.format(Regressor_params))

    # 优化器 Adam
    IO.cprint('Using AdamW')
    optimizer_fe = optim.AdamW(PF_extractor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_rg = optim.AdamW(regressor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler_fe = optim.lr_scheduler.StepLR(optimizer_fe,
                                            step_size=args.decay_epoch,
                                            gamma=args.gamma)
    scheduler_rg = optim.lr_scheduler.StepLR(optimizer_rg,
                                            step_size=args.decay_epoch,
                                            gamma=args.gamma)

    # 损失函数
    criterion_reg = RegressLoss(args) 

    # 保存损失
    train_loss_list = []
    val_loss_regress_list = []

    start_time = time.time()

    for epoch in range(args.num_epochs):
        #################
        ###   Train   ###
        #################
        train_i = 0
        
        PF_extractor.train()  
        regressor.train()

        train_loss = 0.0

        mask_ratio = 0.5

        for batch_idx, data in enumerate(tqdm(train_loader, total=int(96*28 - 95), desc="Train_Loader")):
            train_i += 1

            (x, y) = data
            x, y = x.squeeze(0), y.squeeze(0)

            # 测试数据：防止信息泄露
            fold_from = args.fold * 7
            fold_to = args.fold * 7 + 7
            y[fold_from:fold_to, :, :] = 0

            # 针对(目标域)回归值进行掩码(1:计算损失，0:计算特征)
            mask_target = apply_mask(args, y, mask_ratio) 

            m = y * (1-mask_target)

            # 一起输入提取特征！
            feat_all = PF_extractor(x, m, mask_target) 

            # 回归器根据提取的特征进行预测 (199+24,96)
            pred_all = regressor(feat_all)
            # 计算损失(回归损失)(仅对掩码位置)
            y = y.squeeze(-1)

            loss_regress = criterion_reg(pred_all, y)
            loss_regress = loss_regress * 21/14

            loss_regress.backward()
            
            optimizer_fe.step()
            optimizer_rg.step()
            optimizer_fe.zero_grad()
            optimizer_rg.zero_grad()
            
            train_loss += loss_regress.item()

        scheduler_fe.step()
        scheduler_rg.step()

        # 求均值
        avg_train_loss = train_loss / train_i

        IO.cprint('Train Epoch #{:03d}, Loss Reg: {:.2f}'.format(epoch, avg_train_loss))

        # 绘图保存
        train_loss_list.append(avg_train_loss)

        #################
        ###   Valid   ###
        #################
        PF_extractor.eval()  
        regressor.eval()
        val_loss_regress_ = 0.0
        val_i = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(val_loader, total=int(96*14 - 95), desc="Val_Loader")):
                val_i += 1

                (x, y) = data
                x, y = x.squeeze(0), y.squeeze(0)

                # 测试数据：防止信息泄露
                fold_from = args.fold * 8
                fold_to = args.fold * 8 + 8
                y[fold_from:fold_to, :, :] = 0

                # 测试数据：防止信息泄露
                fold_from = args.fold * 8
                fold_to = args.fold * 8 + 8
                y[fold_from:fold_to, :, :] = 0

                # 针对(目标域)回归值进行掩码(1:计算损失，0:计算特征)
                mask_target = apply_mask(args, y, mask_ratio) 

                m = y * (1-mask_target)

                # 一起输入提取特征！
                feat_all = PF_extractor(x, m, mask_target) 

                # 回归器根据提取的特征进行预测 (199+24,96)
                pred_all = regressor(feat_all)

                # 计算损失(回归损失)(仅对掩码位置)
                y = y.squeeze(-1)

                loss_regress = criterion_reg(pred_all, y)
                loss_regress = loss_regress * 21/14

                val_loss_regress = loss_regress.item() 
                val_loss_regress_ += val_loss_regress

        avg_val_loss_regress = (val_loss_regress_) / val_i

        # 绘图保存
        val_loss_regress_list.append(avg_val_loss_regress)
        
        IO.cprint('Val Epoch #{:03d}, Loss Reg: {:.2f}'.format(epoch, avg_val_loss_regress))
        
        # 选取最佳参数
        if avg_val_loss_regress < best_val_loss:
            best_val_loss = avg_val_loss_regress
            patience_counter = 0
            # 保存当前最佳模型
            best_model_wts_1 = PF_extractor.state_dict()
            best_model_wts_2 = regressor.state_dict()
        else:
            patience_counter += 1

        # 检查是否达到提前停止条件
        if patience_counter >= patience:
            IO.cprint('Early stopping triggered. Best Val_Loss: {:.4f}'.format(best_val_loss))
            break
        
        PF_extractor.load_state_dict(best_model_wts_1)
        regressor.load_state_dict(best_model_wts_2)

    torch.save(PF_extractor, os.path.join(output_dir, 'PF_extractor.pth'))
    torch.save(regressor, os.path.join(output_dir, 'regressor.pth'))

    IO.cprint('The current best model is saved in: {}'.format('******** outputs/%s/model.pth *********' % args.exp_name))

    end_time = time.time()
    IO.cprint('Total training time: {:.4f}s'.format(end_time - start_time))

    return (train_loss_list, val_loss_regress_list)