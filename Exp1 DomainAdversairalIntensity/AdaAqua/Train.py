import time, os
from tqdm import tqdm

import torch
import torch.optim as optim

from AdaAqua.MaskingOperation import apply_mask
from AdaAqua.Encoder import FeatMigrator
from AdaAqua.Regressor import Regressor
from AdaAqua.DomainDiscriminator import DomainDiscriminator, gradient_reversal
from AdaAqua.LossFunciton import RegressLoss, DomainLoss


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
    domain_discriminator = DomainDiscriminator(args).to(device)

    IO.cprint(str(PF_extractor))
    PF_extractor_params = sum(p.numel() for p in PF_extractor.parameters() if p.requires_grad)
    IO.cprint(str(regressor))
    Regressor_params = sum(p.numel() for p in regressor.parameters() if p.requires_grad)
    IO.cprint(str(domain_discriminator))
    Domain_params = sum(p.numel() for p in domain_discriminator.parameters() if p.requires_grad)
    IO.cprint('PF_extractor Model Parameter: {}'.format(PF_extractor_params))
    IO.cprint('Regressor Model Parameter: {}'.format(Regressor_params))
    IO.cprint('Domain Discriminator Model Parameter: {}'.format(Domain_params))

    # 优化器 Adam
    IO.cprint('Using AdamW')
    optimizer_fe = optim.AdamW(PF_extractor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_rg = optim.AdamW(regressor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_dc = optim.AdamW(domain_discriminator.parameters(), lr=args.learning_rate/5, weight_decay=args.weight_decay)

    scheduler_fe = optim.lr_scheduler.StepLR(optimizer_fe,
                                            step_size=args.decay_epoch,
                                            gamma=args.gamma)
    scheduler_rg = optim.lr_scheduler.StepLR(optimizer_rg,
                                            step_size=args.decay_epoch,
                                            gamma=args.gamma)
    scheduler_dc = optim.lr_scheduler.StepLR(optimizer_dc,
                                            step_size=args.decay_epoch,
                                            gamma=args.gamma)

    # 损失函数
    criterion_reg = RegressLoss(args)  
    criterion_dom = DomainLoss(args)

    # 保存损失
    train_loss1_list = []
    train_loss2_list = []
    val_loss_regress_list = []

    start_time = time.time()

    for epoch in range(args.num_epochs):
        #################
        ###   Train   ###
        #################
        train_i = 0
        
        PF_extractor.train()  
        regressor.train()
        domain_discriminator.train()

        train_loss1 = 0.0
        train_loss2 = 0.0

        mask_ratio = 0.5

        for batch_idx, data in enumerate(tqdm(train_loader, total=int(96*28*3 - 95*3), desc="Train_Loader")):
            train_i += 1
            # 加载数据(x为特征；y为预测值; z为源域1/目标域2标签)(1为源域，2为目标域)
            (x1, y1, z1, x2, y2, z2) = data
            x1, y1, z1, x2, y2, z2 = x1.squeeze(0), y1.squeeze(0), z1.squeeze(0), x2.squeeze(0), y2.squeeze(0), z2.squeeze(0)       

            # 测试数据：防止信息泄露
            fold_from = args.fold * 7
            fold_to = args.fold * 7 + 7
            y2[fold_from:fold_to, :, :] = 0

            # 针对(目标域)回归值进行掩码(1:计算损失，0:计算特征)
            mask_source_ = torch.zeros_like(y1)
            mask_target_ = apply_mask(args, y2, mask_ratio) 
            mask_all = torch.cat((mask_source_, mask_target_), dim=0)

            z_all = torch.cat((z1, z2), dim=0)
            # 计算损失的位置
            y_all = torch.cat((y1, y2), dim=0)
            # 计算(已知用水量)特征的位置(除了目标域的位置)
            m_all = torch.cat((y1, y2), dim=0) * (1-mask_all)
            # 全部位置的(建筑环境特征)都假定为已知
            x_all = torch.cat((x1, x2), dim=0)
            # 一起输入提取特征！
            feat_all = PF_extractor(x_all, m_all, mask_all) 

            # 回归器根据提取的特征进行预测
            pred_all = regressor(feat_all)
            # 计算损失(回归损失)(仅对掩码位置)
            y_all = y_all.squeeze(-1)
            # 回归损失(只对非NaN掩码部分)
            mask_all = mask_all.squeeze(-1)
            loss_regress = criterion_reg(pred_all, y_all)
            loss_regress = loss_regress * 223/215

            # 域分类损失(领域对抗损失)(对全部位置)
            feat_grl = gradient_reversal(feat_all, args.alpha)  # alpha 控制反转强度
            sort_all = domain_discriminator(feat_grl)
            loss_domain_adv = criterion_dom(sort_all, z_all)

            loss_total = loss_regress + loss_domain_adv
            loss_total.backward()
            
            optimizer_fe.step()
            optimizer_rg.step()
            optimizer_dc.step()
            optimizer_fe.zero_grad()
            optimizer_rg.zero_grad()
            optimizer_dc.zero_grad()
            
            train_loss1 += loss_regress.item()
            train_loss2 += loss_domain_adv.item()

        scheduler_fe.step()
        scheduler_rg.step()
        scheduler_dc.step()

        # 求均值
        avg_train_loss1 = train_loss1 / train_i
        avg_train_loss2 = train_loss2 / train_i

        IO.cprint('Train Epoch #{:03d}, Loss Reg: {:.2f}, Loss Ada: {:.2f}'.format(
                    epoch, avg_train_loss1, avg_train_loss2
                    ))

        # 绘图保存
        train_loss1_list.append(avg_train_loss1)
        train_loss2_list.append(avg_train_loss2)

        #################
        ###   Valid   ###
        #################
        PF_extractor.eval()  
        regressor.eval()
        domain_discriminator.eval()
        val_loss_regress_ = 0.0
        val_i = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(val_loader, total=int(96*14*3 - 95*3), desc="Val_Loader")):
                val_i += 1
                # 加载数据(x为特征；y为预测值; z为源域1/目标域2标签)(1为源域，2为目标域)
                (x1, y1, z1, x2, y2, z2) = data
                x1, y1, z1, x2, y2, z2 = x1.squeeze(0), y1.squeeze(0), z1.squeeze(0), x2.squeeze(0), y2.squeeze(0), z2.squeeze(0)       

                # 测试数据：防止信息泄露
                fold_from = args.fold * 7
                fold_to = args.fold * 7 + 7
                y2[fold_from:fold_to, :, :] = 0

                # 针对(目标域)回归值进行掩码
                mask_source_ = torch.zeros_like(y1)
                mask_target_ = apply_mask(args, y2, mask_ratio)
                mask_all = torch.cat((mask_source_, mask_target_), dim=0)

                z_all = torch.cat((z1, z2), dim=0) 
                # 计算损失的位置
                y_all = torch.cat((y1, y2), dim=0) * (mask_all) 
                # 计算特征的位置
                m_all = torch.cat((y1, y2), dim=0) * (1-mask_all)
                # 全部位置的(建筑环境特征)都假定为已知
                x_all = torch.cat((x1, x2), dim=0)
                # 一起输入提取特征！
                feat_all = PF_extractor(x_all, m_all, mask_all) 

                # 回归器根据提取的特征进行预测 
                pred_all = regressor(feat_all)
                # 计算损失(回归损失)(仅对掩码位置)
                y_all = y_all.squeeze(-1)
                # 回归损失(只对非NaN掩码部分)
                # 只对非NaN掩码部分计算回归损失
                mask_all = mask_all.squeeze(-1)
                loss_regress = criterion_reg(mask_all*pred_all, mask_all*y_all)
                loss_regress = loss_regress * (mask_all.numel() / mask_all.sum().item())
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
            best_model_wts_3 = domain_discriminator.state_dict()    
        else:
            patience_counter += 1

        # 检查是否达到提前停止条件
        if patience_counter >= patience:
            IO.cprint('Early stopping triggered. Best Val_Loss: {:.4f}'.format(best_val_loss))
            break
        
        PF_extractor.load_state_dict(best_model_wts_1)
        regressor.load_state_dict(best_model_wts_2)
        domain_discriminator.load_state_dict(best_model_wts_3)

    torch.save(PF_extractor, os.path.join(output_dir, 'PF_extractor.pth'))
    torch.save(regressor, os.path.join(output_dir, 'regressor.pth'))
    torch.save(domain_discriminator, os.path.join(output_dir, 'domain_discriminator.pth')) 

    IO.cprint('The current best model is saved in: {}'.format('******** outputs/%s/model.pth *********' % args.exp_name))

    end_time = time.time()
    IO.cprint('Total training time: {:.4f}s'.format(end_time - start_time))

    return (train_loss1_list, train_loss2_list, val_loss_regress_list)