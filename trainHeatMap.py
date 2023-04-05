import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from datasets.BaseDataGen import BaseDataset
from datasets.HeatMapDataGen import HeatMapDataset
from datasets.HeatMapValDataGen import HeatMapValDataGen
from model.heatMapBasedNet import HeatMapBasedNet
from utls.loss import L1_loss, L2_loss, HeatMapLoss
from utls.valLoss import HeatMapValLoss
import os

if __name__ == "__main__":
    # 加载cfg文件
    with open("cfg/trainHeatMap.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cfg_train = config['train']
    if not os.path.exists("save/"+cfg_train['save_pkg']):
        os.makedirs("save/"+ cfg_train['save_pkg'])

    # 创建数据集对象
    train_dataset = HeatMapDataset(config['data'], config['model'])
    train_loader = DataLoader(train_dataset, batch_size=cfg_train['batch_size'], shuffle=True, num_workers = cfg_train['num_workers'])

    cfg_val = config['val']
    val_dataset = HeatMapValDataGen(config['val'], config["model"])
    val_loader = DataLoader(val_dataset, batch_size=cfg_val['batch_size'], shuffle=False, num_workers = 1)

    if config['train']['wandb']:
        import wandb
        wandb.init(
            project = 'UAVKeypointHeatMap',
            config = config
    	)

    # 定义模型和优化器
    model = HeatMapBasedNet(config["model"])
    
    if config['train']['wandb']:
        wandb.watch(model)

    optimizer = torch.optim.AdamW(model.parameters(), cfg_train['learning_rate'], weight_decay=cfg_train['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg_train['cosine_annealing_T'], eta_min=cfg_train['minimal_learning_rate'])
    # 定义损失函数
    myLoss = HeatMapLoss(config['loss'])
    valLoss = HeatMapValLoss(config["val"], config["loss"])
    # print(model.cheakDevice())
    # 训练循环
    max_recall = 0
    max_acc = 0
    for epoch in range(cfg_train['num_epochs']):
        # 模型训练
        model.train()
        train_loss = 0.0
        mean_details = {}
        mean_details["loss_heatmap"] = 0
        mean_details["loss_offsets"] = 0
        for batch, (inputs, label_heatmap, label_offsets) in enumerate(train_loader):
            # 将输入和标签张量移动到GPU上
            inputs, label_heatmap, label_offsets = inputs.to(model.device), label_heatmap.to(model.device), label_offsets.to(model.device)
            optimizer.zero_grad()
            pred_heatmap, pred_offsets = model(inputs)
            loss, details = myLoss(pred_heatmap, label_heatmap, pred_offsets, label_offsets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            for (k, v) in details.items():
                mean_details[k] += v * inputs.size(0)
            # 打印训练结果
            print("epoch %d/%d batch %d loss %.4f heatmap %.4f offsets %.4f"%(epoch,cfg_train['num_epochs'], batch, loss.item(), details["loss_heatmap"], details["loss_offsets"]))
        train_loss /= len(train_loader.dataset)
        for k in mean_details:
            mean_details[k] /= len(train_loader.dataset)
        if config['train']['wandb']:
            wandb.log({'mean_loss': train_loss}, step=epoch)
            wandb.log(mean_details, step=epoch)
        
        # 模型评估
        model.eval()
        num_matches_total = 0
        num_predict_total = 0
        num_label_total = 0
        error_total = 0
        with torch.no_grad():
            for inputs, points_label in val_loader:
                # 将输入和标签张量移动到GPU上
                inputs= inputs.to(model.device)
                pred_heatmap, pred_offsets = model(inputs)
                num_matches, num_predict, num_label, error = valLoss.calc_acc_recall(pred_heatmap.to('cpu'), pred_offsets.to('cpu'), points_label)
                num_matches_total += num_matches
                num_predict_total += num_predict
                num_label_total += num_label
                error_total += error
                # print([num_matches, num_predict, num_label, error])
        acc = num_matches_total / (num_predict_total + 1e-10)
        recall = num_matches_total / (num_label_total + 1e-10)
        avg_error = error_total / (num_matches_total + 1e-10)
        if config['train']['wandb']:
            wandb.log({'val_acc': acc}, step=epoch)
            wandb.log({'val_recall': recall}, step=epoch)
            wandb.log({'val_avg_error': avg_error}, step=epoch)
        
        if epoch >= config['train']['skip_scheduler_epoch']:
            scheduler.step()
        # 打印训练和验证结果
        print("epoch %d/%d loss %.4f acc %.4f racall %.4f error %.4f"%(epoch+1,cfg_train['num_epochs'], train_loss, acc, recall, avg_error))
        # 在每10个 epoch 完成后保存模型
        torch.save(model.state_dict(), "save/"+cfg_train['save_pkg'] + "/model.pth")
        if acc > max_acc or recall > max_recall:
            torch.save(model.state_dict(), "save/"+cfg_train['save_pkg'] + "/model_epoch%d.pth"%(epoch))
        max_recall = max(recall, max_recall)
        max_acc = max(acc, max_acc)
     