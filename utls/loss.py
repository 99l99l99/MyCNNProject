import torch
import torch.nn as nn


class L1_loss(nn.Module):
    def __init__(self):
        super(L1_loss, self).__init__()


    def forward(self, prediction, label):
        return torch.mean(torch.abs(prediction - label))

class L2_loss(nn.Module):
    def __init__(self):
        super(L2_loss, self).__init__()


    def forward(self, prediction, label):
        return torch.mean(torch.pow(prediction - label, 2)) * 1000


class HeatMapLoss(nn.Module):
    def __init__(self, cfg_loss):
        super(HeatMapLoss, self).__init__()
        self.cfg = cfg_loss
        self.alpha = cfg_loss['alpha']
        self.bata = cfg_loss['beta']
    def forward(self, pred_heatmap, label_heatmap, pred_offsets, label_offsets):
        part1 = (1 - pred_heatmap)**self.alpha * torch.log(pred_heatmap + 1e-8)
        part2 = (1 - label_heatmap)**self.bata * pred_heatmap**self.alpha * torch.log(1 - pred_heatmap + 1e-8)
        mask = label_heatmap == 1
        loss_heatmap = - (part1 * mask + part2 * (~mask)).sum() / mask.sum()
        
        loss_offsets = (torch.abs(pred_offsets - label_offsets) * mask).sum() / mask.sum()
        
        details = {}
        details["loss_heatmap"] = loss_heatmap
        details["loss_offsets"] = loss_offsets
        
        return loss_heatmap + loss_offsets * self.cfg['lambda_offsets'], details


class RegressLoss(nn.Module):
    def __init__(self, cfg_loss, cfg_model):
        super(RegressLoss, self).__init__()
        self.cfg_loss = cfg_loss
        self.ratio = cfg_model["ratio"]
        self.outshape = cfg_model["out_shape"]
        self.num_class = cfg_model["num_class"]
        
    def forward(self, prediction, label):
        
        p_prob, p_cls, p_xy = torch.split(prediction, [1,self.num_class,2], dim=1)
        t_prob, t_cls, t_xy = torch.split(label, [1,self.num_class,2], dim=1)
        
        if self.cfg_loss['focal']['enable']:
            prob_loss = - self.cfg_loss['focal']['alpha'] * t_prob * (1-p_prob)**self.cfg_loss['focal']['gamma'] * torch.log(p_prob+1e-8) \
                            - (1 - self.cfg_loss['focal']['alpha']) * (1-t_prob) * p_prob**self.cfg_loss['focal']['gamma'] * torch.log(1-p_prob+1e-8)
            cls_loss = - self.cfg_loss['focal']['alpha'] * t_cls * (1-p_cls)**self.cfg_loss['focal']['gamma'] * torch.log(p_cls+1e-8) \
                            - (1 - self.cfg_loss['focal']['alpha']) * (1-t_cls) * p_cls**self.cfg_loss['focal']['gamma'] * torch.log(1-p_cls+1e-8)
        else:
            prob_loss = - t_prob * (1-p_prob) * torch.log(p_prob+1e-8) \
                           - (1-t_prob) * p_prob * torch.log(1-p_prob+1e-8)
            cls_loss = - t_cls * (1-p_cls) * torch.log(p_cls+1e-8) \
                           - (1-t_cls) * p_cls * torch.log(1-p_cls+1e-8)
        
        distance_loss = torch.sum((p_xy - t_xy)**2 / (self.ratio / 2)**2, dim = 1).unsqueeze(dim=1)
        
        
        ignore_mask_list = []
        
        for b in range(p_xy.shape[0]):
            pcoord = p_xy[b].permute(1,2,0)
            tcoord = t_xy[b]
            
            mask = tcoord.max(dim=0)[0] > 0.01
            tcoord = tcoord.transpose(2,0)[mask]
            num_coord = tcoord.shape[0]

            if num_coord > 0:
                distance = self._distance(pcoord, tcoord)
                mask = torch.min(distance, -1, keepdim=True)[0]
                mask = torch.ge(mask, self.cfg_loss['ignore_threshold'])
            else:
                mask = torch.ones([label.shape[-2],label.shape[-1],1], dtype=torch.float, device=prediction.device)

            ignore_mask_list.append(mask)

        ignore_mask = torch.stack(ignore_mask_list, 0)
        ignore_mask = ignore_mask.permute(0,3,1,2)
        
        with torch.no_grad():
            positive_sum = torch.gt(t_prob, 0.5).float().sum().clamp(min=1)
            
        loss_coord = (distance_loss * t_prob).sum() / positive_sum
        loss_prob_obj = (prob_loss * t_prob).sum() / positive_sum
        loss_prob_noobj = (prob_loss * (1 - t_prob) * ignore_mask).mean()
        loss_class = (cls_loss * t_prob).sum() / positive_sum + (cls_loss * (1-t_prob)).mean() / 2
        
        loss_coord = loss_coord * self.cfg_loss["coord"]
        loss_prob_obj = loss_prob_obj * self.cfg_loss["prob_obj"]
        loss_prob_noobj = loss_prob_noobj * self.cfg_loss["prob_noobj"]
        loss_class = loss_class * self.cfg_loss["cls"]
        loss = loss_coord \
                + loss_prob_obj \
                + loss_prob_noobj \
                + loss_class

        details = {}
        details["loss_coord"] = loss_coord
        details["loss_prob_obj"] = loss_prob_obj
        details["loss_prob_noobj"] = loss_prob_noobj
        details["cls"] = loss_class
        
        return loss, details

    def _distance(self, pcoord, tcoord):
        distance = pcoord.unsqueeze(axis = -1) - tcoord.transpose(0,1)
        
        return torch.norm(distance,dim = 2)