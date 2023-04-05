from utls.loss import RegressLoss, HeatMapLoss
import numpy as np
from utls.utls import match_coordinates, map_nms
class RegressValLoss(RegressLoss):
    def __init__(self, cfg_val, cfg_loss, cfg_model):
        super(RegressValLoss, self).__init__(cfg_loss, cfg_model)
        self.prob_threshold = cfg_val['prob_threshold']
        self.distance_threshold = cfg_val['distance_threshold']
    
    def calc_acc_recall(self, output, points_label):
        num_matches_total = 0
        error_total = 0
        num_label_total = 0
        num_predict_total = 0
        output = np.array(output[0])
        points_label = np.array(points_label[0])
        
        # 处理label
        points_label = np.int0(points_label)
        label_cls1 = points_label[points_label[:,0] == 0]
        label_cls1 = label_cls1[:,1:]
        label_cls2 = points_label[points_label[:,0] == 1]
        label_cls2 = label_cls2[:,1:]
        # 处理output
        mask_prob = output[0,...] > self.prob_threshold
        mask_cls1 = output[1,...] > output[2,...] + 0.3
        mask_cls2 = output[2,...] > output[1,...] + 0.3
        output_cls1 = output[:,mask_prob * mask_cls1]
        output_cls2 = output[:,mask_prob * mask_cls2]
        prob = output[0, mask_prob]
        output_cls1 = output_cls1[3:]
        output_cls1 = np.int0(output_cls1).T
        output_cls2 = output_cls2[3:]
        output_cls2 = np.int0(output_cls2).T
        
        
        #根据label_cls1，label_cls2，output_cls1, output_cls2计算准确率，召回率，平均距离
        num_matches, avg_error = match_coordinates(label_cls1, output_cls1, self.distance_threshold)
        num_matches_total += num_matches
        error_total += num_matches * avg_error
        num_label_total += label_cls1.shape[0]
        num_predict_total += output_cls1.shape[0]
        
        num_matches, avg_error = match_coordinates(label_cls2, output_cls2, self.distance_threshold)
        num_matches_total += num_matches
        error_total += num_matches * avg_error
        num_label_total += label_cls2.shape[0]
        num_predict_total += output_cls2.shape[0]
        
        # acc = num_matches_total / num_predict_total
        # recall = num_matches_total / num_label_total
        # avg_error = error_total / num_matches_total
        
        return num_matches_total, num_predict_total, num_label_total, error_total
    
class HeatMapValLoss(HeatMapLoss):
    def __init__(self, cfg_val, cfg_loss):
        super(HeatMapValLoss, self).__init__(cfg_loss)
        self.suppression_sigma = cfg_val['suppression_sigma']
        self.prob_threshold = cfg_val['prob_threshold']
        self.max_num = cfg_val['max_num']
        self.distance_threshold = cfg_val['distance_threshold']
        self.map_downsample = cfg_val['map_downsample']
    def calc_acc_recall(self, pred_heatmap, pred_offsets, points_label):
        num_matches_total = 0
        error_total = 0
        num_label_total = 0
        num_predict_total = 0
        pred_heatmap = np.array(pred_heatmap[0])
        pred_offsets = np.array(pred_offsets[0])
        points_label = np.array(points_label[0])
        
        # 处理label
        points_label = np.int0(points_label)
        label_cls1 = points_label[points_label[:,0] == 0]
        label_cls1 = label_cls1[:,1:]
        label_cls2 = points_label[points_label[:,0] == 1]
        label_cls2 = label_cls2[:,1:]
        
        # 处理output
        map_cls1 = pred_heatmap[0]
        map_cls2 = pred_heatmap[1]
        keypoints1 = map_nms(map_cls1, self.suppression_sigma, self.prob_threshold, self.max_num)
        keypoints2 = map_nms(map_cls2, self.suppression_sigma, self.prob_threshold, self.max_num)
        if len(keypoints1) > 0:
            keypoints1 = np.array(keypoints1)
        else:
            keypoints1 = np.zeros((0,2), np.int)
        if len(keypoints2) > 0:
            keypoints2 = np.array(keypoints2)
        else:
            keypoints2 = np.zeros((0,2), np.int)

        output_cls1 = (keypoints1 + pred_offsets.transpose(1,2,0)[keypoints1[:,1], keypoints1[:,0]]) * self.map_downsample 
        output_cls2 = (keypoints2 + pred_offsets.transpose(1,2,0)[keypoints2[:,1], keypoints2[:,0]]) * self.map_downsample 
        
        
        #根据label_cls1，label_cls2，output_cls1, output_cls2计算准确率，召回率，平均距离
        num_matches, avg_error = match_coordinates(label_cls1, output_cls1, self.distance_threshold)
        num_matches_total += num_matches
        error_total += num_matches * avg_error
        num_label_total += label_cls1.shape[0]
        num_predict_total += output_cls1.shape[0]
        
        num_matches, avg_error = match_coordinates(label_cls2, output_cls2, self.distance_threshold)
        num_matches_total += num_matches
        error_total += num_matches * avg_error
        num_label_total += label_cls2.shape[0]
        num_predict_total += output_cls2.shape[0]
        
        # acc = num_matches_total / num_predict_total
        # recall = num_matches_total / num_label_total
        # avg_error = error_total / num_matches_total
        
        return num_matches_total, num_predict_total, num_label_total, error_total