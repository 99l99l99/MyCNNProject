# python testHeatMap.py --image data/val/virtual
# python testHeatMap.py --image data/val/rectImages/images/0 --epoch 30
# python testHeatMap.py --video data/val/test1.mp4 --epoch 30
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model.heatMapBasedNet import HeatMapBasedNet
from utls.loss import L2_loss
from utls.utls import square_image, MyImageCapture, bgr_to_gray, map_nms
import yaml
import cv2
import numpy as np
import argparse

def distance_line_point(point0, k, point1):
    return (k * point1[1] - point1[0] + point0[0] - k * point0[1])**2 / (k**2 + 1)

# 将keypoints_a与keypoints_b匹配，匹配后点的连线尽可能平行
# 算法：先旋转线找匹配
# 再最小二乘拟合斜率
def match_keypoints(keypoints_a, keypoints_b):
    # print(keypoints_a)
    # print(keypoints_b)
    
    # print("len(a): %d, len_b: %d"%(len(keypoints_a), len(keypoints_b)))
    n = 10
    max_k = np.tan(10/180*np.pi)
    distance_threshould = 50
    min_mean_distance = 999
    great_k = -1
    for i in range(0,n):
        totol_distance = []
        k = max_k * 2 / (n-1) * i - max_k
        for kp_a in keypoints_a:
            min_distance = 999
            for kp_b in keypoints_b:
                distance = distance_line_point(kp_a, k, kp_b)
                # print(distance)
                if distance < min_distance:
                    min_distance = distance
            if min_distance < distance_threshould:
                totol_distance.append(min_distance)
        if len(totol_distance) > 0:
            mean_distance = np.array(totol_distance).sum() / len(totol_distance)
        else:
            mean_distance = 999
        if mean_distance < min_mean_distance:
            min_mean_distance = mean_distance
            great_k = k
    # 使用最好的k计算match
    matchs = []
    if great_k != -1:
        for kp_a in keypoints_a:
            min_distance = 999
            for kp_b in keypoints_b:
                distance = distance_line_point(kp_a, great_k, kp_b)
                print(distance)
                if distance < min_distance:
                    min_distance = distance
                    match_points = [kp_a, kp_b]
            if min_distance < distance_threshould:
                matchs.append(match_points)
    return great_k, matchs

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None)
parser.add_argument('--video', type=str, default=None)
parser.add_argument('--epoch', type=str, default=None)
args = parser.parse_args()

# 加载cfg文件
with open("cfg/trainHeatMap.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# 修改cfg文件
config['model']['device'] = 'cpu'
# 定义模型
model = HeatMapBasedNet(config['model'])

# 加载预训练的模型参数
if args.epoch != None:
    model.load_state_dict(torch.load("save/"+config['train']['save_pkg'] + "/model_epoch%d.pth"%(int(args.epoch))))
else:
    model.load_state_dict(torch.load("save/"+config['train']['save_pkg'] + "/model.pth"))

# 将模型设置为评估模式
model.eval()

if args.video != None:
    cap = cv2.VideoCapture(args.video)
elif args.image != None:
    cap = MyImageCapture(args.image)
else:
    raise ValueError('please input image or video')
    
while 1:
    # 加载图片并进行预处理
    ret, image = cap.read()
    image,_ = square_image(image, config['data']['resize'][0])
    image = bgr_to_gray(image)
    image_draw_points = image.copy()
    input = torch.tensor(image).permute(2, 0, 1)
    input = input / 255
    input_batch = input.unsqueeze(0)  # 将单张图片转换为 batch
    
    # 使用模型进行预测
    with torch.no_grad():
        heatmap, offsets = model(input_batch)
    
    # 显示map
    offsets = np.array(offsets[0])
    map_append = np.zeros(image.shape)
    keypoints_a = []
    keypoints_b = []
    for i in range(heatmap.shape[1]):
        
        heatmap_layer = heatmap[0][i]
        
        keypoints = map_nms(heatmap_layer, suppression_sigma = 1, prob_threshold = 0.7, max_num = 4)
        if len(keypoints) > 0:
            keypoints = np.array(keypoints)
        else:
            keypoints = np.zeros((0,2), np.int)
        keypoints = (keypoints + offsets.transpose(1,2,0)[keypoints[:,1], keypoints[:,0]]) * 4
        keypoints = np.int0(keypoints)
        heatmap_layer = heatmap_layer * 255
        heatmap_layer = np.array(heatmap_layer, dtype=np.uint8)
        heatmap_layer = cv2.resize(heatmap_layer, None, fx = 4, fy = 4)
        map_append[...,0] = map_append[...,0] + 1 / heatmap.shape[1] * heatmap_layer
        map_append[...,1] = map_append[...,1] + 1 / heatmap.shape[1] * heatmap_layer
        map_append[...,2] = map_append[...,2] + 1 / heatmap.shape[1] * heatmap_layer
        cv2.imshow('%d'%(i), heatmap_layer)
        for j in range(0, keypoints.shape[0]):
            if i == 0:
                cv2.circle(image_draw_points, (keypoints[j]), 3, (255,255,255), 1)
                keypoints_a.append(keypoints[j])
            else:
                cv2.circle(image_draw_points, (keypoints[j]), 3, (0,255,0), 1)
                keypoints_b.append(keypoints[j])
    k , matchs= match_keypoints(keypoints_a, keypoints_b)
    for match in matchs:
        cv2.line(image_draw_points, match[0], match[1], (0, 255,0), 1)
    print(k)
    map_append_image = image * 0.5 + map_append * 1.5
    map_append_image[map_append_image > 255] = 255
    map_append_image = map_append_image.astype(np.uint8)
    cv2.imshow("map_append_image", map_append_image)
    cv2.imshow("image_draw_points", image_draw_points)
    #
    if cv2.waitKey() == 27:
        break
    
    