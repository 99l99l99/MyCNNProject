# python testRegression.py 
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model.regressionBasedNet import RegressionBasedNet
from utls.loss import L2_loss
from utls.utls import square_image, MyImageCapture, bgr_to_gray, match_coordinates
import yaml
import cv2
import numpy as np
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=str, default=None)
args = parser.parse_args()

# 加载cfg文件
with open("cfg/trainRegression.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# 修改cfg文件
config['model']['device'] = 'cpu'
# 定义模型
model = RegressionBasedNet(config['model'])

# 加载预训练的模型参数
if args.epoch != None:
    model.load_state_dict(torch.load("save/"+config['train']['save_pkg'] + "/model_epoch%d.pth"%(args.epoch)))
else:
    model.load_state_dict(torch.load("save/"+config['train']['save_pkg'] + "/model.pth"))

# 将模型设置为评估模式
model.eval()

output_paths = ["data/val/rectImages/images/val_label0.txt", "data/val/rectImages/images/val_label1.txt"]
image_pkgs = ["data/val/rectImages/images/0/", "data/val/rectImages/images/1/"]
assert len(output_paths) == len(image_pkgs)
distance_threshold = 5
prob_threshold = 0.8
image_path_list = []
points_list = []
for i in range(0, len(output_paths)):
    output_path = output_paths[i]
    image_pkg = image_pkgs[i]
    with open(output_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            image_path_list.append(image_pkg + line[0])
            points = [int(i) for i in line[1:]]
            points = np.array(points)
            points = points.reshape(-1,3)
            points_list.append(points)

num_matches_total = np.array([0,0])
num_label_total = np.array([0,0])
num_predict_total = np.array([0,0])
error_total = np.array([0,0])
for i, image_path in enumerate(image_path_list):
    # print(image_path)
    image = cv2.imread(image_path)
    image, trans = square_image(image, config['data']['resize'][0])
    points_label = points_list[i]
    points_label[:,1:] = (points_label[:,1:] + trans["left_top"]) * trans["resize"]
    points_label = np.int0(points_label)
    label_cls1 = points_label[points_label[:,0] == 0]
    label_cls1 = label_cls1[:,1:]
    label_cls2 = points_label[points_label[:,0] == 1]
    label_cls2 = label_cls2[:,1:]
    
    if config['model']['to_gray']:
        image = bgr_to_gray(image)
    input = torch.tensor(image).permute(2, 0, 1)
    input = input / 255
    input_batch = input.unsqueeze(0)  # 将单张图片转换为 batch
    
    # 使用模型进行预测
    with torch.no_grad():
        output = model(input_batch)
    
    # 展示输出
    
    output = output[0]
    mask_prob = output[0,...] > prob_threshold
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
    num_matches, avg_error = match_coordinates(label_cls1, output_cls1, distance_threshold)
    num_matches_total[0] += num_matches
    error_total[0] += num_matches * avg_error
    num_label_total[0] += label_cls1.shape[0]
    num_predict_total[0] += output_cls1.shape[0]
    
    num_matches, avg_error = match_coordinates(label_cls2, output_cls2, distance_threshold)
    num_matches_total[1] += num_matches
    error_total[1] += num_matches * avg_error
    num_label_total[1] += label_cls2.shape[0]
    num_predict_total[1] += output_cls2.shape[0]
    
    print("%d / %d"%(i, len(image_path_list)))


acc = num_matches_total / num_predict_total
recall = num_matches_total / num_label_total
avg_error = error_total / num_matches_total

acc_all = num_matches_total.sum() / num_predict_total.sum()
recall_all = num_matches_total.sum() / num_label_total.sum()
avg_error_all = error_total.sum() / num_matches_total.sum()


print("acc0: %.5f acc1: %.5f acc: %.5f"%(acc[0], acc[1], acc_all))
print("recall0: %.5f recall1: %.5f recall: %.5f"%(recall[0], recall[1], recall_all))
print("avg_error0: %.5f avg_error1: %.5f avg_error: %.5f"%(avg_error[0], avg_error[1], avg_error_all))

