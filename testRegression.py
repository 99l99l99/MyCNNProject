# python testRegression.py --image data/val/virtual
# python testRegression.py --image data/val/rectImages/images/0
# python testRegression.py --video data/val/test1.mp4
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model.regressionBasedNet import RegressionBasedNet
from utls.loss import L2_loss
from utls.utls import square_image, MyImageCapture, bgr_to_gray
import yaml
import cv2
import numpy as np
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None)
parser.add_argument('--video', type=str, default=None)
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
    if config['model']['to_gray']:
        image = bgr_to_gray(image)
    input = torch.tensor(image).permute(2, 0, 1)
    input = input / 255
    input_batch = input.unsqueeze(0)  # 将单张图片转换为 batch
    
    # 使用模型进行预测
    with torch.no_grad():
        output = model(input_batch)
    
    # 展示输出
    prob_threshold = 0.8
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

    print("prob: ", end = '')
    print(prob)
    for i in range(0, output_cls1.shape[0]):
        cv2.circle(image, output_cls1[i], 5, (255,255,255), -1)
    for i in range(0, output_cls2.shape[0]):
        cv2.circle(image, output_cls2[i], 5, (0,255,0), -1)
    cv2.imshow("img", cv2.resize(image, (256,256)))
    

    if cv2.waitKey() == 27:
        break