from torch.utils.data import Dataset
import glob
import cv2
import torch
import numpy as np
from utls.utls import square_image, bgr_to_gray
# 自定义数据集类
class RegressionValDataGen(Dataset):
    def __init__(self, cfg_data, cfg_model):
        self.cfg = cfg_data
        self.to_gray = cfg_model["to_gray"]
        self.num_class = cfg_model["num_class"]
        self.image_path_list = []
        self.points_list = []
        label_paths = cfg_data['label_paths']
        image_pkgs = cfg_data['image_pkgs']
        for i in range(0, len(label_paths)):
            output_path = label_paths[i]
            image_pkg = image_pkgs[i]
            with open(output_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split(',')
                    self.image_path_list.append(image_pkg + line[0])
                    points = [int(i) for i in line[1:]]
                    points = np.array(points)
                    points = points.reshape(-1,3)
                    self.points_list.append(points)

    def __getitem__(self, index):
        image = cv2.imread(self.image_path_list[index])
        image, trans = square_image(image, self.cfg['resize'][0])
        points_label = self.points_list[index]
        points_label[:,1:] = (points_label[:,1:] + trans["left_top"]) * trans["resize"]
        points = np.array(points_label, dtype=np.int) #for show
        coord_map = self.gen_coord_map(points_label)
        
        if self.to_gray:
            image = bgr_to_gray(image)
        image_show = image.copy() #for show
        image = torch.tensor(image).permute(2, 0, 1)
        image = image / 255
        
        
        # # show label
        # for i in range(points.shape[0]):
        #     if points[i][0] == 0:
        #         cv2.circle(image_show, points[i][1:], 2, (255,255,255), 1)
        #     else:
        #         cv2.circle(image_show, points[i][1:], 2, (0,255,0), 1)
        # cv2.imshow("image_change", cv2.resize(image_show, (256,256)))
        # cv2.waitKey()
        
        num_point = points_label.shape[0]
        assert num_point <= 8
        points_label_tensor = torch.ones((8, 3)) * 2
        points_label_tensor[0:num_point] = torch.tensor(points_label)
        return image, coord_map, points_label_tensor

    def __len__(self):
        return len(self.image_path_list)
    
    def gen_coord_map(self, points):
        
        coord_map = np.zeros((1+self.num_class+2, self.cfg['resize'][1] // self.cfg['map_downsample'], self.cfg['resize'][0] // self.cfg['map_downsample']))
        
        for i in range(0, points.shape[0]):
            #将超出边界的points，跳过
            if points[i][1] < 0 or points[i][1] >= self.cfg['resize'][0] or points[i][2] < 0 or points[i][2] >= self.cfg['resize'][1]: 
                continue
            pos_w = int(np.floor(points[i][1] / self.cfg['map_downsample']))
            pos_h = int(np.floor(points[i][2] / self.cfg['map_downsample']))
            coord_map[0, pos_h, pos_w] = 1
            coord_map[int(points[i][0]) + 1, pos_h, pos_w] = 1
            coord_map[-2, pos_h, pos_w] = points[i][1]
            coord_map[-1, pos_h, pos_w] = points[i][2]
        
        return coord_map