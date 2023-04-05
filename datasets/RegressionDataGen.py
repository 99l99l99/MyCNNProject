from torch.utils.data import Dataset
from .BaseDataGen import BaseDataset
import glob
import cv2
import torch
import numpy as np
from utls.utls import gaussian_2d, bgr_to_gray
# 自定义数据集类
class RegressionDataGen(BaseDataset):
    def __init__(self, cfg_data, cfg_model):
        super().__init__(cfg_data)
        self.num_class = cfg_model["num_class"]
        self.to_gray = cfg_model["to_gray"]
    
    def __getitem__(self, index):
        
        backgroud = cv2.imread(self._mix_list[index])

        imw = self.randint(20, backgroud.shape[1]-20)
        imh = self.randint(20, backgroud.shape[0]-20)
        xmin = self.randint(0, backgroud.shape[1]-imw)
        ymin = self.randint(0, backgroud.shape[0]-imh)
        xmax = xmin + imw
        ymax = ymin + imh

        backgroud = cv2.resize(backgroud[ymin:ymax,xmin:xmax], tuple(self.cfg['resize']))
        
        image, label, uav_size = self.paste_uav(backgroud, size = [80, 150], wh_ratio = 0.8, paste_rate = 0.9, cfg = self.cfg['uav'])
        
        if self.cfg['photometric']['enable']:
            image = self.photometric_augmentation(image, self.cfg['photometric']['params'])
            # image_show = image.copy() #for show
        if self.cfg['homographic']['enable']:
            H = self.homographic_augmentation(image.shape[0], image.shape[1], self.cfg['homographic']['params'])
            image = cv2.warpPerspective(image, H, tuple(self.cfg['resize']), borderMode=cv2.BORDER_CONSTANT, borderValue=[self.randint(0,255)]*3)
            if self.to_gray:
                image = bgr_to_gray(image)
            image_show = image.copy() #for show
            label[:,1:] = self.warpPerspectivePoints(label[:,1:], H)

        points = np.array(label, dtype=np.int) #for show
        # convert to net work input
        # generate gauss map
        coord_map = self.gen_coord_map(label)
        
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
        
        
        return image, coord_map

    def __len__(self):
        return len(self.image_list)

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
