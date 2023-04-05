from torch.utils.data import Dataset
from .BaseDataGen import BaseDataset
import glob
import cv2
import torch
import numpy as np
from utls.utls import gaussian_2d, bgr_to_gray, match_mean_std
# 自定义数据集类
class HeatMapDataset(BaseDataset):
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
        
        image, label, uav_size = self.paste_uav(backgroud, size = [60, 200], wh_ratio = 0.8, paste_rate = 0.9, cfg = self.cfg['uav'])
        if self.cfg['normalization']['enable']:
            image = match_mean_std(image, np.array(self.cfg['normalization']['mean']), np.array(self.cfg['normalization']['std']))
            # image_show = image.copy() #for show
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
        gauss_map, offsets_map = self.gen_map(label, uav_size / self.cfg['resize'][0])
        
        image = torch.tensor(image).permute(2, 0, 1)
        image = image / 255
        
        # show label
        cv2.imshow("image_change", image_show)
        
        for i in range(gauss_map.shape[0]):
            heatmap_layer = gauss_map[i]
            heatmap_layer = heatmap_layer * 255
            heatmap_layer = np.array(heatmap_layer, dtype=np.uint8)
            heatmap_layer = cv2.resize(heatmap_layer, None, fx = 4, fy = 4)
            cv2.imshow('%d'%(i), heatmap_layer)
        cv2.waitKey()
        
        return image, torch.tensor(gauss_map), torch.tensor(offsets_map)

    def __len__(self):
        return len(self.image_list)

    def gen_map(self, points, sigama):
        
        gauss_map = np.zeros((self.num_class, self.cfg['resize'][1] // self.cfg['map_downsample'], self.cfg['resize'][0] // self.cfg['map_downsample']))
        x, y = np.meshgrid(np.arange(gauss_map.shape[1]), np.arange(gauss_map.shape[2]))
        offsets_map = np.zeros((2, self.cfg['resize'][1] // self.cfg['map_downsample'], self.cfg['resize'][0] // self.cfg['map_downsample']))
        for i in range(0, points.shape[0]):
            #将超出边界的points，生成全黑高斯图
            if points[i][1] < 0 or points[i][1] >= self.cfg['resize'][0] or points[i][2] < 0 or points[i][2] >= self.cfg['resize'][1]: 
                continue
            x_coord_float = points[i][1] / self.cfg['map_downsample']
            y_coord_float = points[i][2] / self.cfg['map_downsample']
            x_coord_int = int(points[i][1] // self.cfg['map_downsample'])
            y_coord_int = int(points[i][2] // self.cfg['map_downsample'])
            x_coord_offset = x_coord_float - x_coord_int
            y_coord_offset = y_coord_float - y_coord_int
            offsets_map[0, y_coord_int, x_coord_int] = x_coord_offset
            offsets_map[1, y_coord_int, x_coord_int] = y_coord_offset
            gauss_map[int(points[i][0])] += gaussian_2d(x, y, [x_coord_int, y_coord_int], [[sigama, 0], [0, sigama]])
        gauss_map[gauss_map > 1] = 1
        return gauss_map, offsets_map
