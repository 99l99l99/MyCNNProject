import cv2
import numpy as np
import os

def square_image(image, size):
    """将不规则的图片转换为正方形，并用灰色填充多余的部分。
    
    Args:
        image: 输入的图片，可以是任意尺寸和形状的图像。
        size: 目标正方形的大小。
    
    Returns:
        返回转换后的正方形图片。
    """
    # 获取输入图像的宽度和高度
    h, w = image.shape[:2]

    # 计算最长边的长度，作为新图片的大小
    max_dim = max(h, w)
    new_shape = (max_dim, max_dim)

    # 计算左上角坐标
    left = int((max_dim - w) / 2)
    top = int((max_dim - h) / 2)

    # 创建一个空白的正方形图像，并将其填充为灰色
    square_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    square_image.fill(128)

    # 将输入图像复制到正方形图像的中心
    square_image[top:top+h, left:left+w] = image

    # 缩放正方形图像为目标尺寸
    square_image = cv2.resize(square_image, (size, size))

    return square_image, {"resize":size/max_dim, "left_top":(left, top)}

class MyImageCapture():
    def __init__(self,img_pkg):
        self.img_pkg = img_pkg
        self.filenames = os.listdir(img_pkg)
        self.index = 0
    
    def read(self):
        image_path = os.path.join(self.img_pkg, self.filenames[self.index])
        image = cv2.imread(image_path)
        self.index += 1
        return True, image
 
def gaussian_2d(x, y, mu, sigma):
    """
    计算二维高斯分布的概率密度函数
    x: 横坐标，可以是一个数字或一个向量
    y: 纵坐标，可以是一个数字或一个向量
    mu: 二维均值向量，形如 [mu_x, mu_y]
    sigma: 二维协方差矩阵，形如 [[sigma_x^2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y, sigma_y^2]]
    """
    x_mu = x - mu[0]
    y_mu = y - mu[1]
    sigma_inv = np.linalg.inv(sigma)
    z = x_mu**2 * sigma_inv[0][0] + (x_mu*y_mu) * (sigma_inv[0][1] + sigma_inv[1][0]) + y_mu**2 * sigma_inv[1][1]
    return np.exp(-0.5*z)

def bgr_to_gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.stack([image, image, image], axis = -1)
    return image

def match_coordinates(coords1, coords2, dist_threshold):
    num_matches = 0
    total_error = 0
    
    for i in range(coords1.shape[0]):
        for j in range(coords2.shape[0]):
            dist = np.linalg.norm(coords1[i] - coords2[j])
            if dist < dist_threshold:
                num_matches += 1
                total_error += dist
    
    if num_matches > 0:
        avg_error = total_error / num_matches
    else:
        avg_error = 0
    
    return num_matches, avg_error

def show_map_gray(map):
    map = map * 255
    map = np.array(map, dtype=np.uint8)
    map = cv2.resize(map, None, fx = 4, fy = 4)
    cv2.imshow('map',map)
    cv2.waitKey()

def map_nms(map, suppression_sigma, prob_threshold, max_num):
    key_points = []
    for i in range(0, max_num):
        # show_map_gray(map)
        # 找热图最大值，及其下标
        index = np.unravel_index(map.argmax(), map.shape)
        max = map[index]
        
        # 最大值超过阈值，视为特征点，否者退出
        if max > prob_threshold:
            key_points.append([index[1], index[0]])
        else:
            break
        
        #抑制最大值附近的值，采用高斯蒙版抑制，sigama取训练时的均值
        x, y = np.meshgrid(np.arange(map.shape[0]), np.arange(map.shape[1]))
        x_coord = index[1]
        y_coord = index[0]
        gauss_mask = gaussian_2d(x, y, [x_coord, y_coord], [[suppression_sigma, 0], [0, suppression_sigma]])
        map = map * (1 - gauss_mask)
    
    return key_points

# 计算图像的均值和方差
def calculate_mean_std(image):
    image = np.array(image)
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))
    return mean, std

# 将一幅图像的均值和方差变换到和另一幅图像相等
def match_mean_std(image, target_mean, target_std):
    mean, std = calculate_mean_std(image)
    result = ((image - mean) / std) * target_std + target_mean
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


