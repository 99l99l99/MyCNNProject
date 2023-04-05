import cv2
import os
import glob
import numpy as np
from utls.utls import calculate_mean_std, match_mean_std, bgr_to_gray

formwork_pkgs = ["data/val/rectImages/images/0", "data/val/rectImages/images/1"]
output_pkg = "test_out"
change_pkgs = ["data/uav_kaypoints/intensity_600"]

formwork_img_paths = []
for formwork_pkg in formwork_pkgs:
    formwork_img_paths += glob.glob(formwork_pkg + "/*.png")
    
change_img_paths = []
for change_pkg in change_pkgs:
    change_img_paths += glob.glob(change_pkg + "/*.jpeg")

# get formwork mean and std
total_mean = 0
total_std = 0
use_len = len(formwork_img_paths)
for img_path in formwork_img_paths[0:use_len]:
    image = cv2.imread(img_path)
    image = bgr_to_gray(image)
    mean, std = calculate_mean_std(image)
    total_mean += mean
    total_std += std
formwork_mean = total_mean / use_len
formwork_std = total_std / use_len
print(formwork_mean)
print(formwork_std)

# change change_images
for img_path in change_img_paths:
    image = cv2.imread(img_path)
    image = bgr_to_gray(image)
    cv2.imshow("img", image)
    image = match_mean_std(image, formwork_mean, formwork_std)
    cv2.imshow("change_img", image)
    if cv2.waitKey() == ord('q'):
        break