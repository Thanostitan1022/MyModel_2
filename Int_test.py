import os
import sys
import matplotlib
# from matplotlib import imsave
import numpy as np
import torch
import cv2 as cv2

img = cv2.imread('dataset_1/testA/24.bmp', 0)
cv2.imshow("img", img)

# img1 = img[70:90, 195:205]  # 第一个表示行，第二个表示列 像素点位置
# cv2.imshow("img1", img1)
# print(img1)  # 最高两部分是255

# 获取大于平均像素值的部分
# average_pixel = cv2.mean(img)[0]
# print(average_pixel)
# img2 = img > 230  # 返回的是一个 [true,false]矩阵，大于230的为true，astype('uint8')后为[1,0]
# print(img2)
# print(img2.astype('uint8'))
# cv2.imshow('img2', img2.astype('uint8') * 255)

# 获取大于指定像素值的部分
# img2 = np.where(img > 230, img, 0)
# cv2.imshow("img2", img2)

# 将图像转换为一维数组
image_flat = img.flatten()
# 计算像素值的百分位数
percentile_value = np.percentile(image_flat, 90)  # 得到的是一个具体的像素值大小
# print(percentile_value)
# 获取高于百分之90的像素点的部分
img3 = np.where(img > percentile_value, img, 0)  # img中大于阈值的就取img原像素点，否则取0
print(img3)
cv2.imshow('img3', img3)

cv2.waitKey(0)
cv2.destroyAllWindows()

# a = 8
# b = 4
# print(a if a > b else b)

# a = torch.arange(0, 24).reshape(4, 3, 2)
# print(a)


# sample = sample.astype('uint8')
# imsave(os.path.join(os.path.join(out_path, 'recon'), "{}.png".format(img_name.split(".")[0])), sample)