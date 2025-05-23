import math
import cv2 as cv2
import numpy as np


def guidedFilter(I, p, winSize, eps):
    # 输入图像的高，宽
    rows, cols = I.shape
    # I的均值平滑
    mean_I = cv2.blur(I, winSize, borderType=cv2.BORDER_DEFAULT)
    # p的均值平滑
    mean_p = cv2.blur(p, winSize, borderType=cv2.BORDER_DEFAULT)
    # I .* p 的均值平滑
    Ip = I * p
    mean_Ip = cv2.blur(Ip, winSize, borderType=cv2.BORDER_DEFAULT)
    # 协方差
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.blur(I * I, winSize, borderType=cv2.BORDER_DEFAULT)
    # 方差
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    # 对a和b进行均值平滑
    mean_a = cv2.blur(a, winSize, borderType=cv2.BORDER_DEFAULT)
    mean_b = cv2.blur(b, winSize, borderType=cv2.BORDER_DEFAULT)
    q = mean_a * I + mean_b
    return b


if __name__ == '__main__':
    image = cv2.imread('dataset_1/testB/25.bmp', 0)
    # image = cv2.imread('dataset_1/image_F_G1_test/0018.png', 0)
    # 将图像进行归一化
    image_0_1 = image / 255.0
    # 显示原图
    cv2.imshow('image', image)

    # 导向滤波
    result = guidedFilter(image_0_1, image_0_1, (7, 7), math.pow(0.9, 2.0))
    cv2.imshow('guidedFilter', result)

    # 细节增强
    result_enhanced = (image_0_1 - result) * 5 + result
    # result_enhanced = cv2.normalize(result_enhanced, result_enhanced, 1, 0, cv2.NORM_MINMAX)
    # cv2.imshow('result_enhanced', result_enhanced)

    image_sub_1 = image_0_1 - result_enhanced
    # cv2.imshow('image_sub_1', image_sub_1)

    image_sub_2 = (image_0_1 - result) * 2.0
    cv2.imshow('image_sub_2', image_sub_2)   # 此结果是GF_detail.py里的最终滤波结果

    # detail = (image_sub_2 + result_enhanced) / 2.0
    # cv2.imshow('detail', detail)
    # 保存导向滤波的结果
    result = result * 255
    result[result > 255] = 255
    result = np.round(result)
    result = result.astype(np.uint8)
    # cv2.imwrite('guidedFilter.jpg', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
