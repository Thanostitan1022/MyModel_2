import numpy as np
import cv2 as cv2
import torch
from matplotlib import pyplot as plt
from skimage._shared.filters import gaussian

from skimage.color import rgb2gray


def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


def edge_extraction_1(x):  # 传入一个tensor x (即灰度图x)

    x = x.cpu().numpy()
    x = gaussian(x)
    x = x - np.mean(x)
    # print(x.shape)
    # print((type(x)))
    # print(x)
    x_grad = np.gradient(x, axis=(2, 3))

    # print("1111111111111111111111")

    x_grad = norm(x_grad)
    x_grad = torch.tensor(x_grad)
    return x_grad

