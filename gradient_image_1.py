import numpy as np
import cv2 as cv2
import torch
from matplotlib import pyplot as plt
from skimage._shared.filters import gaussian
from skimage.color import rgb2gray


def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


if __name__ == '__main__':
    # img = cv2.imread('dataset_1/testB/27.bmp', 0)
    img = cv2.imread('gradient_test.png', 0)

    # img = rgb2gray(img)
    # print(img.shape)
    print(type(img))
    # print(img)
    # img = gaussian(img)
    # img = img - np.mean(img)
    img_grad = np.gradient(img)
    img_grad = norm(img_grad)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img, 'gray')
    ax = fig.add_subplot(122)
    ax.imshow(img_grad, 'gray')

    plt.show()
