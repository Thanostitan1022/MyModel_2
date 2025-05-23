import torch
import numpy as np
from torch import nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import cv2


def edge_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    # 定义sobel算子参数，所有值除以3个人觉得出来的图更好些
    # sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
    sobel_kernel = np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype='float32')
    # # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 卷积输出通道，这里我设置为3
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
    # 输入图的通道，这里我设置为3
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)

    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    edge_detect = conv_op(im)
    # print(torch.max(edge_detect))
    # 将输出转换为图片格式
    edge_detect = edge_detect.squeeze().detach().numpy()
    edge_detect = cv2.convertScaleAbs(edge_detect)
    return edge_detect


def edge_extraction(x):
    im = cv2.imread(x, flags=-1)
    im = np.transpose(im, (2, 0, 1))
    # 添加一个维度，对应于pytorch模型张量(B, N, H, W)中的batch_size
    im = im[np.newaxis, :]
    im = torch.Tensor(im)
    edge_detect = edge_conv2d(im)
    edge_detect = np.transpose(edge_detect, (1, 2, 0))
    #  求绝对值！！！
    # edge_detect = cv2.convertScaleAbs(edge_detect)
    cv2.imshow('gradient_image', edge_detect)
    cv2.waitKey(0)

    # print(type(edge_detect))  # 此时返回的是numpy.ndarray
    # edge_detect = torch.tensor(edge_detect)
    # # print(type(edge_detect))  # 此时转成了tensor
    #
    # edge_detect = edge_detect.numpy()
    # print(type(edge_detect))   # 此时又恢复成了numpy

    # cv2.imshow('gradient_image', edge_detect)
    # cv2.waitKey(0)
    # 经验证，edge_detect 从numpy->tensor->numpy是可行的
    return edge_detect


image_name = 'output/F_dataset_1/G2/0018.png'
img = cv2.imread(image_name, 0)
# img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(type(img))
print(img.shape)
cv2.imshow('source_image', img)
# cv2.waitKey(0)

edge_extraction(image_name)
