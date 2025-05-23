import torch
import numpy as np
from torch import nn
# from PIL import Image
# from torch.autograd import Variable
# import torch.nn.functional as F
import cv2


def edge_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    # 定义sobel算子参数，所有值除以3个人觉得出来的图更好些
    sobel_kernel = np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype='float32')
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 卷积输出通道，这里我设置为3
    sobel_kernel = np.repeat(sobel_kernel, 1, axis=1)
    # 输入图的通道，这里我设置为3
    sobel_kernel = np.repeat(sobel_kernel, 1, axis=0)

    conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    edge_detect = conv_op(im)
    # print(torch.max(edge_detect))
    # 将输出转换为图片格式
    edge_detect = edge_detect.squeeze().detach().cpu().numpy()
    edge_detect = cv2.convertScaleAbs(edge_detect)
    return edge_detect


def edge_extraction(im):  # im是一个tensor
    edge_detect = edge_conv2d(im)  # im是一个tensor，返回的edge_detect是一个numpy
    edge_detect = torch.tensor(edge_detect, dtype=torch.float32)  # 此时转成了tensor
    return edge_detect
