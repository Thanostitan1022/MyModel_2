#########################################################################
############################  utils.py  ###################################
import os
import glob
import random
import torch
import itertools
import datetime
import time
import sys
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from torchvision.utils import save_image, make_grid


# 先前生成的样本的缓冲区
class ReplayBuffer:
    def __init__(self, max_size=500):  # 原来是50 ！！！！！！！！！！！
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):  # 放入一张图像，再从buffer里取一张出来
        to_return = []  # 确保数据的随机性，判断真假图片的鉴别器识别率
        for element in data.data:
            element = torch.unsqueeze(element, 0)  # 对数据的维度进行扩充
            if len(self.data) < self.max_size:  # 最多放入50张，没满就一直添加
                self.data.append(element)
                to_return.append(element)
            else:  # 放满了max_size张
                if random.uniform(0, 1) > 0.5:  # 满了就1/2的概率从buffer里取，或者就用当前的输入图片
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


# 设置学习率为初始学习率乘以给定lr_lambda函数的值
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):  # (n_epochs = 50, offset = epoch, decay_start_epoch = 30)
        assert (
                       n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"  # 断言，要让n_epochs > decay_start_epoch 才可以
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):  # return    1-max(0, epoch - 30) / (50 - 30)
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
