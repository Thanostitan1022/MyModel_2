#########################################################################
############################  datasets.py  ###################################
import os
import glob
import random

# import numpy
# import torch
# import itertools
# import datetime
# import time
# import sys
# import argparse
# import numpy as np
# import torch.nn.functional as F
# import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
from PIL import Image
# from torchvision.utils import save_image, make_grid
# import cv2


# 如果输入的数据集是灰度图像，将图片转化为rgb图像
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


# 构建数据集
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False,
                 mode="train"):  # (root = "./dataset", unaligned=True:非对其数据)
        self.transform = transforms.Compose(transforms_)  # transform变为tensor数据
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))  # "./dataset/trainA/*.*"
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))  # "./dataset/trainB/*.*"

    def __getitem__(self, index):
        # image_A = Image.open(self.files_A[index % len(self.files_A)])  # 在A中取一张照片
        image_A = Image.open(self.files_A[index % len(self.files_A)]).convert('L')  # 在A中取一张照片,并转换为通道数为1的灰度图

        if self.unaligned:  # 如果采用非配对数据，在B中随机取一张
            # image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('L')
        else:
            # image_B = Image.open(self.files_B[index % len(self.files_B)])
            image_B = Image.open(self.files_B[index % len(self.files_B)]).convert('L')

        # print(type(image_A))  # 此时image_A是PIL.Image.Image

        # 如果是灰度图，把灰度图转换为RGB图
        # if image_A.mode != "RGB":
        #     image_A = to_rgb(image_A)
        # if image_B.mode != "RGB":
        #     image_B = to_rgb(image_B)

        # 把RGB图像转换为tensor图, 方便计算，返回字典数据
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    # 获取A,B数据的长度
    def __len__(self):
        # print("A、B两组数据集的最大长度:"+str(max(len(self.files_A), len(self.files_B))))
        return max(len(self.files_A), len(self.files_B))


# class ImageDataset_G2(Dataset):
#     def __init__(self, root, transforms_=None, unaligned=False,  # 表示已对齐的数据
#                  mode="train"):  # (root = "./dataset", unaligned=True:非对其数据)
#         self.transform = transforms.Compose(transforms_)  # transform变为tensor数据
#         self.unaligned = unaligned
#
#         if mode == 'train':
#             self.files_A = sorted(glob.glob(os.path.join(root, "image_F_G1_train") + "/*.*"))  # "./dataset/trainA/*.*"
#         else:
#             self.files_A = sorted(glob.glob(os.path.join(root, "image_F_G1_test") + "/*.*"))  # "./dataset/trainA/*.*"
#         self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))  # "./dataset/trainB/*.*"
#
#     def __getitem__(self, index):
#         # image_A = Image.open(self.files_A[index % len(self.files_A)])  # 在A中取一张照片
#         image_A = Image.open(self.files_A[index % len(self.files_A)]).convert('L')  # 在A中取一张照片,并转换为通道数为1的灰度图
#
#         if self.unaligned:  # 如果采用非配对数据，在B中随机取一张
#             # image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
#             image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('L')
#         else:
#             # image_B = Image.open(self.files_B[index % len(self.files_B)])
#             image_B = Image.open(self.files_B[index % len(self.files_B)]).convert('L')
#
#         # print(type(image_A))  # 此时image_A是PIL.Image.Image
#
#         # 如果是灰度图，把灰度图转换为RGB图
#         # if image_A.mode != "RGB":
#         #     image_A = to_rgb(image_A)
#         # if image_B.mode != "RGB":
#         #     image_B = to_rgb(image_B)
#
#         # 把RGB图像转换为tensor图, 方便计算，返回字典数据
#         item_A = self.transform(image_A)
#         item_B = self.transform(image_B)
#         return {"A": item_A, "B": item_B}
#
#     # 获取A,B数据的长度
#     def __len__(self):
#         # print("A、B两组数据集的最大长度:"+str(max(len(self.files_A), len(self.files_B))))
#         return max(len(self.files_A), len(self.files_B))
