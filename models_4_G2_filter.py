# ########################################################################
# ###########################  models.py  ###################################
# 此生成器完全密集连接！！！！！
# 且生成器输入为三个单通道（红外，可见光，中间结果）
# 再连接两个滤波图像

# import os
# import glob
# import random
import torch
# import itertools
# import datetime
# import time
# import sys
# import argparse
# import numpy as np
# import torch.nn.functional as F
import torch.nn as nn


# import torchvision.transforms as transforms
# from tensorboard import summary
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
# from PIL import Image
# from torchvision.utils import save_image, make_grid
# from torchsummary import summary


# 定义参数初始化函数
def weights_init_normal(m):
    classname = m.__class__.__name__  # m作为一个形参，原则上可以传递很多的内容, 为了实现多实参传递，每一个moudle要给出自己的name. 所以这句话就是返回m的名字.
    if classname.find("Conv") != -1:  # find():实现查找classname中是否含有Conv字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 0.0,
                              0.02)  # m.weight.data表示需要初始化的权重。nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        if hasattr(m, "bias") and m.bias is not None:  # hasattr():用于判断m是否包含对应的属性bias, 以及bias属性是否不为空.
            torch.nn.init.constant_(m.bias.data, 0.0)  # nn.init.constant_():表示将偏差定义为常量0.
    elif classname.find("BatchNorm2d") != -1:  # find():实现查找classname中是否含有BatchNorm2d字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 1.0,
                              0.02)  # m.weight.data表示需要初始化的权重. nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        torch.nn.init.constant_(m.bias.data, 0.0)  # nn.init.constant_():表示将偏差定义为常量0.


##############################
#  残差块儿ResidualBlock
##############################
# class ResidualBlock(nn.Module):
#     def __init__(self, in_features):
#         super(ResidualBlock, self).__init__()
#
#         self.block = nn.Sequential(  # block = [pad + conv + norm + relu + pad + conv + norm]
#             nn.ReflectionPad2d(1),  # ReflectionPad2d():利用输入边界的反射来填充输入张量
#             nn.Conv2d(in_features, in_features, 3),  # 卷积  参数：(in_channels, out_channels, kernel_size)
#
#             # nn.InstanceNorm2d(in_features),  # InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
#             nn.BatchNorm2d(in_features),
#
#             nn.ReLU(inplace=True),  # 非线性激活
#             nn.ReflectionPad2d(1),  # ReflectionPad2d():利用输入边界的反射来填充输入张量
#             nn.Conv2d(in_features, in_features, 3),  # 卷积
#
#             # nn.InstanceNorm2d(in_features),  # InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
#             nn.BatchNorm2d(in_features)
#
#         )
#
#     def forward(self, x):  # 输入为 一张图像(tensor)
#         return x + self.block(x)  # 输出为 图像加上网络的残差输出  注意这里是 加上！！！ 既有x 也有残差块block


##############################
#  生成器网络GeneratorResNet
##############################

'''
先将两个input_shape为(3,256,256)的两幅图像concat,
再进入输入通道，再连接residual_block
再连接上采样以及输出层
'''


class GeneratorResNet_2(nn.Module):
    def __init__(self, input_shape):  # (input_shape_A 和 input_shape_B = (3, 256, 256),
        # num_residual_blocks = 9)  # 原来还有一个参数 num_residual_blocks  表示有多少个残差块
        super(GeneratorResNet_2, self).__init__()

        input_shape = input_shape
        model = []
        # self.num_residual_blocks = num_residual_blocks
        channels = input_shape[0]  # 图片A输入通道数channels = 3
        channels *= 5

        # 初始化网络结构
        # 首先初始化 输入通道 的网络结构
        out_features = 256  # 输出特征数out_features = 64
        # 111111111111111111111111111111111111111111111111111111111111111111111111111111111
        model_1 = [  # model = [Pad + Conv + Norm + ReLU]
            nn.Conv2d(channels, out_features, kernel_size=5, stride=1, padding=2),
            # 参数：(in_channels, out_channels, kernel_size)
            nn.BatchNorm2d(out_features),
            # nn.ReLU(inplace=True),  # 非线性激活
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.2),
        ]
        self.model_1 = nn.Sequential(*model_1)

        # 222222222222222222222222222222222222222222222222222222222222222222222222222222222
        in_features = out_features  # in_features = 256
        out_features = out_features // 2  # out_features = 128
        model_2 = [
            nn.Conv2d(in_features, out_features, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_features),
            # nn.ReLU(inplace=True),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.2),
        ]
        self.model_2 = nn.Sequential(*model_2)

        # 3333333333333333333333333333333333333333333333333333333333333333333333333333333333
        in_features = 256+128  # in_features = 128  256+128=384
        out_features = 64  # out_features = 64
        model_3 = [
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_features),
            # nn.ReLU(inplace=True),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.2),
        ]
        self.model_3 = nn.Sequential(*model_3)

        # 44444444444444444444444444444444444444444444444444444444444444444444444444444444444
        in_features = 256 + 128 + 64   # in_features = 64
        out_features = 32  # out_features = 32
        model_4 = [
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_features),
            # nn.ReLU(inplace=True),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.2),
        ]
        self.model_4 = nn.Sequential(*model_4)

        # 5555555555555555555555555555555555555555555555555555555555555555555555555555555555
        in_features = 256 + 128 + 64 + 32  # in_features = 32
        out_features = 16  # out_features = 16
        model_5 = [
            nn.Conv2d(in_features, 1, kernel_size=1, stride=1),
            nn.Tanh()
            # nn.Sigmoid()
        ]
        self.model_5 = nn.Sequential(*model_5)

    def forward(self, x1, x2, x3, x4, x5):  # 输入(1, 3, 256, 256) (1, 3, 256, 256) (batchsize, channels, height, width)
        x = torch.cat((x1, x2), dim=1)  # 在图片进入输入通道之前先拼接，concat后，通道数相加
        x = torch.cat((x, x3), dim=1)
        x = torch.cat((x, x4), dim=1)
        x = torch.cat((x, x5), dim=1)
        # x = torch.add(x1, x2)

        result_1 = self.model_1(x)
        result_2 = self.model_2(result_1)

        input_3 = torch.cat((result_1, result_2), dim=1)  # 此时下一步的输入通道等于 前两步的输出通道之和 256+128=384
        result_3 = self.model_3(input_3)

        input_4 = torch.cat((input_3, result_3), dim=1)  # 此时下一步的输入通道等于 前三步的输出通道之和 256+128+64=448
        result_4 = self.model_4(input_4)

        input_5 = torch.cat((input_4, result_4), dim=1)  # 此时下一步的输入通道等于 前四步的输出通道之和 256+128+64+32=480
        result_5 = self.model_5(input_5)

        return result_5  # 输出(1, 3, 256, 256)


# 查看模型的每一步卷积shape！！！非常好用！！！
# test_G = GeneratorResNet_1((1, 256, 256)).cuda()
# summary(test_G, input_size=[(1, 256, 256), (1, 256, 256)])


##############################
#        Discriminator
##############################
class Discriminator_2(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator_2, self).__init__()

        channels, height, width = input_shape  # input_shape:(3, 256, 256)

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)  # output_shape = (1, 16, 16)
        '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!输入图像是3维的，输出图像是1维16*16的概率矩阵!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        patchGAN有关，这是判别器。不是生成器！！！output不是融合图像，而是判别器的输出结果（矩阵）      '''

        def discriminator_block(in_filters, out_filters, normalize=True):  # 鉴别器块儿
            """Returns downsampling layers of each discriminator block"""
            layers = [
                nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]  # layer += [conv + norm + relu]
            if normalize:  # 每次卷积尺寸会缩小一半，共卷积了4次
                # layers.append(nn.InstanceNorm2d(out_filters))
                layers.append(nn.BatchNorm2d(out_filters))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 32, normalize=False),  # layer += [conv(3, 64) + relu]
            *discriminator_block(32, 64),  # layer += [conv(64, 128) + norm + relu]
            *discriminator_block(64, 128),  # layer += [conv(128, 256) + norm + relu]
            *discriminator_block(128, 256),  # layer += [conv(256, 512) + norm + relu]

            nn.ZeroPad2d((1, 0, 1, 0)),  # layer += [pad]  左、右、上、下 分别以0进行填充 1,0,1,0行（列）
            # nn.ZeroPad2d((1, 1, 1, 1)),
            # 这里就是左和上各填充一（列）行0像素
            nn.Conv2d(256, 1, kernel_size=4, padding=1)  # layer += [conv(512, 1)]
        )

    def forward(self, img):  # 输入(1, 3, 256, 256)
        return self.model(img)  # 输出(1, 1, 16, 16)  输出结果为(batchsize=1, channels=1, H=16, W=16)的矩阵，表示patchGAN

# test_D = Discriminator_1((1, 256, 256)).cuda()
# summary(test_D, input_size=(1, 256, 256))
