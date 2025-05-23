# #########################################################################
# ############################  models.py  ###################################
# import os
# import glob
# import random
# import torch
# import itertools
# import datetime
# import time
# import sys
# import argparse
# import numpy as np
# import torch.nn.functional as F
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
# from PIL import Image
# from torchvision.utils import save_image, make_grid
#
#
# # 定义参数初始化函数
# def weights_init_normal(m):
#     classname = m.__class__.__name__  # m作为一个形参，原则上可以传递很多的内容, 为了实现多实参传递，每一个moudle要给出自己的name. 所以这句话就是返回m的名字.
#     if classname.find("Conv") != -1:  # find():实现查找classname中是否含有Conv字符，没有返回-1；有返回0.
#         torch.nn.init.normal_(m.weight.data, 0.0,
#                               0.02)  # m.weight.data表示需要初始化的权重。nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
#         if hasattr(m, "bias") and m.bias is not None:  # hasattr():用于判断m是否包含对应的属性bias, 以及bias属性是否不为空.
#             torch.nn.init.constant_(m.bias.data, 0.0)  # nn.init.constant_():表示将偏差定义为常量0.
#     elif classname.find("BatchNorm2d") != -1:  # find():实现查找classname中是否含有BatchNorm2d字符，没有返回-1；有返回0.
#         torch.nn.init.normal_(m.weight.data, 1.0,
#                               0.02)  # m.weight.data表示需要初始化的权重. nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
#         torch.nn.init.constant_(m.bias.data, 0.0)  # nn.init.constant_():表示将偏差定义为常量0.
#
#
# ##############################
# #  残差块儿ResidualBlock
# ##############################
# class ResidualBlock(nn.Module):
#     def __init__(self, in_features):
#         super(ResidualBlock, self).__init__()
#
#         self.block = nn.Sequential(  # block = [pad + conv + norm + relu + pad + conv + norm]
#             nn.ReflectionPad2d(1),  # ReflectionPad2d():利用输入边界的反射来填充输入张量
#             nn.Conv2d(in_features, in_features, 3),  ## 卷积  参数：(in_channels, out_channels, kernel_size)
#             nn.InstanceNorm2d(in_features),  # InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
#             nn.ReLU(inplace=True),  # 非线性激活
#             nn.ReflectionPad2d(1),  # ReflectionPad2d():利用输入边界的反射来填充输入张量
#             nn.Conv2d(in_features, in_features, 3),  ## 卷积
#             nn.InstanceNorm2d(in_features),  # InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
#         )
#
#     def forward(self, x):  # 输入为 一张图像(tensor)
#         return x + self.block(x)  # 输出为 图像加上网络的残差输出  注意这里是 加上！！！ 既有x 也有残差块block
#
#
# ##############################
# #  生成器网络GeneratorResNet
# ##############################
#
# '''
# 需要两个input_shape 为 (3,256,256) 的输入通道，
# 经过下采样后进行Concat，
# 再连接residual_block,
# 再连接上采样以及输出层
# '''
#
# '''
# 一开始是两个独立的输入通道，分别卷积后再concat
# 现在改成一开始就concat，然后一块卷积，再连接残差块
# '''
#
#
# class GeneratorResNet_1(nn.Module):
#     def __init__(self, input_shape_A, input_shape_B,
#                  num_residual_blocks):  # (input_shape_A 和 input_shape_B = (3, 256, 256), num_residual_blocks = 9)
#         super(GeneratorResNet_1, self).__init__()
#
#         model_input_A = []
#         # self.num_residual_blocks = num_residual_blocks
#         channels_A = input_shape_A[0]  # 图片A输入通道数channels = 3
#         channels_B = input_shape_B[0]  # 图片B输入通道数channels = 3
#
#         # 两幅图像首先在forward()里作concat，通道数由3变6，需要将通道数还原回3
#         # print("channel_A:", channels_A)
#         model_input_A += [nn.Conv2d(in_channels=channels_A*2, out_channels=channels_A, kernel_size=3, padding=1)]
#
#         # 初始化网络结构
#         # 首先初始化 输入图像A 的网络结构
#         out_features_A = 64  # 输出特征数out_features = 64
#         model_input_A += [  # model = [Pad + Conv + Norm + ReLU]
#             nn.ReflectionPad2d(channels_A),  # ReflectionPad2d(3):利用输入边界的反射来填充输入张量
#             nn.Conv2d(channels_A, out_features_A, 7),  # Conv2d(3, 64, 7)  参数：(in_channels, out_channels, kernel_size)
#             nn.InstanceNorm2d(out_features_A),  # InstanceNorm2d(64):在图像像素上对HW做归一化，用在风格化迁移
#             nn.ReLU(inplace=True),  # 非线性激活
#         ]
#         in_features_A = out_features_A  # in_features = 64
#
#         # 下采样，循环2次
#         for _ in range(2):
#             out_features_A *= 2  # out_features = 128 -> 256
#             model_input_A += [  # (Conv + Norm + ReLU) * 2
#                 nn.Conv2d(in_features_A, out_features_A, kernel_size=3, stride=2, padding=1),
#                 nn.InstanceNorm2d(out_features_A),
#                 nn.ReLU(inplace=True),
#             ]
#             in_features_A = out_features_A  # 循环两次后，in_features = 256
#         self.model_input_A = nn.Sequential(*model_input_A)
#         # 至此输入通道完结，开始连接残差块
#
#         # 开始初始化输入图像B的网络结构
#         # out_features_B = 64  # 输出特征数out_features = 64
#         # model_input_B = [  # model = [Pad + Conv + Norm + ReLU]
#         #     nn.ReflectionPad2d(channels_B),  # ReflectionPad2d(3):利用输入边界的反射来填充输入张量
#         #     nn.Conv2d(channels_B, out_features_B, 7),  # Conv2d(3, 64, 7)  参数：(in_channels, out_channels, kernel_size)
#         #     nn.InstanceNorm2d(out_features_B),  # InstanceNorm2d(64):在图像像素上对HW做归一化，用在风格化迁移
#         #     nn.ReLU(inplace=True),  # 非线性激活
#         # ]
#         # in_features_B = out_features_B  # in_features = 64
#         #
#         # # 下采样，循环2次
#         # for _ in range(2):
#         #     out_features_B *= 2  # out_features = 128 -> 256
#         #     model_input_B += [  # (Conv + Norm + ReLU) * 2
#         #         nn.Conv2d(in_features_B, out_features_B, kernel_size=3, stride=2, padding=1),
#         #         nn.InstanceNorm2d(out_features_B),
#         #         nn.ReLU(inplace=True),
#         #     ]
#         #     in_features_B = out_features_B  # 循环两次后，in_features = 256
#         # self.model_input_B = nn.Sequential(*model_input_B)
#
#         '''
#         两个输入通道网络初始化完毕，开始Concat拼接两个输入通道  或者把这一步放入 forward 函数里？？？
#         '''
#         # model = torch.cat(model_input_A, model_input_B)  # 显然不对！！！！！！！
#         in_features = in_features_A
#         out_features = out_features_A
#         channels = channels_A
#
#         '''
#         # Concat拼接完成，开始连接 残差块 以及后续 上采样 和 输出网络 ！！！
#         '''
#         # 残差块儿，循环9次
#         model = []
#         # 由于上一步是Concat操作，通道数由256->512，
#         # 所以需要先加一步卷积操作实现降维，把通道数还原为256
#         # model += [nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)]
#
#         # 开始连接9个residual_block
#         # for _ in range(num_residual_blocks):
#         #     model += [ResidualBlock(out_features)]  # model += [pad + conv + norm + relu + pad + conv + norm]
#
#         # 上采样两次
#         for _ in range(2):
#             out_features //= 2  # out_features = 128 -> 64
#             model += [  # model += [Upsample + conv + norm + relu]
#                 nn.Upsample(scale_factor=2),  # scale_factor 指定输出为输入的多少倍 默认使用 nearest 方法进行上采样
#                 nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
#                 nn.InstanceNorm2d(out_features),
#                 nn.ReLU(inplace=True),
#             ]
#             in_features = out_features  # out_features = 64
#
#         # 网络输出层                                                            ## model += [pad + conv + tanh]
#         model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7),
#                   nn.Tanh()]  # 将(3)的数据每一个都映射到[-1, 1]之间
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x1, x2):  # 输入(1, 3, 256, 256) (1, 3, 256, 256) (batchsize, channels, height, width)
#         x = torch.cat((x1, x2), dim=1)  # 在图片进入输入通道之前先拼接，concat后，通道数会变成6
#         x = self.model_input_A(x)
#         # x1 = self.model_input_A(x1)
#         # x2 = self.model_input_B(x2)
#         # x = torch.cat((x1, x2), dim=1)
#         # concat后会导致通道数叠加，也就是说当前通道数变成了512，在这里需要将维度恢复成256，以便后续添加residual_block时通道数一致
#         # 将通道数恢复成256
#         # x = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)  # 需要把x变成Tensor类型，再连接residual_block
#
#         return self.model(x)  # 输出(1, 3, 256, 256)
#
#
# ##############################
# #        Discriminator
# ##############################
# class Discriminator_1(nn.Module):
#     def __init__(self, input_shape):
#         super(Discriminator_1, self).__init__()
#
#         channels, height, width = input_shape  # input_shape:(3， 256， 256)
#
#         # Calculate output shape of image discriminator (PatchGAN)
#         self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)  # output_shape = (1, 16, 16)
#         '''？？？？？？？？？？？？？？？输入图像是3维的，输出图像是1维的？？？？？？？？？？？？？？？？？？？？？
#         patchGAN有关，这是判别器。不是生成器！！！output不是融合图像，而是判别器的输出结果（矩阵）      '''
#
#         def discriminator_block(in_filters, out_filters, normalize=True):  # 鉴别器块儿
#             """Returns downsampling layers of each discriminator block"""
#             layers = [
#                 nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]  # layer += [conv + norm + relu]
#             if normalize:  # 每次卷积尺寸会缩小一半，共卷积了4次
#                 layers.append(nn.InstanceNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(
#             *discriminator_block(channels, 64, normalize=False),  # layer += [conv(3, 64) + relu]
#             *discriminator_block(64, 128),  # layer += [conv(64, 128) + norm + relu]
#             *discriminator_block(128, 256),  # layer += [conv(128, 256) + norm + relu]
#             *discriminator_block(256, 512),  # layer += [conv(256, 512) + norm + relu]
#             nn.ZeroPad2d((1, 0, 1, 0)),  # layer += [pad]  左、右、上、下 分别以0进行填充 1,0,1,0行（列）
#             # 这里就是左和上各填充一（列）行0像素
#             nn.Conv2d(512, 1, 4, padding=1)  # layer += [conv(512, 1)]
#         )
#
#     def forward(self, img):  # 输入(1, 3, 256, 256)
#         # print(self.model(img))  # # 经验证，确实输出了16*16的矩阵 但是在什么地方进行了取平均值操作呢？也就是说输出最后的真假概率值？
#         return self.model(img)  # 输出(1, 1, 16, 16)  输出结果为(batchsize=1, channels=1, H=16, W=16)的矩阵，表示patchGAN
