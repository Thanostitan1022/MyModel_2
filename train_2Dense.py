#######################################################################
############################  train.py  ###################################

# 此训练生成器1为普通GAN，生成器2为密集连接GAN，且输入为三个单通道（可将光，红外，中间结果）
import os
# import glob
# import random
import torch
import itertools
# import datetime
# import time
import sys
import argparse
# import numpy as np
# import torch.nn.functional as F
# import torch.nn as nn
import torchvision.transforms as transforms
# from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
# from PIL import Image
from torchvision.utils import save_image, make_grid
# import SSIM
# from SSIM import ssim
# import gradient
from datasets import ImageDataset
from gradient_1 import edge_extraction_1
from gradient_2 import edge_extraction_2
# from models_7_G1 import GeneratorResNet_1, Discriminator_1, weights_init_normal
from models_2Dense import GeneratorResNet_2, Discriminator_2, weights_init_normal, Discriminator_detail
from gradient import edge_extraction
from utils import LambdaLR, ReplayBuffer
from GF_blur import blur_gf
from GF_detail import detail_gf
from Intensity import intensity_extract
from OT_loss import OT_similarity

device = torch.device('cuda')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

# 超参数配置
parser = argparse.ArgumentParser()
# parser.add_argument("--start_epoch", type=int, default=0, help="epoch to start training from last time")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=80, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="dataset_adc_dwi",  # 训练集目录 以及 模型保存在什么位置
                    help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")  # 0.0003
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")  # 即学习率的衰减速度
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")  # 即学习率的衰减速度
parser.add_argument("--decay_epoch", type=int, default=3, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=64, help="size of image height")
parser.add_argument("--img_width", type=int, default=64, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")  # 原通道是3！！！！！！
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model checkpoints")  # 每多少轮保存一次模型
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
parser.add_argument('--generator', type=str, default='save/best/MyGenerator_40.pth',  # 使用保存的模型目录
                    help='generator checkpoint file')
opt_train = parser.parse_args()
# opt = parser.parse_args(args=[])                 ## 在colab中运行时，换为此行
print("opt_train", opt_train)

# 创建文件夹
# os.makedirs("images/%s" % opt_train.dataset_name, exist_ok=True)  # images 文件夹保存训练时每100个batch的转换效果图
os.makedirs("save/%s" % opt_train.dataset_name, exist_ok=True)  # save 文件夹用来存放模型参数数据

# input_shape:(3, 256, 256)
input_shape = (opt_train.channels, opt_train.img_height, opt_train.img_width)  # 这个变量还可能会用到，比如可以作为判别器的输入

input_shape_A = (opt_train.channels, opt_train.img_height, opt_train.img_width)
input_shape_B = (opt_train.channels, opt_train.img_height, opt_train.img_width)

# 两个生成器
# MyGenerator = GeneratorResNet_1(input_shape)
MyGenerator_2 = GeneratorResNet_2(input_shape)

# 两个判别器
# MyDiscriminator = Discriminator_1(input_shape)
MyDiscriminator_2 = Discriminator_2(input_shape)
MyDiscriminator_detail = Discriminator_detail(input_shape)

'''
 损失函数LOSS  
 预设：
 LG = L_adv + lamda1 * L_con                                    lamda1=100
 L_adv = L2Loss(D(F),c)  c指一个标签(c是生成器希望鉴别器相信虚假数据的值)
 L_con = (L2Loss(F,Ir) + lambda2 * L2Loss(F,Iv)) / H*W          lamda2=8    
 '''
# 损失函数

# MES 二分类的交叉熵
# L1loss 相比于L2 Loss保边缘
loss1 = torch.nn.L1Loss()  # 表示融合图像F和源图像img_B的L1Loss
loss2 = torch.nn.MSELoss()  # L2Loss

# criterion_GAN = torch.nn.MSELoss()  # 即L2Loss 均方误差          表示 风格A 生成 风格B 的Loss
# criterion_cycle = torch.nn.L1Loss()  # 即MAELoss 平均绝对值误差   表示 最后生成的风格A 和 原图A 的差异Loss
# criterion_identity = torch.nn.L1Loss()  # 表示 风格A 生成 风格A 的Loss

# 如果有显卡，都在cuda模式中运行
if torch.cuda.is_available():
    # MyGenerator = MyGenerator.cuda()
    # MyDiscriminator = MyDiscriminator.cuda()
    MyGenerator_2 = MyGenerator_2.cuda()
    MyDiscriminator_2 = MyDiscriminator_2.cuda()
    MyDiscriminator_detail = MyDiscriminator_detail.cuda()
    loss1.cuda()  # 损失函数的运行也要 cuda()
    loss2.cuda()

# 如果epoch == 0，初始化模型参数; 如果epoch == n, 载入训练到第n轮的预训练模型
# print("Generator_1,Discriminator_1开始加载模型参数！")
# MyGenerator.load_state_dict(torch.load("save/best/MyGenerator_40.pth"))
# # MyDiscriminator.load_state_dict(torch.load("save/best/MyDiscriminator_40.pth"))
# MyDiscriminator.apply(weights_init_normal)
# print("Generator_1,Discriminator_1加载model成功!")

# print("MyGenerator_1初始化模型参数!")
# MyGenerator.apply(weights_init_normal)
# MyDiscriminator.apply(weights_init_normal)
# print("MyGenerator_1初始化模型成功!")

if opt_train.epoch != 0:
    # G1,D1载入训练到第n轮的预训练模型
    # print("MyGenerator_1开始加载 G1 model_%d" % opt_train.epoch)
    # MyGenerator.load_state_dict(
    #     torch.load("save/%s/G1/MyGenerator_%d.pth" % (opt_train.dataset_name, opt_train.epoch)))
    # MyDiscriminator.load_state_dict(
    #     (torch.load("save/%s/G1/MyDiscriminator_%d.pth" % (opt_train.dataset_name, opt_train.epoch))))
    # print("MyGenerator_1加载 G1 model_%d成功!" % opt_train.epoch)

    # G2,D2载入训练到第n轮的预训练模型
    print("MyGenerator_2开始加载 G2 model_%d" % opt_train.epoch)
    MyGenerator_2.load_state_dict(
        torch.load("save/%s/G2/MyGenerator_2_%d.pth" % (opt_train.dataset_name, opt_train.epoch)))
    MyDiscriminator_2.load_state_dict(
        (torch.load("save/%s/G2/MyDiscriminator_2_%d.pth" % (opt_train.dataset_name, opt_train.epoch))))
    print("MyGenerator_2加载 G2 model_%d成功!" % opt_train.epoch)
    # print(MyGenerator.state_dict())

else:
    #  初始化模型参数
    # print("MyGenerator_1初始化模型参数!")
    # MyGenerator.apply(weights_init_normal)
    # MyDiscriminator.apply(weights_init_normal)
    # print("MyGenerator_1初始化模型成功!")
    # 初始化模型参数
    print("MyGenerator_2初始化模型参数!")
    MyGenerator_2.apply(weights_init_normal)
    MyDiscriminator_2.apply(weights_init_normal)
    MyDiscriminator_detail.apply(weights_init_normal)
    print("MyGenerator_2初始化模型成功!")

# 定义优化函数,优化函数的学习率为0.0003
# optimizer_MyGenerator_12 = torch.optim.Adam(
#     itertools.chain(MyGenerator.parameters(), MyGenerator_2.parameters()),
#     lr=opt_train.lr, betas=(opt_train.b1, opt_train.b2)
# )
optimizer_MyGenerator_12 = torch.optim.Adam(MyGenerator_2.parameters(), lr=opt_train.lr,
                                            betas=(opt_train.b1, opt_train.b2))
# optimizer_MyDiscriminator = torch.optim.Adam(MyDiscriminator.parameters(), lr=opt_train.lr,
#                                              betas=(opt_train.b1, opt_train.b2))
optimizer_MyDiscriminator_2 = torch.optim.Adam(MyDiscriminator_2.parameters(), lr=opt_train.lr,
                                               betas=(opt_train.b1, opt_train.b2))
optimizer_MyDiscriminator_detail = torch.optim.Adam(MyDiscriminator_detail.parameters(), lr=opt_train.lr,
                                                    betas=(opt_train.b1, opt_train.b2))

# 学习率更新进程
lr_scheduler_MyGenerator_12 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_MyGenerator_12, lr_lambda=LambdaLR(opt_train.n_epochs, opt_train.epoch, opt_train.decay_epoch).step
)
# lr_scheduler_MyDiscriminator = torch.optim.lr_scheduler.LambdaLR(
#     optimizer_MyDiscriminator, lr_lambda=LambdaLR(opt_train.n_epochs, opt_train.epoch, opt_train.decay_epoch).step
# )
lr_scheduler_MyDiscriminator_2 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_MyDiscriminator_2, lr_lambda=LambdaLR(opt_train.n_epochs, opt_train.epoch, opt_train.decay_epoch).step
)
lr_scheduler_MyDiscriminator_detail = torch.optim.lr_scheduler.LambdaLR(
    optimizer_MyDiscriminator_detail,
    lr_lambda=LambdaLR(opt_train.n_epochs, opt_train.epoch, opt_train.decay_epoch).step
)

# 图像 transformations  图像裁剪和正则化
transforms_ = [
    # transforms.Resize(int(opt_train.img_height * 1.12)),  # 图片放大1.12倍
    # transforms.RandomCrop((opt_train.img_height, opt_train.img_width)),  # 随机裁剪成原来的大小
    # transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.Resize([opt_train.img_height, opt_train.img_width]),
    transforms.ToTensor(),  # 变为Tensor数据
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 正则化 归一化到(-1,1) 两个参数是 mean 和 std
    transforms.Normalize(0.5, 0.5),  # 正则化 归一化到(-1,1) 两个参数是 mean 和 std
]

# Training data loader
dataloader = DataLoader(
    ImageDataset(opt_train.dataset_name, transforms_=transforms_, unaligned=False),
    # "./dataset" , unaligned:设置非对其数据
    batch_size=opt_train.batch_size,
    shuffle=True,
    num_workers=0,  # num_workers 大了找batch找得快，费内存， 小了找batch找的慢，默认0就是不额外找batch
)


def train():
    # ----------
    #  Training
    # ----------
    print("开始执行train_2Dense.py")
    for epoch in range(opt_train.epoch, opt_train.n_epochs + 1):  # for epoch in (0, 50)
        torch.cuda.empty_cache()
        for i, batch in enumerate(
                dataloader):  # batch is a dict, batch['A']:(1, 3, 256, 256), batch['B']:(1, 3, 256, 256)
            #       print('here is %d' % i)
            # 读取数据集中的真图片
            # 将tensor变成Variable放入计算图中，tensor变成Variable之后才能进行反向传播求梯度
            # 在这取到了训练所用的两个源图像A和B  img_A 和 img_B
            img_A = Variable(batch["A"]).cuda()  # 获取源图像A
            img_B = Variable(batch["B"]).cuda()  # 获取源图像B

            # print(img_A)  # 此时的img_A是有负数的
            img_A = 0.5 * img_A.data + 0.50
            img_B = 0.5 * img_B.data + 0.50
            # print("img_A.shape:" + str(img_A.shape))  # [batch_size,3,256,256] 对的
            # print("img_B.shape:" + str(img_B.shape))  # [batch_size,3,256,256] 对的
            #  保存img_A和img_B
            if i % 500 == 0:
                save_image(img_A, 'dataset_check/dataA/%02d_%04d.png' % (epoch, (i + 1)))
                save_image(img_B, 'dataset_check/dataB/%02d_%04d.png' % (epoch, (i + 1)))

            # 全真，全假的标签
            valid_label_1 = Variable(torch.ones((img_A.size(0), *MyDiscriminator_2.output_shape_1)),
                                     requires_grad=False).cuda()
            # size(0)表示第0维的数据 D_A.output_shape 表示输出的16*16的一维矩阵
            # 定义真实的图片label为1 ones((1, 1, 16, 16))
            # 其中第一个1是real_A.size(0), 1,16,16是D_A的output_shape 把这四个维度的元素全部置为1,表示全真标签
            fake_label_1 = Variable(torch.zeros((img_A.size(0), *MyDiscriminator_2.output_shape_1)),
                                    requires_grad=False).cuda()  # 定义假的图片的label为0 zeros((1, 1, 8, 8))

            #  64batch的通道标签
            valid_label_2 = Variable(torch.ones((img_A.size(0), *MyDiscriminator_2.output_shape_2)),
                                     requires_grad=False).cuda()
            # size(0)表示第0维的数据 D_A.output_shape 表示输出的16*16的一维矩阵
            # 定义真实的图片label为1 ones((1, 1, 16, 16))
            # 其中第一个1是real_A.size(0), 1,16,16是D_A的output_shape 把这四个维度的元素全部置为1,表示全真标签
            fake_label_2 = Variable(torch.zeros((img_A.size(0), *MyDiscriminator_2.output_shape_2)),
                                    requires_grad=False).cuda()  # 定义假的图片的label为0 zeros((1, 1, 16, 16))

            # valid_label_3表示细节判别器的真假标签
            valid_label_3 = Variable(torch.ones((img_A.size(0), *MyDiscriminator_detail.output_shape_1)),
                                     requires_grad=False).cuda()
            fake_label_3 = Variable(torch.zeros((img_A.size(0), *MyDiscriminator_detail.output_shape_1)),
                                    requires_grad=False).cuda()  # 定义假的图片的label为0 zeros((1, 1, 8, 8))
            # -----------------
            #  Train Generator
            # 原理：目的是希望生成的假的融合图片被判别器判断为真的图片，
            # 在此过程中，   将判别器固定，   将假的融合图片传入判别器的结果与真实的label对应，
            # 反向传播更新的参数是生成网络里面的参数，
            # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的, 这样就达到了对抗的目的
            # -----------------
            # MyGenerator.train()  # model.train() 是指将模式调为训练模式,调整dropout层、BN层等
            MyGenerator_2.train()
            # MyDiscriminator.train()    # 因为判别器需要固定，所以这里MyDiscriminator不需要train!!!!!

            # img_F = MyGenerator(img_A, img_B)  # 生成器G1生成 img_F  此时imgF在 -1到1之间
            # img_F = 0.5 * MyGenerator(img_A, img_B).data + 0.5

            # 在此用guided_filter处理img_A和img_B，保留这两个图像的边缘细节。
            # detail_img_A = detail_gf(img_A).cuda()
            # detail_img_B = detail_gf(img_B).cuda()

            # ##########  test ###########
            # print(type(img_B))
            # print(img_B.shape)
            # print(type(detail_img_B))
            # print(detail_img_B.shape)
            # ########### test ###########

            # 在此用GF处理img_F和img_B，模糊两个图像
            # blur_img_F = blur_gf(img_F)
            # blur_img_B = blur_gf(img_B)
            # img_F = img_A
            img_F_G2 = MyGenerator_2(img_A, img_B)  # 生成器G2生成 img_F_G2
            # 越往后训练越像第二个图像 可能是因为判别器中以第二个图像参数为大块？

            # 保存该img_F查看效果
            if i % 600 == 0:
                # save_image(img_F, 'middle_result/img_F/%02d_%04d.png' % (epoch, (i + 1)))
                save_image(img_F_G2, 'middle_result/image_F_G2/%02d_%04d.png' % (epoch, (i + 1)))
            # img_F = Variable(img_F).cuda()
            '''
            损失函数LOSS
            预设：
            LG = L_adv + lamda1 * L_con + lamda2 * L_cycle

            LG = L_adv + lamda1 * L_con                                    lamda1=100
            L_adv = L2Loss(D(F),c)  c指一个标签(c是生成器希望鉴别器相信虚假数据的值)（好像是全1标签？？？？？？？？？？）
            L_con = (L2Loss(F,Ir) + lambda3 * L1Loss(F,Iv)) / H*W          lamda3=8
            '''

            #  开始 G1 损失
            # loss_adv 对抗损失 即 把假的图片判断为真
            # a = MyDiscriminator(img_F)[0]
            # loss_adv_G1 = loss2(a, valid_label_1)  # 这里原来是loss2
            #
            # # loss_con 内容损失
            # loss_con1_G1 = loss2(img_F, img_A)  # 融合图像F 和 红外图像A 的L2Loss
            # # loss_con1 = 1.0 - ssim(img_F, img_A).cuda()
            # loss_con2_G1 = loss2(img_F, img_B)  # 融合图像F 和 可见光图像B 的L2Loss
            # # 提取F和B的图像梯度
            # flag_edge_extraction = 0  # 该flag决定用哪一种梯度提取方式
            # if flag_edge_extraction == 0:
            #     gradient_B = edge_extraction(img_B)
            #     gradient_F = edge_extraction(img_F)
            # # elif flag_edge_extraction == 1:
            # #     gradient_B = edge_extraction_1(img_B)
            # #     gradient_F = edge_extraction_1(img_F)
            # # elif flag_edge_extraction == 2:
            # #     gradient_B = edge_extraction_2(img_B)
            # #     gradient_F = edge_extraction_2(img_F)
            #
            # # print(gradient_B.shape)
            # # if i % 500 == 0 and flag_edge_extraction != 0:
            # #     save_image(gradient_F, 'dataset_check/gradient_F/F_%02d_%04d.png' % (epoch, (i + 1)))
            # #     save_image(gradient_B, 'dataset_check/gradient_B/B_%02d_%04d.png' % (epoch, (i + 1)))
            #
            # loss_con3_G1 = loss2(gradient_F, gradient_B)  # F和B 梯度图像的L1Loss
            # # loss_con2 = ssim(img_F, img_B)
            # # loss_con = (loss_con1 + 8 * loss_con2) / (opt_train.img_height * opt_train.img_width) * 10000  # 应该用2！！！！
            #
            # loss_con_G1 = (1.5 * loss_con1_G1 + 1.0 * loss_con2_G1 + 0.1 * loss_con3_G1)
            # # 生成器损失 loss_G = loss_adv + lamda1 * loss_con
            # loss_Generator_1 = loss_adv_G1 + 3 * loss_con_G1

            #  开始 G2 损失
            # b = MyDiscriminator_2(img_F_G2)[0]
            # loss_adv_G2 = loss2(b, valid_label_1)

            # 这里的(上边这一行)损失函数应该改为：
            b1 = MyDiscriminator_2(img_F_G2)[0]
            loss_adv_G2_1 = loss2(b1, valid_label_1)

            b2 = MyDiscriminator_2(img_F_G2)[1]
            loss_adv_G2_2 = loss2(b2, valid_label_2)

            b3 = MyDiscriminator_detail(img_F_G2)
            loss_adv_G2_3 = loss2(b3, valid_label_3)

            loss_adv_G2 = (loss_adv_G2_1 + loss_adv_G2_2 + loss_adv_G2_3) / 3.0

            gradient_B = edge_extraction(img_B)

            loss_con1_G2 = loss2(img_F_G2, img_A)  # 融合图像img_F_G2 和 红外图像img_A 的L2损失
            loss_con2_G2 = loss2(img_F_G2, img_B)  # 融合图像img_F_G2 和 可见光图像B 的L2损失
            gradient_F_G2 = edge_extraction(img_F_G2)
            loss_con3_G2 = loss2(gradient_F_G2, gradient_B)  # 融合图像Img_F_G2 和 可见光图像B 的梯度损失

            # 提取最终图像、可见光图像、红外图像的边缘图像  即导向滤波图像
            detail_img_F_G2 = detail_gf(img_F_G2)
            detail_img_B = detail_gf(img_B)
            detail_img_A = detail_gf(img_A)

            loss_con4_G2 = loss2(detail_img_F_G2, detail_img_B)  # 最终图像和可见光图像的detail边缘损失
            loss_con5_G2 = loss2(detail_img_F_G2, detail_img_A)  # 最终图像和中间图像（偏红外）的detail边缘损失

            # 提取红外图像亮度（像素强度）
            intensity_ir = intensity_extract(img_A)
            intensity_img_F_G2 = intensity_extract(img_F_G2)
            loss_con6_G2 = loss2(intensity_img_F_G2, intensity_ir)

            loss_con_G2 = (2.0 * loss_con1_G2 + 1.8 * loss_con2_G2 + 15.0 * loss_con3_G2
                           + 0.01 * loss_con4_G2 + 0.01 * loss_con5_G2 + 0.01 * loss_con6_G2)

            # 先放弃红外亮度损失项
            # loss_con_G2 = (2.0 * loss_con1_G2 + 1.8 * loss_con2_G2 + 15 * loss_con3_G2
            #                + 0.01 * loss_con4_G2 + 0.01 * loss_con5_G2)

            # loss_con_G2 = 1.5 * loss_con1_G2 + loss_con2_G2 + 5 * loss_con3_G2
            loss_Generator_2 = loss_adv_G2 + 3.0 * loss_con_G2

            # 总体Generator损失
            # loss_Generator_12 = (loss_Generator_1 + loss_Generator_2) / 2
            loss_Generator_12 = loss_Generator_2
            '''
            因为grad在 反向传播 的过程中是累加的，也就是说上一次反向传播的结果会对下一次的反向传播的结果造成影响，
            则意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需要把梯度清零。
            '''
            optimizer_MyGenerator_12.zero_grad()  # 在反向传播之前，先将梯度归0
            loss_Generator_12.backward()  # 将误差反向传播
            optimizer_MyGenerator_12.step()  # 更新参数

            # -----------------------
            # Train Discriminator_1
            # 分为两部分：1、真的图像判别为真；2、假的图像判别为假
            # -----------------------

            # loss_real_my = loss2(MyDiscriminator(img_A)[0], valid_label_1)  # 16*16
            # loss_fake_my = loss2(MyDiscriminator(img_F.detach())[0], fake_label_1)
            #
            # loss_real_my_64 = loss2(MyDiscriminator(img_A)[1], valid_label_2)  # 64*64
            # loss_fake_my_64 = loss2(MyDiscriminator(img_F.detach())[1], fake_label_2)
            # # Total loss
            # loss_Discriminator = (loss_real_my + loss_fake_my) / 2 + (loss_real_my_64 + loss_fake_my_64) / 2
            #
            # optimizer_MyDiscriminator.zero_grad()  # 在反向传播之前，先将梯度归0
            # loss_Discriminator.backward()  # 将误差反向传播
            # optimizer_MyDiscriminator.step()  # 更新参数

            # -----------------------
            # Train Discriminator_2
            # 分为两部分：1、真的图像判别为真；2、假的图像判别为假
            # -----------------------

            loss_real_my_2 = loss2(MyDiscriminator_2(img_B)[0], valid_label_1)  # 16*16
            loss_fake_my_2 = loss2(MyDiscriminator_2(img_F_G2.detach())[0], fake_label_1)  # 这里要用detach()!!!不知道为啥

            # # 将5D张量展平为2D张量
            # img_B_flat = img_B.view(img_B.size(0), -1)
            # img_F_G2_flat = img_F_G2.view(img_F_G2.size(0), -1)

            # # 计算最优传输距离
            # distance_F_B = torch.nn.functional.pdist(torch.stack([img_F_G2_flat, img_B_flat]), p=2)
            # # 计算相似度（最小成本）
            # no_similarity_F_B = 1.0 - 1.0 / (1.0 + distance_F_B.item())

            loss_real_my_2_64 = loss2(MyDiscriminator_2(img_A)[1], valid_label_2)  # 64*64
            loss_fake_my_2_64 = loss2(MyDiscriminator_2(img_F_G2.detach())[1], fake_label_2)

            # img_A_flat = img_A.view(img_A.size(0), -1)

            # # 计算最优传输距离
            # distance_F_A = torch.nn.functional.pdist(torch.stack([img_F_G2_flat, img_A_flat]), p=2)
            # # 计算相似度（最小成本）
            # no_similarity_F_A = 1.0 - 1.0 / (1.0 + distance_F_A.item())

            loss_OT_F_B = OT_similarity(img_F_G2, img_B)
            loss_OT_F_A = OT_similarity(img_F_G2, img_A)
            # Total loss
            loss_Discriminator_2 = (loss_real_my_2 + loss_fake_my_2) / 2.0 + (loss_real_my_2_64 + loss_fake_my_2_64) / \
                                   2.0 + (loss_OT_F_A + loss_OT_F_B) / 2.0

            optimizer_MyDiscriminator_2.zero_grad()  # 在反向传播之前，先将梯度归0
            loss_Discriminator_2.backward()  # 将误差反向传播
            optimizer_MyDiscriminator_2.step()  # 更新参数

            # -----------------------
            # Train Discriminator_detail
            # 1.获得红外图像的intensity图像
            # 2.获得可见光图像的guided_filter图像
            # 3.intensity+guided_filter
            # 4.获取融合图像的 int+gf
            # 分为两部分：1、真的图像判别为真；2、假的图像判别为假
            # -----------------------
            ir_int = intensity_extract(img_A).cuda()
            vis_gf = detail_gf(img_B).cuda()
            source_int_gf = torch.add(ir_int, vis_gf).cuda()
            fusion_int = intensity_extract(img_F_G2).cuda()
            fusion_gf = detail_gf(img_F_G2).cuda()
            fusion_int_gf = torch.add(fusion_int, fusion_gf).cuda()

            loss_real_my_detail = loss2(MyDiscriminator_detail(source_int_gf.detach()), valid_label_3)  # 8*8
            loss_fake_my_detail = loss2(MyDiscriminator_detail(fusion_int_gf.detach()),
                                        fake_label_3)  # 这里要用detach()!!!不知道为啥

            # Discriminator_detail loss
            loss_Discriminator_detail = (loss_real_my_detail + loss_fake_my_detail) / 2.0

            optimizer_MyDiscriminator_detail.zero_grad()  # 在反向传播之前，先将梯度归0
            loss_Discriminator_detail.backward()  # 将误差反向传播
            optimizer_MyDiscriminator_detail.step()  # 更新参数

            # Discriminator total loss
            loss_Discriminator_12 = loss_Discriminator_2 + loss_Discriminator_detail
            # ----------------------
            #  打印日志Log Progress
            # ----------------------

            # Print log
            sys.stdout.write(
                # "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                # " \r[Epoch %d/%d] [Batch %d/%d] [D Loss: %f] [G loss: %f] ETA: %s"
                " \r[Epoch %d/%d] [Batch %d/%d] [G_loss: %f] [D_loss:%f] [adv2: %f] [conG2: %f] [dis_d:%f]"
                % (
                    epoch,
                    opt_train.n_epochs,
                    i + 1,
                    len(dataloader),
                    loss_Generator_12.item(),
                    loss_Discriminator_12.item(),
                    # loss_adv_G1.item(),
                    # loss_con_G1.item(),
                    loss_adv_G2.item(),
                    loss_con_G2.item(),
                    loss_Discriminator_detail
                    # loss_Discriminator_2.item(),
                )
            )

        # 更新学习率
        lr_scheduler_MyGenerator_12.step()
        # lr_scheduler_MyDiscriminator.step()
        lr_scheduler_MyDiscriminator_2.step()
        lr_scheduler_MyDiscriminator_detail.step()

        # 每间隔几个epoch保存一次模型
        if opt_train.checkpoint_interval != -1 and epoch % opt_train.checkpoint_interval == 0:
            # Save model checkpoints
            # torch.save(MyGenerator.state_dict(), "save/%s/G1/MyGenerator_%d.pth" % (opt_train.dataset_name, epoch))
            # torch.save(MyDiscriminator.state_dict(),
            #            "save/%s/G1/MyDiscriminator_%d.pth" % (opt_train.dataset_name, epoch))

            torch.save(MyGenerator_2.state_dict(), "save/%s/MyGenerator_2_%d.pth" % (opt_train.dataset_name, epoch))
            torch.save(MyDiscriminator_2.state_dict(),
                       "save/%s/MyDiscriminator_2_%d.pth" % (opt_train.dataset_name, epoch))
            torch.save(MyDiscriminator_detail.state_dict(),
                       "save/%s/MyDiscriminator_detail_%d.pth" % (opt_train.dataset_name, epoch))

            print("\nsave my model_%d finished" % epoch)
