import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets import ImageDataset
from model_5 import StrokeDetectionModel
import warnings
import argparse
import torch.optim as optim
from utils import *
from Pre import Preprocess
from GF_detail import detail_gf
from Intensity import intensity_extract
from datasets import ImageDataset
# from utils import LambdaLR, ReplayBuffer
# torch.set_printoptions(profile="full")

# 超参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='', help='root directory of the dataset')
parser.add_argument('--channels', type=int, default=1, help='number of channels of input data')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of channels of output data')
parser.add_argument('--size', type=int, default=64, help='size of the data (squared assumed)')
parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument('--model', type=str, default='', help='model checkpoint file')

opt_test = parser.parse_args()
print("opt_test", opt_test)


def test():
    print("开始执行test_2Dense.py")
    input_shape = (opt_test.channels, opt_test.size, opt_test.size)
    netG_2 = StrokeDetectionModel(input_shape)
    # 使用cuda
    if opt_test.cuda:
        netG_2 = netG_2.cuda()

    netG_2.load_state_dict(torch.load(opt_test.MyModel))
    netG_2.eval()
    transforms_ = [
        # transforms.Resize([opt_test.img_height, opt_test.img_width]),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transforms.Normalize(0.5, 0.5)]
    dataloader = DataLoader(ImageDataset(opt_test.dataroot, transforms_=transforms_, mode='test'),
                            batch_size=opt_test.batchSize, shuffle=False, num_workers=0)

    for i, batch in enumerate(dataloader):
        torch.cuda.empty_cache()
        img_A = Variable(batch["A"]).cuda()
        img_A = 0.5 * img_A.data + 0.50
        img_F_G2 = netG_2(img_A)
        # img_F_G2 = netG_2(img_F, img_B, blur_img_F, blur_img_B)
        if i == 1:
            print('\rimg_F_G2' + str(type(img_F_G2)))
            print(img_F_G2)
        # 保存图片
        save_image(img_F_G2, 'output/%04d.png' % (i + 1))  # 输出图片的位置
        sys.stdout.write('\rprocessing (%04d)-th image...' % (i + 1))
    print("测试完成")


if __name__ == '__main__':
    test()
