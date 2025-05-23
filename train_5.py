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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
# 超参数配置
parser = argparse.ArgumentParser()
parser.add_argument("--start_epoch", type=int, default=0, help="epoch to start training from last time")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=80, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=3, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=64, help="size of image height")
parser.add_argument("--img_width", type=int, default=64, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in model")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
parser.add_argument('--model', type=str, default='', help='model checkpoint file')
opt_train = parser.parse_args()
# opt = parser.parse_args(args=[])
print("opt_train", opt_train)

# 创建文件夹
# os.makedirs("images/%s" % opt_train.dataset_name, exist_ok=True)
os.makedirs("save/%s" % opt_train.dataset_name, exist_ok=True)

# input_shape:(3, 256, 256)
input_shape = (opt_train.channels, opt_train.img_height, opt_train.img_width)

loss1 = torch.nn.L1Loss()  # 表示融合图像F和源图像img_B的L1Loss
loss2 = torch.nn.MSELoss()  # L2Loss

batch_size = 16
num_epochs = 10
learning_rate = 0.001

model = StrokeDetectionModel()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

model.train()

# 定义损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 加载数据集
train_dataset = Dataset("Dataset_ISLES/")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

if opt_train.epoch != 0:
    print("model开始加载model_%d" % opt_train.epoch)
    model.load_state_dict(
        torch.load("save/%s/model_%d.pth" % (opt_train.dataset_name, opt_train.epoch)))
    model.load_state_dict(
        (torch.load("save/%s/model_%d.pth" % (opt_train.dataset_name, opt_train.epoch))))
    print("model加载model_%d成功!" % opt_train.epoch)

else:
    print("model初始化模型参数!")
    model.apply(weights_init_normal)
    print("model初始化模型成功!")

optimizer_MyModel_12 = torch.optim.Adam(model.parameters(), lr=opt_train.lr,
                                        betas=(opt_train.b1, opt_train.b2))

lr_scheduler_MyModel_12 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_MyModel_12, lr_lambda=LambdaLR(opt_train.n_epochs, opt_train.epoch, opt_train.decay_epoch).step
)

transforms_ = [
    # transforms.Resize(int(opt_train.img_height * 1.12)),
    transforms.RandomCrop((opt_train.img_height, opt_train.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.Resize([opt_train.img_height, opt_train.img_width]),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Normalize(0.5, 0.5),
]

dataloader = DataLoader(
    ImageDataset(opt_train.dataset_name, transforms_=transforms_, unaligned=False),
    batch_size=opt_train.batch_size,
    shuffle=True,
    num_workers=0,
)


def train():
    # ----------
    #  Training
    # ----------
    print("开始执行train.py")
    for epoch in range(opt_train.epoch, opt_train.n_epochs + 1):
        torch.cuda.empty_cache()
        for i, batch in enumerate(
                dataloader):
            #       print('here is %d' % i)
            img_A = Variable(batch["A"]).cuda()
            img_B = PreProcess(img_A)
            img_B = Variable(batch["B"]).cuda()

            img_A = 0.5 * img_A.data + 0.50
            img_B = 0.5 * img_B.data + 0.50

            model.train()
            img_F_G2 = model(img_A)

            gradient_B = edge_extraction(img_B)

            loss_con1_G2 = loss2(img_F_G2, img_A)
            loss_con2_G2 = loss2(img_F_G2, img_B)
            gradient_F_G2 = edge_extraction(img_F_G2)
            loss_con3_G2 = loss2(gradient_F_G2, gradient_B)

            detail_img_F_G2 = detail_gf(img_F_G2)
            detail_img_B = detail_gf(img_B)
            detail_img_A = detail_gf(img_A)

            loss_con4_G2 = loss2(detail_img_F_G2, detail_img_B)
            loss_con5_G2 = loss2(detail_img_F_G2, detail_img_A)

            intensity_ir = intensity_extract(img_A)
            intensity_img_F_G2 = intensity_extract(img_F_G2)
            loss_con6_G2 = loss2(intensity_img_F_G2, intensity_ir)

            loss_con_G2 = (2.0 * loss_con1_G2 + 1.8 * loss_con2_G2 + 15.0 * loss_con3_G2
                           + 0.01 * loss_con4_G2 + 0.01 * loss_con5_G2 + 0.01 * loss_con6_G2)
            loss_MyModel_12 = loss_con_G2

            optimizer_MyModel_12.zero_grad()
            loss_MyModel_12.backward()
            optimizer_MyModel_12.step()

            sys.stdout.write(
                # "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                # " \r[Epoch %d/%d] [Batch %d/%d] [D Loss: %f] [G loss: %f] ETA: %s"
                " \r[Epoch %d/%d] [Batch %d/%d] [G_loss: %f]"
                % (
                    epoch,
                    opt_train.n_epochs,
                    i + 1,
                    len(dataloader),
                    loss_MyModel_12.item(),
                )
            )

        # 更新学习率
        lr_scheduler_MyModel_12.step()

        # 每间隔几个epoch保存一次模型
        if opt_train.checkpoint_interval != -1 and epoch % opt_train.checkpoint_interval == 0:
            torch.save(MyGenerator_2.state_dict(), "save/%s/MyModel_%d.pth" % (opt_train.dataset_name, epoch))
            print("\nsave my model_%d finished" % epoch)


if __name__ == '__main__':
    train()
