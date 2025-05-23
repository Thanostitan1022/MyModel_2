# 导入所需的库
import os
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_list = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_list[idx])
        image = Image.open(img_name).convert('L')  # Convert to grayscale image
        if self.transform:
            image = self.transform(image)
        return image


# 设置数据集文件夹路径
data_dir = 'dataset_dwi/images/'

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 创建自定义数据集实例
dataset = CustomDataset(data_dir, transform=transform)


# 使用自定义数据集进行训练等操作
# 可以根据需要将custom_dataset传递给DataLoader进行数据加载和训练

# 定义生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return torch.sigmoid(self.main(x))


# 设置训练参数
batch_size = 64
lr = 0.0002
num_epochs = 100

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# dataset = datasets.MNIST(root='dataset_dwi/images/', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化生成器和判别器
G = Generator()

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(G.parameters(), lr=lr)

# 训练生成器
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()

        real = data
        # real, _ = data
        input = Variable(torch.randn(real.size(0), 100))
        output = G(input)

        # 对生成器输出进行sigmoid操作
        output = output.view(-1, 784)
        loss = criterion(output, real.view(-1, 784))

        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step[%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(dataset) // batch_size, loss.data.item()))

# 生成仿造图像并保存在指定文件夹
output_dir = 'output/model_3_test/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with torch.no_grad():
    fake = G(torch.randn(64, 100))
    for i in range(len(fake)):
        save_image(fake[i].view(1, 28, 28), f'{output_dir}fake_image_{i}.png')
