import torch
import torch.nn as nn
import torchvision.transforms as T
from einops import rearrange


class PreProcessing(nn.Module):
    def __init__(self):
        super(PreProcessing, self).__init__()
        self.brightness = T.ColorJitter(brightness=0.2)

    def forward(self, x):
        # x: [B, 1, H, W]
        # 简化导向滤波为高斯滤波效果
        blurred = nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        enhanced = self.brightness(blurred)
        return enhanced


class FeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=False):
        super(FeatureBlock, self).__init__()
        self.up = up
        if up:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )

    def forward(self, x):
        return self.conv(x)


class WeightAllocation(nn.Module):
    def __init__(self, in_channels):
        super(WeightAllocation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        w1 = self.conv1(x1)
        w2 = self.conv2(x2)
        weights = self.softmax(torch.cat([w1, w2], dim=1))
        w1, w2 = torch.chunk(weights, 2, dim=1)
        return w1 * x1 + w2 * x2


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.fc(x)


class SwinBlock(nn.Module):
    def __init__(self, dim, window_size=7):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.norm2(x)
        x = self.mlp(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class StrokeDetectionModel(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.preprocess = PreProcessing()

        # 特征金字塔（原图 + 预处理图）
        self.up1 = FeatureBlock(in_channels, base_channels, up=True)
        self.up2 = FeatureBlock(base_channels, base_channels * 2, up=True)
        self.up3 = FeatureBlock(base_channels * 2, base_channels * 4, up=True)

        self.down1 = FeatureBlock(in_channels, base_channels, up=False)
        self.down2 = FeatureBlock(base_channels, base_channels * 2, up=False)
        self.down3 = FeatureBlock(base_channels * 2, base_channels * 4, up=False)

        # WA模块
        self.wa1 = WeightAllocation(base_channels)
        self.wa2 = WeightAllocation(base_channels * 2)
        self.wa3 = WeightAllocation(base_channels * 4)

        # Transformer模块
        self.trans1 = SwinBlock(base_channels)
        self.trans2 = SwinBlock(base_channels * 2)
        self.trans3 = SwinBlock(base_channels * 4)

        # 输出层
        self.predict1 = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.predict2 = nn.Conv2d(base_channels * 2, 1, kernel_size=1)
        self.predict3 = nn.Conv2d(base_channels * 4, 1, kernel_size=1)

    def forward(self, x):
        pre_x = self.preprocess(x)

        # 原图金字塔
        u1 = self.up1(x)
        u2 = self.up2(u1)
        u3 = self.up3(u2)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        # 预处理图金字塔
        up1_p = self.up1(pre_x)
        up2_p = self.up2(up1_p)
        up3_p = self.up3(up2_p)

        down1_p = self.down1(pre_x)
        down2_p = self.down2(down1_p)
        down3_p = self.down3(down2_p)

        # 特征融合（WA模块）
        f1 = self.wa1(d1, down1_p)
        f2 = self.wa2(d2, down2_p)
        f3 = self.wa3(d3, down3_p)

        # Transformer处理
        t1 = self.trans1(f1)
        t2 = self.trans2(f2)
        t3 = self.trans3(f3)

        # 预测输出
        out1 = self.predict1(t1)
        out2 = self.predict2(t2)
        out3 = self.predict3(t3)

        return out1, out2, out3

class StrokeDetectionModel(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, num_classes=1):
        super().__init__()
        self.preprocess = PreProcessing()

        # 特征金字塔（原图 + 预处理图）
        self.up1 = FeatureBlock(in_channels, base_channels, up=True)
        self.up2 = FeatureBlock(base_channels, base_channels * 2, up=True)
        self.up3 = FeatureBlock(base_channels * 2, base_channels * 4, up=True)

        self.down1 = FeatureBlock(in_channels, base_channels, up=False)
        self.down2 = FeatureBlock(base_channels, base_channels * 2, up=False)
        self.down3 = FeatureBlock(base_channels * 2, base_channels * 4, up=False)

        # WA模块
        self.wa1 = WeightAllocation(base_channels)
        self.wa2 = WeightAllocation(base_channels * 2)
        self.wa3 = WeightAllocation(base_channels * 4)

        # # Transformer模块
        # self.trans1 = SwinBlock(base_channels)
        # self.trans2 = SwinBlock(base_channels * 2)
        # self.trans3 = SwinBlock(base_channels * 4)

        # # 输出层
        # self.predict1 = nn.Conv2d(base_channels, 1, kernel_size=1)
        # self.predict2 = nn.Conv2d(base_channels * 2, 1, kernel_size=1)
        # self.predict3 = nn.Conv2d(base_channels * 4, 1, kernel_size=1)

        # Transformer输出
        self.trans1 = SwinBlock(base_channels)
        self.trans2 = SwinBlock(base_channels * 2)
        self.trans3 = SwinBlock(base_channels * 4)

        # 检测头
        self.head1 = DetectionHead(base_channels, num_classes=num_classes)
        self.head2 = DetectionHead(base_channels * 2, num_classes=num_classes)
        self.head3 = DetectionHead(base_channels * 4, num_classes=num_classes)

    def forward(self, x):
        pre_x = self.preprocess(x)

        # 原图金字塔
        u1 = self.up1(x)
        u2 = self.up2(u1)
        u3 = self.up3(u2)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        # 预处理图金字塔
        up1_p = self.up1(pre_x)
        up2_p = self.up2(up1_p)
        up3_p = self.up3(up2_p)

        down1_p = self.down1(pre_x)
        down2_p = self.down2(down1_p)
        down3_p = self.down3(down2_p)

        # 特征融合（WA模块）
        f1 = self.wa1(d1, down1_p)
        f2 = self.wa2(d2, down2_p)
        f3 = self.wa3(d3, down3_p)

        # # Transformer处理
        # t1 = self.trans1(f1)
        # t2 = self.trans2(f2)
        # t3 = self.trans3(f3)

        # # 预测输出
        # out1 = self.predict1(t1)
        # out2 = self.predict2(t2)
        # out3 = self.predict3(t3)

        # 特征融合（WA）与 Transformer
        t1 = self.trans1(f1)
        t2 = self.trans2(f2)
        t3 = self.trans3(f3)

        # 检测输出
        bbox1, cls1 = self.head1(t1)
        bbox2, cls2 = self.head2(t2)
        bbox3, cls3 = self.head3(t3)

        # 输出所有尺度的预测
        return [(bbox1, cls1), (bbox2, cls2), (bbox3, cls3)]


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors=3, num_classes=1):
        super().__init__()
        self.num_anchors = num_anchors
        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)  # [x, y, w, h]
        self.cls_score = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)

    def forward(self, x):
        bbox = self.bbox_reg(x)  # [B, A*4, H, W]
        cls = self.cls_score(x)  # [B, A*C, H, W]
        return bbox, cls


model = StrokeDetectionModel()
x = torch.randn(1, 1, 256, 256)  # 模拟输入图像
out1, out2, out3 = model(x)
print(out1.shape, out2.shape, out3.shape)
