import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt
from einops import rearrange


# 定义参数初始化函数
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0,
                              0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0,
                              0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class PreProcessing(nn.Module):
    def __init__(self):
        super().__init__()
        self.brightness = T.ColorJitter(brightness=0.2)

    def forward(self, x):
        # 模拟导向滤波 + 亮度增强
        blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        enhanced = self.brightness(blurred)
        return enhanced


def PreProcess(img):
    return detail_gf(img) + intensity_extract(img)


class FeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=False):
        super().__init__()
        self.up = up
        if up:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )

    def forward(self, x):
        return self.conv(x)


class WeightAllocation(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 5, padding=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        w1 = self.conv1(x1)
        w2 = self.conv2(x2)
        w_all = self.softmax(torch.cat([w1, w2], dim=1))
        w1, w2 = torch.chunk(w_all, 2, dim=1)
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
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4)

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.norm2(x)
        x = self.mlp(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors=1, num_classes=1):
        super().__init__()
        self.bbox = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.cls = nn.Conv2d(in_channels, num_anchors * num_classes, 1)

    def forward(self, x):
        return self.bbox(x), self.cls(x)


class StrokeDetectionModel(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, num_classes=1):
        super().__init__()
        self.preprocess = PreProcessing()

        self.up1 = FeatureBlock(in_channels, base_channels, up=True)
        self.up2 = FeatureBlock(base_channels, base_channels * 2, up=True)
        self.up3 = FeatureBlock(base_channels * 2, base_channels * 4, up=True)

        self.down1 = FeatureBlock(in_channels, base_channels, up=False)
        self.down2 = FeatureBlock(base_channels, base_channels * 2, up=False)
        self.down3 = FeatureBlock(base_channels * 2, base_channels * 4, up=False)

        self.up1_p = FeatureBlock(in_channels, base_channels, up=True)
        self.up2_p = FeatureBlock(base_channels, base_channels * 2, up=True)
        self.up3_p = FeatureBlock(base_channels * 2, base_channels * 4, up=True)

        self.down1_p = FeatureBlock(in_channels, base_channels, up=False)
        self.down2_p = FeatureBlock(base_channels, base_channels * 2, up=False)
        self.down3_p = FeatureBlock(base_channels * 2, base_channels * 4, up=False)

        self.wa1 = WeightAllocation(base_channels)
        self.wa2 = WeightAllocation(base_channels * 2)
        self.wa3 = WeightAllocation(base_channels * 4)

        # Transformer
        self.trans1 = SwinBlock(base_channels)
        self.trans2 = SwinBlock(base_channels * 2)
        self.trans3 = SwinBlock(base_channels * 4)

        # 检测头
        self.head1 = DetectionHead(base_channels, num_classes=num_classes)
        self.head2 = DetectionHead(base_channels * 2, num_classes=num_classes)
        self.head3 = DetectionHead(base_channels * 4, num_classes=num_classes)

    def forward(self, x):
        pre = self.preprocess(x)

        u1 = self.up1(x);
        u2 = self.up2(u1);
        u3 = self.up3(u2)
        d1 = self.down1(x);
        d2 = self.down2(d1);
        d3 = self.down3(d2)

        u1p = self.up1_p(pre);
        u2p = self.up2_p(u1p);
        u3p = self.up3_p(u2p)
        d1p = self.down1_p(pre);
        d2p = self.down2_p(d1p);
        d3p = self.down3_p(d2p)

        f1 = self.wa1(d1, d1p)
        f2 = self.wa2(d2, d2p)
        f3 = self.wa3(d3, d3p)

        t1 = self.trans1(f1)
        t2 = self.trans2(f2)
        t3 = self.trans3(f3)

        bbox1, cls1 = self.head1(t1)
        bbox2, cls2 = self.head2(t2)
        bbox3, cls3 = self.head3(t3)

        return [(bbox1, cls1), (bbox2, cls2), (bbox3, cls3)]


def decode_predictions(preds, conf_thresh=0.5):
    boxes = []
    for bbox, cls in preds:
        B, _, H, W = cls.shape
        cls = torch.sigmoid(cls)
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    score = cls[b, 0, i, j]
                    if score > conf_thresh:
                        dx, dy, dw, dh = bbox[b, :, i, j]
                        x_center = j + dx.item()
                        y_center = i + dy.item()
                        w = torch.exp(dw).item()
                        h = torch.exp(dh).item()
                        x1 = x_center - w / 2
                        y1 = y_center - h / 2
                        x2 = x_center + w / 2
                        y2 = y_center + h / 2
                        boxes.append([x1, y1, x2, y2, score.item()])
    return boxes


def draw_boxes(image, boxes):
    img = image.squeeze().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for box in boxes:
        x1, y1, x2, y2, score = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, f"{score:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.imshow(img)
    plt.axis('off')
    plt.title("Detected Stroke Lesions")
    plt.show()


model = StrokeDetectionModel()
model.eval()

input_image = torch.randn(1, 1, 256, 256)

with torch.no_grad():
    preds = model(input_image)
    boxes = decode_predictions(preds, conf_thresh=0.5)
    draw_boxes(input_image, boxes)
