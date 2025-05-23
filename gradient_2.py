import cv2 as cv
import torch


# Sobel算子
def edge_extraction_2(x):  # 传入一个tensor x
    x = x.cpu().numpy()
    print(x.shape)
    print(type(x))
    grad_x = cv.Sobel(x, cv.CV_32F, 1, 0)  # 对x求一阶导    # 错误点在cv.Sobel()函数没法对batches个图像进行处理
    grad_y = cv.Sobel(x, cv.CV_32F, 0, 1)  # 对y求一阶导
    grad_x = cv.convertScaleAbs(grad_x)  # 用convertScaleAbs()函数将其转回原来的uint8形式
    grad_y = cv.convertScaleAbs(grad_y)
    grad_xy = cv.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)  # 图片融合
    grad_xy = torch.tensor(grad_xy)
    return grad_xy

