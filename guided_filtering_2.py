import cv2
import numpy as np


def guideFilter(I, p, winSize, eps, s):
    # 输入图像的高、宽
    h, w = I.shape[:2]

    # 缩小图像
    size = (int(round(w * s)), int(round(h * s)))
    small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    small_p = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)

    # 缩小滑动窗口
    X = winSize[0]
    small_winSize = (int(round(X * s)), int(round(X * s)))

    # I的均值平滑 p的均值平滑
    mean_small_I = cv2.blur(small_I, small_winSize)
    mean_small_p = cv2.blur(small_p, small_winSize)

    # I*I和I*p的均值平滑
    mean_small_II = cv2.blur(small_I * small_I, small_winSize)
    mean_small_Ip = cv2.blur(small_I * small_p, small_winSize)

    # 方差、协方差
    var_small_I = mean_small_II - mean_small_I * mean_small_I
    cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p

    small_a = cov_small_Ip / (var_small_I + eps)
    small_b = mean_small_p - small_a * mean_small_I

    # 对a、b进行均值平滑
    mean_small_a = cv2.blur(small_a, small_winSize)
    mean_small_b = cv2.blur(small_b, small_winSize)

    # 放大
    size1 = (w, h)
    mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)

    q = mean_a * I + mean_b

    return q


if __name__ == '__main__':
    eps = 0.01
    winSize = (3, 3)  # 类似卷积核（数字越大，磨皮效果越好）
    image = cv2.imread(r'dataset_1/testB/1.bmp', cv2.IMREAD_ANYCOLOR)
    image = cv2.resize(image, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
    I = image / 255.0  # 将图像归一化
    p = I
    s = 3  # 步长
    guideFilter_img = guideFilter(I, p, winSize, eps, s)

    # 保存导向滤波结果
    guideFilter_img = guideFilter_img * 255  # (0,1)->(0,255)
    guideFilter_img[guideFilter_img > 255] = 255  # 防止像素溢出
    guideFilter_img = np.round(guideFilter_img)
    guideFilter_img = guideFilter_img.astype(np.uint8)
    cv2.imshow("image", image)
    cv2.imshow("winSize_16", guideFilter_img)

    image_sub = image - guideFilter_img
    # image_sub = cv2.normalize(image_sub, image_sub, 1, 0, cv2.NORM_MINMAX)
    cv2.imshow("image_sub", image_sub)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

