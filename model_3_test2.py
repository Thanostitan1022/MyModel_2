import os
import random
import cv2
import numpy as np

# 指定图像文件夹路径
data_dir = 'dataset_dwi/images/'

# 获取文件夹下所有图像文件名
image_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
# 保存路径
save_path = "output/model_3_test_3/"
# 打乱图像文件顺序
random.shuffle(image_files)


# 定义模糊和噪声处理函数
def apply_blur_noise(image):
    # 模糊处理
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # 添加噪声
    noise = np.random.normal(0, 12, blurred.shape)
    noisy_image = np.clip(blurred + noise, 0, 255).astype(np.uint8)

    return noisy_image


# 处理每张图像
for image_file in image_files:
    image_path = os.path.join(data_dir, image_file)
    image = cv2.imread(image_path)

    if image is not None:
        # 应用模糊和噪声处理
        processed_image = apply_blur_noise(image)

        # 保存处理后的图像

        cv2.imwrite(os.path.join(save_path, 'processed_' + image_file), processed_image)


# # 指定图像文件夹路径
# data_dir = save_path
#
# # 获取文件夹下所有图像文件名
# image_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
#
# # 重命名图像文件
# for i, image_file in enumerate(image_files, start=1):
#     image_path = os.path.join(data_dir, image_file)
#
#     # 构建新的文件名，格式为0001-n
#     new_filename = f'{i:04d}-{image_file}'
#
#     # 构建新的文件路径
#     new_image_path = os.path.join(data_dir, new_filename)
#
#     # 重命名图像文件
#     os.rename(image_path, new_image_path)