import cv2
import os


def crop_images(input_folder, output_folder, crop_size, stride):
    image_files = [file for file in os.listdir(input_folder) if file.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)

        height, width, _ = image.shape
        num_rows = (height - crop_size) // stride + 1
        num_cols = (width - crop_size) // stride + 1

        for row in range(num_rows):
            for col in range(num_cols):
                left = col * stride
                upper = row * stride
                right = left + crop_size
                lower = upper + crop_size

                cropped_image = image[upper:lower, left:right]

                output_file = os.path.join(output_folder, f"cropped_{row}_{col}_{image_file}")
                cv2.imwrite(output_file, cropped_image)
                print(f"Saved cropped image {row}_{col} of {image_file} to {output_file}")


# 调用函数进行裁剪和保存
input_folder = 'dataset_CT_MRI/trainB_1'  # 输入文件夹的路径
output_folder = 'dataset_CT_MRI/trainB'  # 输出文件夹的路径
crop_size = 128  # 子图像的尺寸
stride = 64  # 步幅

crop_images(input_folder, output_folder, crop_size, stride)

# import cv2
# import numpy as np
# import os
# import random
#
#
# def crop_and_save_images(input_folder, crop_size, num_crops, output_folder):
#     image_files = [file for file in os.listdir(input_folder) if file.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
#
#     for image_file in image_files:
#         image_path = os.path.join(input_folder, image_file)
#         image = cv2.imread(image_path)
#         height, width, _ = image.shape
#         crop_width, crop_height = crop_size
#
#         for i in range(num_crops):
#             left = random.randint(0, width - crop_width)
#             upper = random.randint(0, height - crop_height)
#             right = left + crop_width
#             lower = upper + crop_height
#
#             cropped_image = image[upper:lower, left:right]
#             output_file = os.path.join(output_folder, f"cropped_{i + 1}_{image_file}")
#             cv2.imwrite(output_file, cropped_image)
#             if i == num_crops - 1:
#                 print(f"Saved cropped image {i + 1} of {image_file} to {output_file}")
#
#
# # 调用函数进行剪裁和保存
# input_folder = 'dataset_2/trainB_2'  # 输入文件夹的路径
# crop_size = (128, 128)  # 每个子块的尺寸
# num_crops = 5  # 剪裁的子块数量
# output_folder = 'dataset_2/trainB'  # 输出文件夹的路径
#
# crop_and_save_images(input_folder, crop_size, num_crops, output_folder)
