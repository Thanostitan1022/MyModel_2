# 用于 图像和掩膜 的检测
from PIL import Image, ImageDraw
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

"""
    生成COCO数据格式的json标注文件
    提取mask掩码信息，自动生成目标检测框COCO数据
    注：需要建立文件夹annotations
"""

# 将以下路径替换为您的实际数据集路径
masks_folder = 'D:\\program\\Pycharm\\MyModel_2_test\\dataset_CT\\masks_1\\'
# masks_folder = 'D:\\program\\Pycharm\\MyModel_2_test\\dataset_adc_dwi\\masks_A\\'
# fusion_path = 'D:\\program\\Pycharm\\MyModel_2_test\\output\\CT_MRI\\fusion\\'
# images_folder = os.path.join(dataset_path, 'testB')
images_folder = 'D:\\program\\Pycharm\\MyModel_2_test\\dataset_CT\\images\\'
# images_folder = 'D:\\program\\Pycharm\\MyModel_2_test\\dataset_adc_dwi\\testA\\'
# images_folder = 'D:\\program\\Pycharm\\MyModel_2_test\\output\\enhance_test_2\\'

# masks_folder = os.path.join(dataset_path, 'maskB_fusion')


def extract_coordinates_and_draw_boxes():
    annotations = []  # 存储COCO格式的标注信息
    image_id = 0  # 图像的唯一标识符
    annotations_id = 0  # 掩码的唯一标识符

    # 遍历图像文件夹
    for image_filename in tqdm(os.listdir(images_folder)):
        if image_filename.endswith('.png'):
            image_path = os.path.join(images_folder, image_filename)
            image = Image.open(image_path)
            # print("打开图像文件成功！！")
            # # 检查图像的形状
            # img_shape = image.size
            # print(f"\nimage ID:{image_id}, shape: {img_shape}")
            # # 检查图像的颜色类型
            # img_color_mode = image.mode
            # print(f"image ID:{image_id}, color mode: {img_color_mode}")

            # 提取文件名的前43个字符
            prefix = image_filename[:-4]

            # 遍历掩码文件夹，找到对应的掩码
            for mask_filename in os.listdir(masks_folder):
                if mask_filename.startswith(prefix) and mask_filename.endswith('.png'):
                    mask_path = os.path.join(masks_folder, mask_filename)
                    mask = Image.open(mask_path)
                    # # 检查图像的形状
                    # mask_shape = mask.size
                    # print(f"\nmask ID:{image_id}, shape: {mask_shape}")
                    # # 检查图像的颜色类型
                    # mask_color_mode = mask.mode
                    # print(f"mask ID:{image_id}, color mode: {mask_color_mode}")

                    # 将掩码图像转换为numpy数组
                    mask_np = np.array(mask)
                    # 获取掩码图像的轮廓
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # 对每个轮廓提取矩形框
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        # print("\nboundingRect的xywh", x, y, w, h)
                        bbox = [x, y, w, h]
                        # print("bbox的xywh", x, y, w, h)
                        annotations_id += 1
                        # print("annotations_id", annotations_id)
                        annotations.append({
                            "id": annotations_id,
                            "image_id": image_id,
                            "category_id": 1,  # 假设只有一个类别
                            "bbox": bbox,
                            "area": w * h,
                            "iscrowd": 0
                        })

                        # 在原图和掩码图上绘制矩形框
                        draw_image = ImageDraw.Draw(image)
                        draw_mask = ImageDraw.Draw(mask)
                        draw_image.rectangle([x, y, x + w, y + h], outline="red", width=1)
                        draw_mask.rectangle([x, y, x + w, y + h], outline="red", width=1)

                        # print("draw的xywh", x, y, w, h)
                        #
                        # 使用matplotlib依次展示图像
                        # fig, axs = plt.subplots(1, 1)
                        # axs.imshow(image)
                        # axs.set_title('Image')
                        #
                        # axs[0].imshow(image)
                        # axs[0].set_title('Image')
                        # axs[1].imshow(mask, cmap='gray')
                        # axs[1].set_title('Mask')
                        # plt.tight_layout()
                        # plt.show()
                        plt.imshow(image)
                        plt.axis('off')
                        plt.savefig('output/adc_dwi_fusion_detection/' + image_filename, bbox_inches='tight', pad_inches=0)
                        plt.clf()
                        plt.close()

            image_id += 1

    return annotations


# 函数：将标注信息转换为COCO格式，并保存为JSON文件
def save_to_coco_format(annotations):
    images_info = []
    for i, filename in enumerate(os.listdir(images_folder)):
        if filename.endswith('.png'):
            image_path = os.path.join(images_folder, filename)
            image = Image.open(image_path)
            images_info.append({
                "id": i,
                "file_name": f"images/{filename}",
                "width": image.width,
                "height": image.height
            })

    coco_format = {
        "info": {},
        "licenses": [],
        "images": images_info,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "ISL"}]  # 假设只有一个类别
    }

    with open('annotations/instances_train2017.json', 'w') as f:
        json.dump(coco_format, f)
    # with open('D:\\program\\Pycharm\\MyModel_2_test\\dataset_CT_1/annotations/instances_train2017.json', 'w') as f:
    #     json.dump(coco_format, f)


# 执行函数
annotations = extract_coordinates_and_draw_boxes()
save_to_coco_format(annotations)
