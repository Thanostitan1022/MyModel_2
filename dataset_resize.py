import os
from PIL import Image

# 原始图像文件夹路径
input_folder = 'dataset_256/CT_MRI/mask'

# 调整大小后的图像保存文件夹路径
output_folder = 'dataset_256/CT_MRI/mask'

# 确保保存文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历原始图像文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 构建输入和输出图像的完整路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 打开原始图像
        original_image = Image.open(input_path)

        # 调整大小
        resized_image = original_image.resize((256, 256))

        # 保存调整大小后的图像
        resized_image.save(output_path)
