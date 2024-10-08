import os
import random
import shutil

# 指定目录路径
directory = r""

# 指定训练集和测试集目录路径
train_dir = r""
test_dir = r""

# 创建训练集和测试集目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 获取目录中的所有图片文件（包括子目录中的图片文件）
image_files = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".JPG") or file.endswith(".jpg") or file.endswith(".png"):
            image_files.append(os.path.join(root, file))

# 计算划分的索引
total_files = len(image_files)
train_size = int(0.7 * total_files)
test_size = total_files - train_size

# 随机打乱图片文件列表的顺序
random.shuffle(image_files)

# 将图片文件分配到训练集和测试集
train_files = image_files[:train_size]
test_files = image_files[train_size:]

# 将训练集图片复制到训练集目录
for file in train_files:
    file_name = os.path.basename(file)
    dst_file = os.path.join(train_dir, file_name)
    shutil.copy2(file, dst_file)

# 将测试集图片复制到测试集目录
for file in test_files:
    file_name = os.path.basename(file)
    dst_file = os.path.join(test_dir, file_name)
    shutil.copy2(file, dst_file)

print("图片划分完成！")
