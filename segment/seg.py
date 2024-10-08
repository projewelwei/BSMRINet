import cv2
import os
import pickle
import numpy as np

# 读取点的位置数据
with open("H_W.pickle", "rb") as f:
    points = pickle.load(f)

# 将点的位置数据转换为 numpy 数组
points = np.array(points)

# 读取输入图片
input_image_path = r"output.jpg"
img = cv2.imread(input_image_path)

# 水平翻转图像
img_flipped = cv2.flip(img, 1)

# 顺时针旋转90度
img_rotated = cv2.rotate(img_flipped, cv2.ROTATE_90_COUNTERCLOCKWISE)


if img_rotated is None:
    print("Failed to load input image")
    exit()

# 绘制形状并分割椎体
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)  # 创建保存输出图片的目录

for idx, point_set in enumerate(points):
    # 检查点的顺序，按照指定的顺序连接
    shape = np.array(point_set)
    shape = shape[[0, 1, 3, 2]]  # 调整点的顺序

    # 绘制形状
    cv2.polylines(img_rotated, [shape.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
    # cv2.imshow('Image', img_rotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 分割椎体并保存
    roi = np.zeros_like(img_rotated)  # 创建一个和输入图片大小相同的黑色图像
    # cv2.imshow('1-Image', roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.fillPoly(roi, [shape.reshape((-1, 1, 2))], (255, 255, 255))  # 在黑色图像中填充形状区域
    # cv2.imshow('2-Image', roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
    # cv2.imshow('3-Image', roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    roi = cv2.bitwise_and(img_rotated, img_rotated, mask=roi)  # 将输入图片和形状区域取交集
    # cv2.imshow('4-Image', roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
    # cv2.imshow('5-Image', roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    output_path = os.path.join(output_dir, f"vertebra_{idx}.jpg")  # 输出图片的路径
    cv2.imwrite(output_path, roi)  # 保存分割后的椎体图像

# 保存绘制了形状的图片
output_image_path = "output.jpg"
cv2.imwrite(output_image_path, img_rotated)  # 保存绘制了形状的图片