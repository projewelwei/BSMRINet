import cv2
import numpy as np
import os

# 加载图片
image_path = r'output.jpg'  # 请替换为你的图片路径
img = cv2.imread(image_path)

# 转换为HSV色彩空间，以便于检测绿色
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义绿色的范围
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# 创建绿色的掩膜
mask = cv2.inRange(hsv, lower_green, upper_green)

# 在掩膜上找到轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 确保输出目录存在
output_dir = r""
os.makedirs(output_dir, exist_ok=True)

# 分割并保存每个绿色框内的区域
for i, contour in enumerate(contours):
    # 获取轮廓的边界框坐标
    x, y, w, h = cv2.boundingRect(contour)

    # 从原图中提取ROI（感兴趣区域）
    roi = img[y:y + h, x:x + w]

    # 保存分割后的图像
    segment_path = os.path.join(output_dir, f'segment_{i + 1}.jpg')
    cv2.imwrite(segment_path, roi)
    print(f"保存了分割图像: {segment_path}")

print("分割完成。")
