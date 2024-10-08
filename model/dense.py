import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from densemodel import DenseNet
from sklearn.metrics import roc_auc_score, recall_score, f1_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为模型输入尺寸
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 图像归一化
])

# 定义训练和测试集路径
train_path = '/root/autodl-tmp/data/train'  # 替换为你的训练集路径
test_path = '/root/autodl-tmp/data/test'  # 替换为你的测试集路径

# 加载训练集和测试集
train_data = ImageFolder(train_path, transform)
test_data = ImageFolder(test_path, transform)

# 定义批处理大小和数据加载器
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 创建dense模型
model = DenseNet(num_classes=4)  # 设置num_classes为4

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 GPU 或 CPU 进行训练
model.to(device)

num_epochs = 30
for epoch in range(num_epochs):
    total_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss / len(train_loader)))
# 在测试集上评估模型
test_correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        test_correct += torch.sum(preds == labels.data)

test_acc = test_correct.double() / len(test_data)
print('Test_acc: {:.4f}'.format(test_acc))

