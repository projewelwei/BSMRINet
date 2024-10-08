import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from model import resnet18


transform = transforms.Compose([
    transforms.Resize(256),  # 调整图片大小为 256x256
    transforms.CenterCrop(224),  # 居中裁剪 224x224 的图片
    transforms.ToTensor(),  # 将图片转换为 Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 64  # 定义批次大小

# train 数据集加载器
train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# val 数据集加载器
val_dataset = datasets.ImageFolder(root='./data/val', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# test 数据集加载器
test_dataset = datasets.ImageFolder(root='./data/test', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model_weight_path = "./resnet50.pth"
# resnet = torchvision.models.resnet50(pretrained=True)
resnet = resnet18(2)
# num_features = resnet.fc.in_features
# resnet.fc = nn.Linear(num_features, 2)  # 替换掉原有输出层
resnet = resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)

num_epochs = 30

train_losses = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):
    # 训练模型
    resnet.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # 梯度清零
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    # 在验证集上评估模型
    resnet.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = resnet(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            val_loss += loss.item() * images.size(0)
            val_correct += torch.sum(preds == labels.data)

    train_loss = train_loss / len(train_dataset)
    val_loss = val_loss / len(val_dataset)
    val_acc = val_correct.double() / len(val_dataset)
    # 记录训练损失、验证损失和验证准确率
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)


    print('Epoch [{}/{}] train_loss: {:.4f} val_loss: {:.4f} val_acc: {:.4f}'.format(
        epoch+1, num_epochs, train_loss, val_loss, val_acc))
save_path = "resnet18.pth"
torch.save(resnet.state_dict(), save_path)
# 在测试集上评估模型
test_correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = resnet(images)
        _, preds = torch.max(outputs, 1)

        test_correct += torch.sum(preds == labels.data)

test_acc = test_correct.double() / len(test_dataset)
print('Test_acc: {:.4f}'.format(test_acc))

fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
ax[0].plot(train_losses, label='Train Loss')
ax[0].plot(val_losses, label='Val Loss')
ax[0].legend()
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].set_title('Train and Val Loss')

ax[1].plot(val_accs, label='Val Acc')
ax[1].legend()
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('Validation Accuracy')

plt.show()
