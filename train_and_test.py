import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os

torch.manual_seed(1)

# 超参数
EPOCH = 5
BATCH_SIZE = 64
LR = 0.001
DOWNLOAD_MNIST = False

# 训练集
train_data = torchvision.datasets.MNIST(
    root='/home/zy/pytorch+MNIST/MINIST',
    train=True,
    transform=torchvision.transforms.ToTensor(),  #归一化
    download=False#由于仓库中已经携带了MNIST数据集，所以设置为False，如果设置为true则会再次下载数据集
)

# 测试集
test_data = torchvision.datasets.MNIST(
    root='/home/zy/pytorch+MNIST/MINIST',
    train=False,
    transform=torchvision.transforms.ToTensor()
)

# 训练集 DataLoader
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8
)

# 测试集 DataLoader
test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# 用class类来建立CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

cnn = CNN()
print(cnn)

# 优化器选择Adam
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-5)
loss_func = nn.CrossEntropyLoss() # 目标标签是one-hotted

# 训练循环开始

# 初始化最佳准确率
best_acc = 0.0

for epoch in range(EPOCH):
    cnn.train()
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            # 使用 test_loader 分批计算准确率
            cnn.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                for test_inputs, test_labels in test_loader:
                    test_outputs = cnn(test_inputs)
                    _, predicted = torch.max(test_outputs.data, 1)
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()
            accuracy = correct / total
            print(f'Epoch: {epoch} | Step: {step} | Loss: {loss.item():.4f} | Acc: {accuracy:.2f}')
            
            # 如果当前模型更好，则保存
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(cnn.state_dict(), 'best_cnn.pkl')  # 保存最佳模型
                print(f'Best model saved with accuracy: {best_acc:.2f}')
            
            
            cnn.train()
#至此，训练结束，训练过程中保存了最佳模型，下面进行测试
#注意训练完成后以上的代码可以注释掉，无需再训练

# 加载模型
cnn.load_state_dict(torch.load('best_cnn.pkl', map_location=torch.device('cpu')))
cnn.eval()

# 预测
test_iter = iter(test_loader)
inputs, labels = next(test_iter)
outputs = cnn(inputs[:32])
pred_y = torch.max(outputs, 1)[1].numpy()

# 显示结果
img = torchvision.utils.make_grid(inputs[:32])
img = img.numpy().transpose(1, 2, 0)

print('Predicted:', pred_y)
print('Real:', labels[:32].numpy())

plt.imshow(img)
plt.title('Prediction vs Real')
plt.show()
