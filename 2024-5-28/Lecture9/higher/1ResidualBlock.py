import torch
import torch.nn as nn
# from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # 导入MINIST.datasets,transforms
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cpu")
MSE = []
Unity_epoch = []
un = 1

input_size = 28
batch_size = 64

# Data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

train_set = datasets.MNIST('../MNIST_data',
                           train=True,
                           download=False,
                           transform=transform)

train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True)

text_set = datasets.MNIST('../MNIST_data',
                          train=True,
                          download=False,
                          transform=transform)

text_loader = DataLoader(dataset=text_set,
                         batch_size=batch_size,
                         shuffle=False)


# 防止梯度消失单元
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)#W:nan
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)#W:nan
        self.activate = nn.ReLU()

    def forward(self, x):
        y = self.activate(self.conv1(x))
        y = self.activate(self.conv1(y) + x)  # Residual
        return y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)  # C:1->16  W:28-5+1=24
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # C:16->32  W:24-5+1=20
        # Residual
        self.residual1 = ResidualBlock(16)
        self.residual2 = ResidualBlock(32)
        # Pool
        self.pool = nn.MaxPool2d(kernel_size=2)  # W: -2+1
        # Active
        self.activation = nn.ReLU()
        # Linear
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        # 1
        x = self.pool(self.activation(self.conv1(x)))
        x = self.residual1(x)

        # 2
        x = self.pool(self.activation(self.conv2(x)))
        x = self.residual2(x)

        x = x.view(in_size, -1)
        return self.linear(x)


def train(epoch):
    global un
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 数据转移
        # forward+backward+update
        outputs = model(data)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # print(f' loss:{loss.item()}')
        if batch_idx % 300 == 299:
            # 绘图数据
            MSE.append(running_loss / 300)
            Unity_epoch.append(un)
            un += 1
            print(f'Train Epoch:{running_loss / 300}')


def text():
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in text_loader:
            data, target = data.to(device), target.to(device)  # 数据转移
            outputs = model(data)
            _, predicted = torch.max(outputs.data, dim=1)  # 按列从左到右扫描数据返回最大值序号到predicted
            total += target.size(0)
            correct += (predicted == target).sum().item()
        print(f'正确率：{100 * correct / total}')
        print()


def draw(x, y):  # 画图
    plt.plot(x, y)
    plt.xlabel('nums')
    plt.ylabel('loss')
    plt.show()


# Model
model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')  # 模型转移
model.to(device)

# criterion  optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

if __name__ == "__main__":
    for epoch in range(5):
        train(epoch)
        text()
    draw(Unity_epoch, MSE)
