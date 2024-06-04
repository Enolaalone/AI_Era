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


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        # 分支线构造
        self.branch1x1_16 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_16_1 = nn.Conv2d(in_channels, 16, kernel_size=1)  # kernel_size -2*padding+1为图像边长变化
        self.branch5x5_24_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_16_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_24_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_24_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool1x1_24_2 = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch_1 = self.branch_pool1x1_24_2(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))
        branch_2 = self.branch1x1_16(x)
        branch_3 = self.branch5x5_24_2(self.branch5x5_16_1(x))
        branch_4 = self.branch3x3_24_3(self.branch3x3_24_2(self.branch3x3_16_1(x)))

        out = [branch_1, branch_2, branch_3, branch_4]
        return torch.cat(out, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.inception1 = Inception(10)
        self.inception2 = Inception(20)

        self.MaxPool = nn.MaxPool2d(2)  # 池化 不改变channels
        self.activate = nn.ReLU()  # 激活 不改变channels

        self.fc1 = nn.Linear(1408, 10,bias=False)

    def forward(self, x):
        in_size = x.size(0)
        x = self.inception1(self.activate(self.MaxPool(self.conv1(x))))
        x = self.inception2(self.activate(self.MaxPool(self.conv2(x))))

        x = x.view(in_size, -1)
        return self.fc1(x)


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
            MSE.append(running_loss / (300*batch_idx) )
            Unity_epoch.append(un)
            un += 1
            print(f'Train Epoch:{running_loss / (300*batch_idx) }')


def test():
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
        test()
    draw(Unity_epoch, MSE)
