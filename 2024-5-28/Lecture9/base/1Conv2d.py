import torch
import torch.nn as nn
# from torch.utils.data import Dataset
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)  # 卷积层
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.MaxPool = nn.MaxPool2d(2, 2)  # 池化层
        self.l1 = nn.Linear(320, 10, bias=False)  # 线性层
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.MaxPool(self.activation(self.conv1(x)))  # 1
        x = self.MaxPool(self.activation(self.conv2(x)))  # 2

        x = x.view(-1, 320)
        # print(x.shape)#测试linear输入列数
        return self.l1(x)


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
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

if __name__ == "__main__":
    for epoch in range(5):
        train(epoch)
        text()
    draw(Unity_epoch, MSE)
