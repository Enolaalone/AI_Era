import torch
from torchvision import transforms,datasets  # 图像转换
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSE = []
Unity_epoch = []
un = 1
batch_size = 64
# Data
transform = transforms.Compose([transforms.ToTensor(),  # 转为图像张量[[a1,...,a28],
                                # [],
                                # ...
                                # []]
                                transforms.Normalize((0.1307,), (0.3081,))  # 图像灰度转换 均值与标准差
                                ])

train_set = datasets.MNIST('MNIST_data',  # 60000张
                           train=True,
                           download=False,
                           transform=transform)

train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True)

text_set = datasets.MNIST('MNIST_data',  # 10000张
                          train=False,
                          download=False,
                          transform=transform)

text_loader = DataLoader(dataset=text_set,
                         batch_size=batch_size,
                         shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 10)
        
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)  # 改变[28,28]张量形状，-1自动计算Batch数量

        x = self.activation (self.l1(x))
        x = self.activation (self.l2(x))
        x = self.activation (self.l3(x))
        x = self.activation (self.l4(x))
        return self.l5(x)  # 不做激活


def draw(x, y):  # 画图
    plt.plot(x, y)
    plt.xlabel('nums')
    plt.ylabel('loss')
    plt.show()


def train():  # 一轮训练
    global un
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # forward + backward + update
        outputs = model(data)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx+1) % 100 == 0:
            # 绘图数据
            MSE.append(running_loss / batch_size)
            Unity_epoch.append(un)
            un += 1
            print(f'Train Epoch:{running_loss / batch_size}')


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 无梯度
        for data, target in text_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, dim=1)  # 返回 最大值 和 最大值的下标
            total += target.size(0)  # 返回一组样本大小
            correct += (predicted == target).sum().item()
        print('Test set: Average loss: {:.4f}', format(100 * correct / total))
        print()


model = Net().to(device)

# 损失&优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

if __name__ == "__main__":
    for epoch in range(3):
        train()
        test()
    draw(Unity_epoch, MSE)
