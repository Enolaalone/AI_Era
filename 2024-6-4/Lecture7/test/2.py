
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L = []
N = []
n = 1


class MyData(Dataset):
    def __init__(self):
        super(MyData, self).__init__()
        self.data = np.loadtxt('../diabetes.csv', delimiter=',', dtype=np.float32)  # 读取数据
        self.x = torch.from_numpy(self.data[:, :-1])
        self.y = torch.from_numpy(self.data[:, [-1]]).long()
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        # Con2d
        self.conv1 = nn.Conv2d(in_channels,  in_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d( in_channels, in_channels, kernel_size=5, padding=2)
        # act
        self.activation = nn.ReLU()

    def forward(self, input_):
        input_ = self.activation(self.conv1(input_))
        input_ = self.activation(self.conv2(input_) + input_)
        return input_


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Con2d
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5,padding=2)
        # Pool
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # Residual
        self.residual1 = ResidualBlock(10)
        self.residual2 = ResidualBlock(20)
        # active
        self.active = nn.ReLU()
        # Linear
        self.fc1 = nn.Linear(1280, 64, bias=False)

    def forward(self, input_):
        batch_size = input_.size(0)
        # print(input_.size())
        input_ = self.pool(self.active(self.conv1(input_)))
        input_ = self.residual1(input_)
        input_ = self.pool(self.active(self.conv2(input_)))
        input_ = self.residual2(input_)
        input_ = input_.view(batch_size, -1)
        # print(input_.size(1))
        return self.fc1(input_).view(64, -1,1)


def train(epoch):
    global n
    for idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        data = data.view(8,1,8, -1)
        target = target
        outputs = model(data)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            L.append(loss.item())
            N.append(n)
            n += 1
            # print(f'Epoch {epoch},idx{idx}')
            # print(f'Loss {loss.item():.4f}')
    # print()


def draw():
    plt.plot(N, L)
    plt.xlabel('TenBatch')
    plt.ylabel('Loss')
    plt.show()


# Data
input_size = 8
batch_size = 64
train_set = MyData()
train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True)

# Model
model = Net().to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train
if __name__ == "__main__":
    for epoch in range(1000):
        train(epoch)
    draw()
