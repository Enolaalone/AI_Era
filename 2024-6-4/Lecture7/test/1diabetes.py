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
        self.data = np.loadtxt('../diabetes2.csv', delimiter=',', dtype=np.float32)  # 读取数据
        self.x = torch.from_numpy(self.data[:, :-1])
        self.y = torch.from_numpy(self.data[:, [-1]])
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 96)
        self.fc2 = nn.Linear(96, 48)
        self.fc3 = nn.Linear(48, 24)
        self.fc4 = nn.Linear(24,12)
        self.fc5 = nn.Linear(12, 1)
        self.active1 = nn.Sigmoid()
        self.active2 = nn.ELU()

    def forward(self, inputs):
        inputs = self.active2(self.fc1(inputs))
        inputs = self.active2(self.fc2(inputs))
        inputs = self.active2(self.fc3(inputs))
        inputs = self.active2(self.fc4(inputs))
        inputs = self.active1(self.fc5(inputs))
        return inputs


def train(epoch):
    global n
    for idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
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
    plt.title("ELU")
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
model = Net(input_size).to(device)

# Loss & Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train
if __name__ == "__main__":
    for epoch in range(1000):
        train(epoch)
    draw()
