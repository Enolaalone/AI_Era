import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L = []
N = []
n = 1


# Preprocess CSV file to make it compatible with np.loadtxt

class TitanicDataset(Dataset):
    def __init__(self, train=True):
        super(TitanicDataset, self).__init__()
        file_name = "train.csv" if train else "test.csv"
        self.read=pd.read_csv(file_name, na_values='',usecols=['PassengerId','Pclass', 'Age','SibSp', 'Parch','Fare','Survived'])
        self.data = self.read.dropna().to_numpy(dtype=np.float32)  # 删除包含缺失值的行并转换为 numpy 数组

        # self.data = np.loadtxt(self.data2,skiprows=1,dtype=np.float32)  # 支持str读取，skipprows跳过首行

        self.len = self.data.shape[0]
        self.x = torch.from_numpy(self.data[:, 2:])
        self.y = torch.from_numpy(self.data[:, [1]])
        # print(self.x[0])
        # print(self.y[0])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


class Net(nn.Module):
    def __init__(self,input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 96)
        self.fc2 = nn.Linear(96, 48)
        self.fc3 = nn.Linear(48, 24)
        self.fc4 = nn.Linear(24, 12)
        self.fc5 = nn.Linear(12, 1)
        self.active1 = nn.Sigmoid()
        self.active2 = nn.Tanh()

    def forward(self, inputs):
        inputs = self.fc1(self.active2(inputs))
        inputs = self.fc2(self.active2(inputs))
        inputs = self.fc3(self.active2(inputs))
        inputs = self.fc4(self.active2(inputs))
        inputs = self.fc5(self.active2(inputs))
        return self.active1(inputs)


def train(epoch):
    global n
    for idx, (x, y) in enumerate(train_loader, 1):
        x, y = x.to(device), y.to(device)
        # print(x.shape, y.shape)
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 5 == 0:  # Print loss every 10 batches
            L.append(loss.item())
            N.append(n)
            n += 1
            print(f'Epoch {epoch}, BCE: {loss.item():.4f}')

# def test():
#     global n
#     with torch.no_grad():
#         for idx, (x, y) in enumerate(test_loader, 1):
#             x, y = x.to(device), y.to(device)
#             # print(x.shape, y.shape)
#             out = model(x)
#             loss = criterion(out, y)
#
#             if idx % 5 == 0:  # Print loss every 10 batches
#                 L.append(loss.item())
#                 N.append(n)
#                 n += 1

def draw():
    plt.plot(N, L)
    plt.xlabel('FiveBatch')
    plt.ylabel('Loss')
    plt.show()


# Data
epochs = 5000
batch_size = 64
input_size = 5
train_data = TitanicDataset(True)
train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          shuffle=True)
# test_data = TitanicDataset(False)
# test_loader = DataLoader(dataset=test_data,
#                          batch_size=batch_size,
#                          shuffle=False)

# Model
model = Net(input_size).to(device)
# Loss & Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Train
if __name__ == "__main__":
    for epoch in range(epochs):
        train(epoch)
        # test()

    draw()
