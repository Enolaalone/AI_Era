import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np


class MyDataset(Dataset):
    def __init__(self, filename):
        super(MyDataset, self).__init__()  # __init__()不加参数
        self.xy = np.loadtxt(filename, delimiter=',', dtype=np.float32)  # 文件名称，分割符号，转为数据类型
        self.x_data = torch.from_numpy(self.xy[:, :-1])  # 第一到倒数第二列数据
        self.y_data = torch.from_numpy(self.xy[:, [-1]])  # 所有行，最后的数据转为  [[1],[0],[1]]
        # print('x_data shape', self.x_data.shape)
        self.len = self.xy.shape[0]

    def __getitem__(self, index):  # 获得特定是数据
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # 获得长度
        return self.len


# Model
class HdNet(nn.Module):
    def __init__(self):
        super(HdNet, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        return x


def train(epochs):
    if __name__ == '__main__':  # windows不能掉这一句
        for epoch in range(epochs):  # 训练轮数
            mse = 0
            for i, data in enumerate(train_loader):
                # forward
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # GPU
                # loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # 损失
                # backward
                optimizer.zero_grad()  # 优化器清除梯度
                loss.backward()
                # update
                optimizer.step()


# Data
train_data = MyDataset('diabetes.csv')
train_loader = DataLoader(dataset=train_data,
                          batch_size=5,
                          shuffle=True,
                          num_workers=3)
# Model
model = HdNet()

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model.to(device)
# 损失 优化器
criterion = nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# train
train(epochs=10)


