import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MyDataset(Dataset):
    def __init__(self):
        self.x_data = torch.from_numpy(np.arange(0, 5))
        self.y_data = torch.from_numpy(np.arange(0, 10, 2))
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Net:  # Model
    def __init__(self):  # 参数初始化
        self.w = 10

    def forward(self, x):  # 前馈
        return self.w * x

    def update(self, w):  # 更新参数
        self.w = w

    def get_w(self):  # 获取参数
        return self.w


class Loss:
    def __init__(self):
        pass

    def lost(self, output, target):
        return float((output - target) ** 2)


class Optimizer:
    def __init__(self, model, lr):
        self.model = model  # 引用
        self.w = model.get_w()
        self.lr = lr
        self.grad = 0

    def zero_grads(self):
        self.grad = 0

    def step(self, input, target):
        self.grad = 2 * (self.model.forward(input) - target) * self.w  # 反向传播计算梯度更新
        self.w -= self.lr * self.grad
        self.model.update(self.w)


def train(epoch):
    for idx, (data, target) in enumerate(train_loader, 1):
        output = model.forward(data)
        loss = criterion.lost(output, target)
        optimizer.zero_grads()
        optimizer.step(data, target)

        if not (idx % 2):
            print(f"Epoch: {epoch}, idx: {2 * idx},\n loss: {loss:.4f}")
            print(f"pre_y :{output.item():.2f} ,y_true :{target.item()}, w : {model.get_w().item()}")
            print()
    print("\n\n")


# Data
train_dataset = MyDataset()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=1,
                          shuffle=True)
# Model
model = Net()

# Loss & optimizer
criterion = Loss()
optimizer = Optimizer(model, lr=0.01)

# Train
for epoch in range(10):
    train(epoch + 1)
