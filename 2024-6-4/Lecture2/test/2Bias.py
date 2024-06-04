import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset



class MyDataset(Dataset):
    def __init__(self):
        self.x_data = torch.from_numpy(np.arange(0, 20))
        self.y_data = torch.from_numpy(np.arange(0, 40, 2))
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Net:  # Model
    def __init__(self):  # 参数初始化
        self.w = torch.tensor([10.0], requires_grad=True)  # 张量才可以计算梯度 记得加中括号
        self.b = torch.tensor([10.0], requires_grad=True)

    def forward(self, x):  # 前馈
        return self.w * x + self.b

    def get_parameters(self):  # 获取参数
        return self.w, self.b


class Loss:
    def __init__(self):
        pass

    def lost(self, output, target):
        return (output - target) ** 2


class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters  # 参数元组
        self.lr = lr

    def zero_grads(self):  # 梯度归零
        for param in self.parameters:
            # print(param)
            if param.grad is not None:  # 如果梯度不为零
                param.grad.zero_()

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad.data


def train(epoch):
    for idx, (data, target) in enumerate(train_loader, 1):
        # forward
        output = model.forward(data)
        # print(output, target)
        loss = criterion.lost(output, target)
        optimizer.zero_grads()

        # backward
        loss.backward()
        # update
        optimizer.step()

        if not (idx % 2):
            print(f"Epoch: {epoch}, idx: {2 * idx},\n loss: {loss.item():.4f}")
            print(f"pre_y :{output.item():.2f} ,y_true :{target.item()}, w : {model.get_parameters()[0].item():.3f},"
                  f"b : {model.get_parameters()[1].item():.3f}")
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
optimizer = Optimizer(model.get_parameters(), lr=0.001)

# Train
for epoch in range(100):
    train(epoch + 1)
