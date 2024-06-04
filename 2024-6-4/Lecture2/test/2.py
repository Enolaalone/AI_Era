import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        self.x = torch.from_numpy(np.arange(0, 20))
        self.y = torch.from_numpy(np.arange(0, 40, 2))
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


class Net:
    def __init__(self):
        self.w = torch.tensor([10.0],
                              requires_grad=True)  # Only Tensors of floating point and complex dtype can require gradients
        self.b = torch.tensor([5.0], requires_grad=True)

    def forward(self, x):
        return self.w * x + self.b

    def get_params(self):
        return self.w, self.b


class Loss:
    def __init__(self):
        pass

    def cost(self, output, target):
        return (output - target).pow(2)


class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:  # 梯度必须存在
                param.grad.data.zero_()

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad.data


def train(epoch):
    for idx, (data, target) in enumerate(train_loader, 1):
        output = model.forward(data)
        loss = criterion.cost(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not (idx % 10):
            print(f'Epoch: {epoch}\t idx :{idx}')
            print(f'Loss: {loss.item():.3f}\tw: {model.w.item():.3f}\tb: {model.b.item():.3f}')
    print()


# Data
train_set = MyDataset()
train_loader = DataLoader(dataset=train_set,
                          batch_size=1,
                          shuffle=True)
# Model
model = Net()

# Loss & optimizer
criterion = Loss()
optimizer = Optimizer(model.get_params(), lr=0.001)

# train
for epoch in range(100):
    train(epoch)
