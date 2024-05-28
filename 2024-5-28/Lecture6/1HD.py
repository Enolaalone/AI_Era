import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

# 数据导入
xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)

x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])

Epoch = []
MSE = []


# model
class HdNet(nn.Module):
    def __init__(self):
        super(HdNet, self).__init__()
        self.linear1 = nn.Linear(8, 72)  # 降维函数
        self.linear2 = nn.Linear(72, 64)
        self.linear3 = nn.Linear(64, 1)
        self.activation = nn.Sigmoid() # 激活函数

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        return x


model = HdNet()

# 损失
criterion = nn.BCELoss(reduction='mean')

# 优化函数
optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(5000):
    pre_y = model(x_data)  # forward
    loss = criterion(pre_y, y_data)  # loss
    Epoch.append(epoch)
    MSE.append(loss.item())

    optimiser.zero_grad()
    loss.backward()  # backward

    optimiser.step()  # Update

    #print
    print(f'epoch:{epoch},loss:{loss}')

plt.plot()
plt.plot(Epoch,MSE)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()