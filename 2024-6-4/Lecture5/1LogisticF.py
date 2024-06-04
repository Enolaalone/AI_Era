import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 数据集合-----------------------------------------------------
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


# Logistic模型---------------------------------------------------
class LogisticNet(nn.Module):
    def __init__(self):
        super(LogisticNet, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return F.sigmoid(self.linear(x))


# 模型
model = LogisticNet()

# 损失BCE
criterion = nn.BCELoss(reduction='mean')

# 优化函数
optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

# Training------------------------------------------------------------
for epoch in range(1000):
    pre_y = model(x_data)  # Model调用前馈函数
    loss = criterion(pre_y, y_data)  # 损失

    optimiser.zero_grad()  # 梯度清除
    loss.backward()  # 反馈

    optimiser.step()  # 更新

    # Print
    print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))
