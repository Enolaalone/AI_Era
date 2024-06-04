import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as op

lr = 0.01
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

w_list = []
b_list = []
mse_list = []


def draw(x, y, lx, ly):  # 坐标图
    plt.plot(x, y)
    plt.xlabel(f'{lx}')
    plt.ylabel(f'{ly}')
    plt.show()


def draw2(x, y, loss):
    fig = plt.figure()  # 创建画布
    ax = fig.add_subplot(1, 1, 1, projection='3d')  # 一行一列一个子图，3绘图
    ax.plot_trisurf(x, y, loss, cmap='viridis')  # 基于坐标数列构建坐标，颜色基于大小变化
    # 标签
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_zlabel('Loss')
    ax.set_title('3D plot of w, b and Loss')  # 标题
    fig.show()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()  # 继承父类
        self.linear = nn.Linear(1, 1)  # 训练函数---pre_y,x维度

    def forward(self, x):  # 前馈函数 override
        return self.linear(x)


# 创建模型
model = Model()
criterion = nn.MSELoss(reduction='sum')  # 损失函数
optimizer = op.SGD(model.parameters(), lr)  # 梯度更新函数

for epoch in range(1000):
    y_pred = model(x_data)  # 计算y_pre
    loss = criterion(y_pred, y_data)  # 计算损失
    optimizer.zero_grad()  # 梯度清理
    loss.backward()  # 反馈
    optimizer.step()  # 更新
    # 绘图数据
    mse_list.append(loss.item())
    w_list.append(model.linear.weight.item())
    b_list.append(model.linear.bias.item())

    # print
    print('Epoch:', epoch, 'Loss:', loss.item())
    print(f"w =  {model.linear.weight.item():.2f}   b = {model.linear.bias.item()} ")
    print()

draw2(w_list, b_list, mse_list)
draw(w_list, mse_list, 'w', 'MSE')
draw(b_list, mse_list, 'b', 'MSE')

# Text
text_x = torch.tensor([[4.0]])
print(f'text = {model(text_x).item()}')
