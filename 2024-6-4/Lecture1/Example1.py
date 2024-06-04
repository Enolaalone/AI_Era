import numpy as np
import matplotlib.pyplot as plt

# 训练集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 模型
def forward(x):
    return x * w


# 误差计算
def loss(x, y):
    _y_pred = forward(x)
    return (_y_pred - y) * (_y_pred - y)


w_list = []  # 参数集合
mse_list = []   # 平均误差集合

for w in np.arange(0.0, 4.1, 0.1):
    print('w =', w)
    l_sum = 0

    for x_val, y_val in zip(x_data, y_data):    # 数据集合转换为元组
        y_pred = forward(x_val)     # 预测值
        loss_val = loss(x_val, y_val)   # 损失
        l_sum += loss_val   # 损失和

        print('\t', x_val, y_val, loss_val)
    print('MSE =', l_sum / 3)

    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list)
plt.xlabel('w')
plt.ylabel('Loss')
plt.show()
