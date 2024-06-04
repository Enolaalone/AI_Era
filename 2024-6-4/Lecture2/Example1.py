import numpy as np
import matplotlib.pyplot as plt

learn_a = 0.01
w = 0.0

x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([2.0, 4.0, 6.0])
n = 3

w_list = []
mse_list = []


# 模型
def sigmoid(x):
    return w * x


# 损失
def cost(x, y):
    MSE = 0
    for x_i, y_i in zip(x, y):
        pre_y = sigmoid(x_i)  # 预测
        MSE += (pre_y - y_i) ** 2
    MSE /= len(x)
    mse_list.append(MSE)
    return


def gradient(x, y):
    grad = 0
    for x_i,y_i in zip(x,y):
        grad += 2 * x_i * (sigmoid(x_i) - y_i)
    grad /= len(x)
    return grad  # 返回梯度


def update_w():
    global w, x_data, y_data
    w -= learn_a * gradient(x_data, y_data)
    w_list.append(w)


for epoch in range(100):
    update_w()
    print(f"Epoch: {epoch}\t Cost: {cost(x_data, y_data)}")

plt.plot(w_list, mse_list)
plt.xlabel("W")
plt.ylabel("MSE")
plt.show()