import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 三维图库

# 参数
w = 0
b = 0

# 训练集
x_data = [1.0, 2.0, 3.0]
y_data = [2.1, 4.1, 6.1]
n = 3

# 参数集合
w_list = np.arange(0.0, 4.1, 0.1)
b_list = np.arange(0.0, 2.0, 0.1)

loss_S = 0
MSE = np.zeros((len(w_list), len(b_list)))  # 基于w b 创建二维数组


# 模型
def sigmoid(x):
    return w * x + b


# 损失平方
def loss(pre_y, y):
    return pow(y - pre_y, 2)


#   Guess线性
for i, w in enumerate(w_list):  # i 为序号 w为值
    for j, b in enumerate(b_list):  # 先b后w
        print(f'w = {w:.2f}')
        print(f'b = {b:.2f}')
        loss_S = 0

        # 训练集合
        for x_i, y_i in zip(x_data, y_data):
            p_y = sigmoid(x_i)  # y预测值
            loss_S += loss(p_y, y_i)  # 损失
            # 打印输出
            print(f"x_val = {x_i:.2f} "
                  f"\ty_val= {y_i:.2f} "
                  f"\tpre_y= {p_y:.2f} "
                  f"\tloss= {loss(x_i, y_i):.2f}"
                  )

        print(f'\tMSE={(loss_S / n):.2f}')
        MSE[i, j] = loss_S / n

# 三维图形模式
# B, W = np.meshgrid(b_list, w_list)
W, B = np.meshgrid(w_list, b_list)  # 先w后b

fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(W, B, MSE.T, cmap='rainbow')  # 画出(xs1,ys1,zs1)的散点图。

ax.set_xlabel('w label')  # 画出坐标轴
ax.set_ylabel('b label')
ax.set_zlabel('MSE label')

plt.show()
