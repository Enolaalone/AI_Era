import numpy as np
import matplotlib.pyplot as plt

# 参数
w = 0
b = 0

# 训练集
x_data = [1.0, 2.0, 3.0]
y_data = [2.1, 4.1, 6.1]
n = 3

# 参数集合
w_list = []
b_list = []

loss_S = 0
MSE = []


# 模型
def sigmoid(x):
    return w * x + b


# 损失平方
def loss(x, y):
    pre_y = sigmoid(x)
    return pow(y - pre_y, 2)


for w in np.arange(0.0, 4.1, 0.1):
    w_list.append(w)
    for b in np.arange(0.0, 2.0, 0.1):
        b_list.append(b)

        print('w =', w)
        print('b =', b)
        loss_S = 0

        # 训练集合
        for x_i, y_i in zip(x_data, y_data):
            p_y = sigmoid(x_i)  # y预测值
            loss_S += loss(x_i, y_i)  # 损失
            # 打印输出
            print('x_val =', x_i,
                  '\ty_val', y_i,
                  '\tpre_y=', p_y,
                  '\tloss=', loss(x_i, y_i))

        print('\tMSE=', loss_S / n)
        MSE.append(loss_S / n)

# 三维图形模式
fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(np.array(w_list), np.array(b_list), np.array(MSE), cmap='rainbow')  # 画出(xs1,ys1,zs1)的散点图。

ax.set_xlabel('w label')  # 画出坐标轴
ax.set_ylabel('b label')
ax.set_zlabel('MSE label')

plt.show()
