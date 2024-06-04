import matplotlib.pyplot as plt
from torch import Tensor

a = 0.01
MSE = []
w1_data = []
w2_data = []
b_data = []
# 训练集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# Tensor
w1 = Tensor([1.0])
w1.requires_grad_(True)

w2 = Tensor([1.0])
w2.requires_grad_(True)

b = Tensor([1.0])
b.requires_grad_(True)

pre_y = Tensor([0.0])
pre_y.requires_grad_(True)


# 模型
def forward(x):

    return w1 * x ** 2 + w2 * x + b


# 损失
def loss(x, y):
    global pre_y
    pre_y = forward(x)
    return (pre_y - y) ** 2


for epoch in range(1000):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        MSE.append(l.item())
        l.requires_grad_(True)
        l.backward()

        # update_grad
        w1.data = w1.data - a * w1.grad.data
        w2.data = w2.data - a * w2.grad.data
        b.data = b.data - a * b.grad.data

        print("Epoch: {}, Loss: {}".format(epoch, l))
        print("w1: {}, w2: {}, b: {}".format(w1,w2,b))
        print()

        w1_data.append(w1.item())
        w2_data.append(w2.item())
        b_data.append(b.item())

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()

# w1_data = w1.tolist()
# w2_data = w1.tolist()
# b_data = w1.tolist()

plt.plot(w1_data, MSE)
plt.xlabel('w1')
plt.ylabel('Loss')
plt.show()

plt.plot(w2_data, MSE)
plt.xlabel('w2')
plt.ylabel('Loss')
plt.show()

plt.plot(b_data, MSE)
plt.xlabel('b')
plt.ylabel('Loss')
plt.show()