import matplotlib
import random
import torch
from d2l import torch as d2l
    
def synthetic_data(w,b,num_example):
    # 生成 y = Xw+b+噪声
    X= torch.normal(0,1,(num_example,len(w)))# shape [10,2]
    # print(X)
    y=torch.matmul(X,w)+b # output

    # 添加噪声
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])# 权重
true_b = 4.2# 偏置量
features , labels = synthetic_data(true_w,true_b,1000)
# print(features[0],labels[0])

# ------------绘图--------------------
# d2l.set_figsize()
# #  X，Y ,dot_size 从计算图取出
# d2l.plt.scatter(features[:,(1)].detach().numpy(),labels.detach().numpy(),1)
# d2l.plt.show()

#----------------小批量读取-----------------------

def data_iter(batch_size,features,labels):
    num_example = len(features)
    indices = list(range(num_example))# 样本索引数据
    random.shuffle(indices)# 打乱顺序

    for i in range(0,num_example,batch_size):
        # 建立批量序号和数据映射
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_example)])
        # print(batch_indices)
        yield features[batch_indices],labels[batch_indices]#取出批量样本

batch_size = 10
for x,y in data_iter(batch_size,features,labels):
    print(x,'\n',y)
    break

# ------------------模型--------------------------

w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

def linear(x,w,b):
    return torch.matmul(x,w)+b

# ----------------损失函数 - 优化器--------------------

def square_loss(y_hat,y):
    return (y_hat-y)**2/2

def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

# ---------train----------------------------------

lr = 0.01
num_epoch = 3
net = linear
loss = square_loss

for epoch in range(num_epoch):
    for x,y in data_iter(batch_size,features,labels):
        l = loss(net(x,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)

    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch{epoch},loss{float(train_l.sum().item()/1000)}')