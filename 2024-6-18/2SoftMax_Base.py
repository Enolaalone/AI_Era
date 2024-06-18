import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from IPython import display
from d2l import torch as d2l


# Data
batch_size = 256
transform = transforms.ToTensor()

mnist_train = torchvision.datasets.FashionMNIST(root="../data",train=True,transform=transform,download=False)

mnist_test = torchvision.datasets.FashionMNIST(root="../data",train=False,transform=transform,download=False)

train_loader = DataLoader(dataset=mnist_train,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=mnist_test,
                          batch_size=batch_size,
                          shuffle=False)

#---------------------------------------------
num_inputs = 784
num_outputs = 10 #十种分类

# 参数
W = torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs,requires_grad=True)



#----------softMax-------------------------------

def softMax(x):
    X_exp = torch.exp(x)
    partition = X_exp.sum(dim=1,keepdim=True)
    return X_exp/partition

#----------------Model-------------------

def net(x):
    return softMax(torch.matmul(x.reshape((-1,W.shape[0])),W)+b)


#----------------------cross_entropy--------------
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])


#---------------optimizer----------------
lr = 0.1
def updater(batch_size):
    return d2l.sgd([W,b],lr,batch_size)


#---------------------精确度-------------------

def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,data_loader):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for x,y in data_loader:
        metric.add(accuracy(net(x),y),y.numel())

        print(metric[0]/metric[1])
    return metric[0]/metric[1]
#-----------------累加器------------------------
class Accumulator:
    def __init__(self,n):
        self.data = [0.0]*n
    def add(self,*args):
        self.data = [a + float(b) for a,b in zip(self.data,args)]

    def reset(self):
        self.data = [0.0]*len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
#---------------训练------------------------

def train_epoch_ch3(net,train_loader,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()

    metric = Accumulator(3)
    for x,y in train_loader:

        y_hat= net(x)
        l = loss(y_hat,y)

        # torch.optim 
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()

            metric.add(float(l)*len(y),accuracy(y_hat,y),y.size().numel())
        
        else:
            l.sum().backward()
            updater(x.shape[0])

            metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
        
        print(metric[0]/metric[2],metric[1]/metric[2])

    return metric[0]/metric[2],metric[1]/metric[2]

def train_ch3(net,train_loader,test_loader,loss,num_epochs,updater):
    for epoch in range(num_epochs):
        test_acc = evaluate_accuracy(net,test_loader)
        train_metrics = train_epoch_ch3(net,train_loader,loss,updater)

    train_loss, train_acc = train_metrics

    return train_loss,train_acc,test_acc


num_epochs = 10
if __name__=="__main__":
    tl,ta,tc = train_ch3(net,train_loader,test_loader,cross_entropy,num_epochs,updater)
    print(tl,ta,tc)