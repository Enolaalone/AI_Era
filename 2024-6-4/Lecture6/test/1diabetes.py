import torch
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L =[]
N =[]
n =1
class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset,self).__init__()
        self.data = np.loadtxt('../diabetes2.csv',delimiter =',',dtype = np.float32)
        self.x =torch.from_numpy(self.data[:,:-1])
        self.y =torch.from_numpy(self.data[:,[-1]])
        self.len = len(self.data)

    def __getitem__(self,index):
        return self.x[index],self.y[index]

    def __len__(self):
        return self.len

class Net(nn.Module):
    def __init__(self,input_size):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_size,96)
        self.fc2 = nn.Linear(96,48)
        self.fc3 = nn.Linear(48,24)
        self.fc4 = nn.Linear(24,12)
        self.fc5 = nn.Linear(12,1)
        self.activation = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
    def forward(self,inputs):
        inputs = self.fc1(self.activation(inputs))
        inputs = self.fc2(self.activation(inputs))
        inputs = self.fc3(self.activation(inputs))
        inputs = self.fc4(self.activation(inputs))
        inputs = self.fc5(self.activation(inputs))
        return self.Sigmoid(inputs)

def train(epoch):
    global n
    for idx ,(data,target) in enumerate(train_loader,1):
        data ,target = data.to(device),target.to(device)
        outputs = model(data)
        loss = criterion(outputs,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            L.append(loss.item())
            N.append(n)
            n+=1
def draw():
    plt.plot(N,L)
    plt.xlabel('TenN')
    plt.ylabel('Loss')
    plt.show()
#Data
input_size = 8
batch_size = 64
train_set = MyDataset()
train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True)

#Model
model = Net(input_size).to(device)

#Loss & optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)
#train

if __name__=="__main__":
    for epoch in range(1000):
        train(epoch)
    draw()