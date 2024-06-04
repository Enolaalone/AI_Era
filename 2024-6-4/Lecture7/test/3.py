
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L = []
N = []
n = 1


class MyData(Dataset):
    def __init__(self):
        super(MyData, self).__init__()
        self.data = np.loadtxt('../diabetes.csv', delimiter=',', dtype=np.float32)  # 读取数据
        self.x = torch.from_numpy(self.data[:, :-1])
        self.y = torch.from_numpy(self.data[:, [-1]]).long()
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

class Inception(nn.Module):
    def __init__(self,input_size):
        super( Inception, self).__init__()
        #branch1
        self.pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.conv_1x1_24 = nn.Conv2d(in_channels=input_size,out_channels=24,kernel_size=1)
        #branch2
        self.conv_1x1_16_2 = nn.Conv2d(in_channels=input_size,out_channels=16,kernel_size=1)
        #branch3
        self.conv_1x1_16_3 = nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=1)
        self.conv_5x5_24 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5,padding=2)
        #branch4
        self.conv_1x1_16_4 = nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=1)
        self.conv_3x3_24_1 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3,padding=1)
        self.conv_3x3_24_2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3,padding=1)

    def forward(self, input_):
        branch1 = self.conv_1x1_24(self.pool(input_))
        branch2 = self.conv_1x1_16_2(input_)
        branch3 = self.conv_5x5_24(self.conv_1x1_16_3(input_))
        branch4 = self.conv_3x3_24_2 (self.conv_3x3_24_1((self.conv_1x1_16_4(input_))))

        input_ = torch.cat((branch1,branch2,branch3,branch4), dim=1)
        # print(input_.shape)
        return input_

class Net(nn.Module):
    def __init__(self,input_size,output_size):
        super(Net, self).__init__()
        #Conv2d
        self.conv1 = nn.Conv2d(input_size,10,5,padding=2)
        self.conv2 = nn.Conv2d(88,20,5,padding=2)
        #inception
        self.inception1 = Inception(10)
        self.inception2 = Inception(20)
        #activate
        self.activate = nn.ReLU()
        #Pool
        self.pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        #Linear
        self.linear1 = nn.Linear(8,output_size,bias=False)
    def forward(self,inp):
        batch = inp.size(0)
        inp = self.inception1(self.pool(self.activate(self.conv1(inp))))
        inp = self.inception2(self.pool(self.activate(self.conv2(inp))))
        # print(inp.shape)
        inp = inp.view(64,88,-1)#用=更新inp
        return self.linear1(inp)

def train(epoch):
    global n
    for idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        data = data.view(8,1,-1,8)
        outputs = model(data)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            L.append(loss.item())
            N.append(n)
            n += 1
            # print(f'Epoch {epoch},idx{idx}')
            # print(f'Loss {loss.item():.4f}')
    # print()


def draw():
    plt.plot(N, L)
    plt.xlabel('TenBatch')
    plt.ylabel('Loss')
    plt.show()


# Data
input_size = 1
batch_size = 64

train_set = MyData()
train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True)

# Model
model = Net(input_size,input_size).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train
if __name__ == "__main__":
    for epoch in range(1000):
        train(epoch)
    draw()
