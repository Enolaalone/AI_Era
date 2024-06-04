import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

L = []
E = []
n=1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.active = nn.ReLU()
        self.fc1 = nn.Linear(720,10,bias=False)

    def forward(self, input_):
        batch_size = input_.size(0)
        # print(input_.size())
        input_ = self.pool(self.active(self.conv1(input_)))
        input_ = self.pool(self.active(self.conv2(input_)))
        input_ = input_.view(batch_size, -1)
        # print(input_.size(1))
        return self.fc1(input_)

def train(epoch):
    global n
    CBL = 0
    for idx,(data,target) in enumerate(train_loader,1):
        data,target = data.to(device),target.to(device)

        output = model(data)
        loss = criterion(output,target)
        CBL += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            L.append(loss.item()/idx)
            E.append(n)
            n += 1
            print(f'Epoch: {epoch},Loss: {100*CBL/idx:.4f}%')
    print()


def test(epoch):
    Loss = 0
    with torch.no_grad():
        for idx,(data,target) in enumerate(test_loader,1):
            data,target = data.to(device),target.to(device)
            output = model(data)
            loss = criterion(output,target)
            Loss += loss.item()
            if idx % 100 == 0:
                print(f'Epoch: {epoch},Loss: {100*Loss/idx:.2f}%')
        print()


def draw():
    plt.plot(E,L)
    plt.xlabel('HundredEpoch')
    plt.ylabel('Loss')
    plt.show()
# Data
batch_size = 64
pic_size = 28
EPOCH = 5

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])
train_set = datasets.MNIST(root='../MNIST_data',
                           train=True,
                           download=False,
                           transform=transform)

train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True)

test_set = datasets.MNIST(root='../MNIST_data',
                          train=False,
                          download=False,
                          transform=transform)

test_loader = DataLoader(dataset=test_set,
                         batch_size=batch_size,
                         shuffle=False)


#Modal
model = Net().to(device)
#Loss optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)
#Train
if __name__ == "__main__":
    for epoch in range(EPOCH):
        train(epoch)
        test(epoch)
    draw()