import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 28 * 28
output_size = 10
EPOCHS = 5

n = 1
N = []
Loss = []


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        self.active1 = nn.ReLU()
        self.active2 = nn.Sigmoid()

    def forward(self, inp):
        inp = self.active1(self.fc1(inp))
        inp = self.active1(self.fc2(inp))
        inp = self.active1(self.fc3(inp))
        return self.active2(self.fc4(inp))


def train(epoch):
    global n
    Av_Loss = 0
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        # print(data.shape, target.shape)
        data = data.view(-1, input_size)
        # forward+ backward+ update
        output = model(data)
        loss = criterion(output, target)

        Av_Loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            N.append(n)
            Loss.append(Av_Loss / batch_idx)
            n += 1
            print(f'Train Epoch:{epoch}  Loss:{Av_Loss / batch_idx:.3f}')


def test():
    with torch.no_grad():
        correct = 0
        for idx, (data, target) in enumerate(test_loader, 1):
            data, target = data.to(device), target.to(device)
            data = data.view(-1, input_size)
            output = model(data)

            pred = output.max(dim=1,keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(f'Epoch:{epoch}  Accuracy:{correct / len(test_set):.3f}')




def draw():
    plt.plot(N, Loss)
    plt.xlabel('HundredBatch')
    plt.ylabel('Loss')
    plt.show()


# Data
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
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

# Model
model = Net(input_size, output_size).to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train
if __name__ == "__main__":
    for epoch in range(EPOCHS):
        train(epoch)
        test()
    draw()
