import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ep = []
Loss = []
n = 0


class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        super(MyDataset, self).__init__()
        self.x_data = torch.Tensor(x_data).view(-1, 1)
        self.y_data = torch.Tensor(y_data).view(-1, 1)
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc1(x)


def train():
    global n
    n += 1
    MSE = 0
    for idx, (x, y) in enumerate(train_dataloader, 1):
        x, y = x.to(device), y.to(device)
        pre_y = model(x)
        loss = criterion(pre_y, y)

        MSE += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep.append(n)
        Loss.append(loss.item())
        print(f"loss:{MSE / idx}")
    print()


def test():
    mse = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(train_dataloader, 1):
            x, y = x.to(device), y.to(device)
            pre_y = model(x)
            loss = criterion(pre_y, y)
            mse += loss.item()

            print(f"loss:{mse / idx}")

def draw():
    plt.plot(ep,Loss)
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.show()

# data
batch_size = 1
epoch_num = 10

train_set = MyDataset([1.0, 2.0, 3.0], [2.1, 4.1, 6.1])
train_dataloader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)

text_set = MyDataset([4.0, 5.0], [8.1, 10.1])
text_dataloader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)

# model
model = Net().to(device)

# loss & optimizer
criterion = nn.MSELoss()  # MSE损失
optimizer = optim.SGD(model.parameters(), lr=0.01)

# train
if __name__ == "__main__":
    for epoch in range(epoch_num):
        train()
        test()
    draw()

