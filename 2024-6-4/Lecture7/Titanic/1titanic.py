import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TitanicDataset(Dataset):
    def __init__(self):
        super(TitanicDataset, self).__init__()
        self.data = np.loadtxt('gender_submission.csv', delimiter=',', dtype=np.float32)
        self.len = self.data.shape[0]  # 数据大小
        self.x = torch.from_numpy(self.data[:, :-1])
        self.y = torch.from_numpy(self.data[:, [-1]])


    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


class Titanic(nn.Module):
    def __init__(self):
        super(Titanic, self).__init__()
        self.ln1 = nn.Linear(1, 48)
        self.ln2 = nn.Linear(48, 24)
        self.ln3 = nn.Linear(24, 1)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.ReLU()

    def forward(self, inputs):
        inputs = self.act2(self.ln1(inputs))
        inputs = self.act2(self.ln2(inputs))
        return self.act1(self.ln3(inputs))


def train():
    BCE = 0
    for idx, (x, y) in enumerate(train_loader, 1):
        x, y = x.to(device), y.to(device)
        out = model(x)
        print(out)
        loss = criterion(out, y)

        BCE += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'BCE:{BCE / idx}')


# Data
epoches = 10
batch_size = 2
train_data = TitanicDataset()
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# Model
model = Titanic().to(device)
# loss & Optimizer
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train
if __name__ == "__main__":
    for epoch in range(epoches):
        train()
