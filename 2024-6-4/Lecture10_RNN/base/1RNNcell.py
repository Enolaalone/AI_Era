import torch
import torch.nn as nn
import torch.optim as optim

a = ['h', 'e', 'l', 'o']  # 字典

look_up = [[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]]

x_data = [0, 1, 2, 2, 3]  # hello
y_data = [3, 0, 2, 3, 2]  # ohlol
input_size = 4
hidden_size = 4
batch_size = 1
seq_length = 5
one_hot_x = [look_up[x] for x in x_data]  # 独热

inputs = torch.Tensor(one_hot_x).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)  # 交叉损失必须是LongTensor


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.cell = nn.RNNCell(self.input_size, self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.cell(input, hidden)

        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


def train():
    for epoch in range(15):
        loss = 0
        correct = 0
        hidden = model.init_hidden()#h0初始化
        for idx,(input, label) in enumerate(zip(inputs, labels)):
            hidden = model.forward(input, hidden)#前馈
            loss += criterion(hidden, label)
            _, prediction = torch.max(hidden.data, dim=1)#返回[0,1,0,0]中1的下标
            correct += (prediction == y_data[idx]).item()
            print(a[prediction], end='')
        print(f'\t正确率：{100*correct/seq_length}%')
        print(f"loss{loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# model
model = Net(input_size, hidden_size, batch_size)

# 损失 & 优化器

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# train
train()
