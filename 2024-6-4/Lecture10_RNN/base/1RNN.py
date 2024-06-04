import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
a = ['h', 'e', 'l', 'o']

batch_size = 1
input_size = 4
hidden_size = 4
num_layers = 2

x_data = [0, 1, 2, 2, 3]
y_data = [3, 0, 2, 3, 2]
seq_length = len(x_data)  # 序列长度
one_hot = [[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]]

x_one_hot = [one_hot[x] for x in x_data]

input = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data)


class Net(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size, num_layers):
        super(Net, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, inputs):
        hidden = self.init_hidden()
        hidden = hidden.to(device)
        outputs, _ = self.rnn(inputs, hidden)  # 先out 层再 当前层hidden
        print(outputs.shape)  # 会输出3维
        return outputs

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)


def train():
    for epoch in range(15):
        loss = 0
        correct = 0  # 正确率
        # Forward
        outputs = model(input).view(seq_length, -1)  # 加view转为2维
        loss = criterion(outputs, labels)  # criterion输入为[B,W]和[C]

        _, predicted = torch.max(outputs.data, dim=1)  # 查询二维张量output行中[1,0,0,0]中最大的下标
        correct = (predicted == labels).sum().item()

        print(''.join(a[predicted[i]] for i in range(len(predicted))), end='')
        print(f'\nLoss: {loss.item():.4f}, Accuracy: {100 * correct / seq_length}%\n')

        # Backward+Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# model
model = Net(input_size, batch_size, hidden_size, num_layers)
model.to(device)
input, labels = input.to(device), labels.to(device)

# 损失 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# train
train()
