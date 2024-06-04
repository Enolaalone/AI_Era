import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
a = ['h', 'e', 'l', 'o']
batch_size = 1
input_size = 4
hidden_size = 8
num_layers = 2
embedding_size = 10

x_data = [0, 1, 2, 2, 3]
y_data = [3, 0, 2, 3, 2]
seq_length = len(x_data)  # 序列长度
one_hot = [[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]]

input = torch.LongTensor(x_data).view(batch_size, seq_length)  # x_data 升为二维
labels = torch.LongTensor(y_data)


class Net(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size, num_layers, embedding_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size

        # Embedding层 ： batch ~ seq  = [1, 5]
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size)
        # 输入要求: batch ~ seq ~ input_size  使用batch_first转置矩阵: seq ~ batch ~ input_size
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # Linear
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, inputs):
        # Embedding层： batch ~ seq ~ embedding = [1, 5, 10]
        inputs = self.embedding(inputs)
        # print(inputs.shape)

        # RNN层:  batch ~ seq ~ hidden = [1,5,8]
        hidden = self.init_hidden()
        hidden = (hidden[0].to(device), hidden[1].to(device))

        outputs, _ = self.lstm(inputs, hidden)  # 先out层 再当前层hidden
        # print(outputs.shape)  # 会输出3维

        # Linear层
        return self.linear(outputs)

    def init_hidden(self):  # 需要两个hidden的输入
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))


def train():
    for epoch in range(15):
        loss = 0
        correct = 0  # 正确率
        # Forward
        outputs = model(input).view(-1, input_size)  # 加view转为2维
        loss = criterion(outputs, labels)  # criterion输入为[B,W]和[C]

        _, predicted = torch.max(outputs.data, dim=1)  # 按照第二维度：列；查询二维张量output行中[1,0,0,0]中最大的下标
        correct = (predicted == labels).sum().item()

        print(''.join(a[predicted[i]] for i in range(len(predicted))), end='')  # .jion输出字符串
        print(f'\nLoss: {loss.item():.4f}, Accuracy: {100 * correct / seq_length}%\n')

        # Backward+Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# model
model = Net(input_size, batch_size, hidden_size, num_layers, embedding_size)
model.to(device)
input, labels = input.to(device), labels.to(device)

# 损失 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# train
train()
