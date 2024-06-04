import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataset(Dataset):
    def __init__(self, seq_size):
        super(MyDataset, self).__init__()
        self.x = torch.zeros(8).view(-1, seq_size).long()  # 转为长整型张量
        self.y = torch.LongTensor([0, 1, 0, 2, 3, 4, 5, 6]).view(-1, seq_size)  # 转为长整型张量
        self.len = self.x.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):  # Data_loader会自动调用
        return self.len


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size  # hidden初始化
        # Embedding
        self.embedding = nn.Embedding(input_size, hidden_size)  # dic_num字典索引长度 , embedding_size嵌入层大小
        # RNN
        self.rnn = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True)  # 输入张量结构 batch~seq~hidden； batch在第一位 避免在外部转置
        # Linear
        self.linear = nn.Linear(hidden_size, output_size)  # output_size句子序列seq长度

    def forward(self, input_, hidden):
        embedded = self.embedding(input_)
        output, hidden = self.rnn(embedded, hidden)
        return self.linear(output)

    def __hidden_init__(self, batch_size, num_layers):
        return torch.zeros(num_layers, batch_size, self.hidden_size)  # shape: layer~batch~hidden


def train():
    for (data, target) in train_loader:
        data, target = data.to(device), target.to(device)
        hidden = model.__hidden_init__(batch_size, num_layers).to(device)  # hidden_shape: layers~batch~hidden_size
        output = model(data, hidden)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())


# data
dic = ['i', 'l', 'k', 'e', 'y', 'o', 'u']
input_size = len(dic)
hidden_size = 20
num_layers = 2
batch_size = 1
output_size = 8
train_dataset = MyDataset(output_size)
train_loader = DataLoader(dataset=train_dataset,  # 内部自动调用Dataset def __len__
                          batch_size=batch_size,
                          shuffle=True)

# Model
model = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers).to(device)
# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# train
if __name__ == "__main__":
    for epoch in range(100):
        train()
