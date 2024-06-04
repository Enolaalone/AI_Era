import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional=True):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = 2 if bidirectional else 1#双向RNN

        # Embedding
        self.embedding = nn.Embedding(input_size, hidden_size)
        # RNN GRU
        self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional)  # bidirectional为bool；双向RNN
        # Linear
        self.fc = nn.Linear(hidden_size * self.bidirectional, output_size)#hidden为拼接后长度

    def forward(self, input, seq_length):  # seq为张量
        batch_size = input.size(0)
        # input = input.t()  # 转置为 seq·batch
        # print(input.shape)

        # embedding
        embedded = self.embedding(input)#batch·seq
        # print(input.shape)

        # Rnn
        hidden = self.init_hidden(batch_size)#hidden初始化
        packed = nn.utils.rnn.pack_padded_sequence(embedded, seq_length.to('cpu'),batch_first=True)  # 去除填充0打包,  seq_lenggth设置到cpu
        output,hidden= self.rnn(packed, hidden)
        # print(hidden.shape)

        if (self.bidirectional == 2):  # 检测是否要拼接
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)  # 沿着 embedding基维拼接
        else:
            hidden = hidden[-1]
        # print(hidden.shape)

        return self.fc(hidden)


    def init_hidden(self, batch_size):  # hidden初始化
        return torch.zeros(self.num_layers * self.bidirectional, batch_size, self.hidden_size).to(device)  # 保证数据在GPU上
