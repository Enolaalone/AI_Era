import torch
import torch.nn as nn

# DATA
seq_size = 10
batch_size = 64
input_size = 10
hidden_size = 20
cell_num = 10  # 单层网络长度
dataset = torch.randn(seq_size, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)

# model
model = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
# train
if __name__ == "__main__":
    print(hidden)
    for idx, data in enumerate(dataset, 1):
        for _ in range(cell_num):
            hidden = model(data, hidden)

    print(hidden)
