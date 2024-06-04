import torch
import torch.nn as nn

input_size = 4
hidden_size = 2
sequence_length = 3
batch_size = 1

cell = nn.RNNCell(input_size=input_size,hidden_size= hidden_size)#确定的输入输出维度

dataset = torch.randn(sequence_length, batch_size,input_size)

hidden = torch.zeros(batch_size, hidden_size)#引层

print(dataset.shape,'\n')
for idx,inputs in enumerate(dataset):
    print(inputs.shape)
    print(hidden.shape)
    print()
    hidden = cell(inputs, hidden)

