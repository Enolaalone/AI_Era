import torch
import torch.nn as nn

input_size = 4
hidden_size = 2
output_size = hidden_size
sequence_length = 3
batch_size = 1
layers = 2

RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=layers)

inputs = torch.randn(sequence_length,batch_size, input_size)
hidden = torch.zeros(layers, batch_size, hidden_size)

output, hidden = RNN(inputs, hidden)

print(output.size(),'layer,batch,hidden')
print(hidden.size())