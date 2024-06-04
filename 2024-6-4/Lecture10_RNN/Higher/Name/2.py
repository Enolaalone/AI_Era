import torch
import torch.nn as nn
import NameDataset
import RNN
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def TensorToGPU(tenser):
    tenser = tenser.to(device)


def nameASCII(name):  # 名字转ASCII
    name_list = []
    for char in name:
        name_list.append(ord(char))
    return name_list, len(name_list)


def get_tensor(names, countries):
    seq_and_length = [nameASCII(sl) for sl in names]
    length = torch.LongTensor([sl[1] for sl in seq_and_length])
    # print(seq_and_length,length)
    seq_name = [sl[0] for sl in seq_and_length]
    countries = torch.LongTensor(countries)

    # 名字张量扩充
    name_tensors = torch.zeros(len(length), torch.max(length, dim=0)[0]).long()
    for idx, (nm, l) in enumerate(zip(seq_name, length)):  # 替换
        name_tensors[idx, :l] = torch.LongTensor(nm)  # 用字母ASCII张量替换前[0,l)

    # 排序
    length, sorts = length.sort(dim=0, descending=True)
    # print(sorts)
    return name_tensors[sorts], length, countries[sorts]  # 返回排序后的张量


def train():
    t_loss = 0
    for idx, (name, country_id) in enumerate(train_loader, 1):
        name_T, length, country_t = get_tensor(name, country_id)
        name_T, length, country_t = name_T.to(device), length.to('cpu'), country_t.to(device)  # 转移模型


        outputs = model(name_T, length)
        loss = criterion(outputs, country_t)  # [name,Letter]映射[country_id]

        t_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = 0
        _,pre = torch.max(outputs, dim=1)
        for idx, p in enumerate(pre):
            if(country_t[idx].item()==p):
                correct+=1
        print(f'correct : {100 * correct / len(pre)}%,{correct}')

        if idx % 10 == 0:
             print(f'Epoch: {idx} | Loss: {t_loss / (idx * name_T.size(0)):.4f}')


def test():
    total = 0
    # print(total)
    with torch.no_grad():
        for idx, (names, country_id) in enumerate(test_loader, 1):
            name_t, length, country_t = get_tensor(names, country_id)
            name_t, length, country_t = name_t.to(device), length.to('cpu'), country_t.to(device)
            # print(country_t)
            outputs = model(name_t, length)
            # pre = outputs.max(dim=1, keepdim=True)[1]  # 返回预测值下标
            # correct += country_t.eq(pre.view(-1)).sum().item()  # 比较预测正确个数

            # correct = 0
            # _, pre = torch.max(outputs.data, dim=1)
            # total = len(pre)
            # for i, p in enumerate(pre):
            #     if country_t[i] == p:
            #         correct += 1
            # print(f'correct : {100 * correct / total}%,{correct}')


# Data
batch_size = 64
hidden_size = 100
num_layers = 2
num_epochs = 100
n_chars = 128

train_set = NameDataset.NameDataset(is_train_set=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_set = NameDataset.NameDataset(is_train_set=False)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

country_num = train_set.get_country_num()  # 国家数目,之后作为模型输出所有国家的可能性

# model
model = RNN.RNN(input_size=n_chars, output_size=country_num, hidden_size=hidden_size, num_layers=num_layers,
                bidirectional=True)
model.to(device)
# 损失&优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# train
if __name__ == "__main__":
    for epoch in range(num_epochs):
        train()
        test()
