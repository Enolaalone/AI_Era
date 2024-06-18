import torch 
import random
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据生成----------------------------------
def synthetic_data(w,b,num_example):
    # 生成 y = Xw+b+噪声
    X= torch.normal(0,1,(num_example,len(w)))# shape [10,2]
    # print(X)
    y=torch.matmul(X,w)+b # output

    # 添加噪声
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])# 权重
true_b = 4.2# 偏置量
features , labels = synthetic_data(true_w,true_b,1000)

features,labels = features.to(device),labels.to(device)#GPU

# DataLoader---------------------------------------------
def data_iter(features,labels,batch_size):
    num_examples =len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0,num_examples,batch_size):
        batch_indices = indices[i:min(i+batch_size,num_examples)]
        yield features[batch_indices],labels[batch_indices]


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2,1)

    def forward(self,x):
        return self.fc(x)

def train(epoch,features,labels,batch_size):
    for idx,(feature,label) in enumerate(data_iter(features,labels,batch_size),1):
        feature,label =feature.to(device),label.to(device)
        output = model(feature)
        loss = criterion(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx %10==0:
            print(f'epoch{epoch},idx{idx},loss{loss.item()}')


#Data

batch_size = 10
# Model
model = Net().to(device)

# Loss & optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.03)

# train
if __name__ =='__main__':
    for epoch in range(100):
        train(epoch,features,labels,batch_size)
