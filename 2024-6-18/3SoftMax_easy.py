
import torch
import torchvision 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#Data
inputs_size = 784
outputs_size = 10
batch_size = 64
transform = transforms.ToTensor()
train_set = torchvision.datasets.FashionMNIST(root='../data',
                                                 train= True,
                                                 transform=transform,
                                                 download=False
                                               )

test_set = torchvision.datasets.FashionMNIST(root='../data',
                                                 train= False,
                                                 transform=transform,
                                                 download=False
                                               )
train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_set,
                          batch_size=batch_size,
                          shuffle=False)
def accuracy(outputs,label):
    _,pred =torch.max(outputs,dim=-1)
    return (pred==label).sum().item()

def train(epoch):
    for idx,(feature,label) in enumerate(train_loader,1):
        feature,label = feature.to(device),label.to(device)

        outputs = model(feature)
        loss = criterion(outputs,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx%20 ==0:
            print(f"correct_num = {accuracy(outputs,label)}")


#Model

class Net(nn.Module):
    def __init__(self,inputs_size,outputs_size):
        super(Net,self).__init__()
        self.inputs_size =inputs_size
        self.fc = nn.Linear(inputs_size,outputs_size)

    def forward(self,x):
        x = x.view(-1,self.inputs_size)
        return self.fc(x)
    
model = Net(inputs_size,outputs_size).to(device)

#loss optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.05)

#train
if __name__=="__main__":
    for epoch in range(10):
        train(epoch)
