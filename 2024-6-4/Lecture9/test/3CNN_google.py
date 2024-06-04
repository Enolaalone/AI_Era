import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Inception(nn.Module):
    def __init__(self,input_size):
        super( Inception, self).__init__()
        #branch1
        self.pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.conv_1x1_24 = nn.Conv2d(in_channels=input_size,out_channels=24,kernel_size=1)
        #branch2
        self.conv_1x1_16_2 = nn.Conv2d(in_channels=input_size,out_channels=16,kernel_size=1)
        #branch3
        self.conv_1x1_16_3 = nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=1)
        self.conv_5x5_24 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5,padding=2)
        #branch4
        self.conv_1x1_16_4 = nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=1)
        self.conv_3x3_24_1 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3,padding=1)
        self.conv_3x3_24_2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3,padding=1)

    def forward(self, input_):
        branch1 = self.conv_1x1_24(self.pool(input_))
        branch2 = self.conv_1x1_16_2(input_)
        branch3 = self.conv_5x5_24(self.conv_1x1_16_3(input_))
        branch4 = self.conv_3x3_24_2 (self.conv_3x3_24_1((self.conv_1x1_16_4(input_))))

        input_ = torch.cat((branch1,branch2,branch3,branch4), dim=1)
        # print(input_.shape)
        return input_

class Net(nn.Module):
    def __init__(self,input_size,output_size):
        super(Net, self).__init__()
        #Conv2d
        self.conv1 = nn.Conv2d(input_size,10,5)
        self.conv2 = nn.Conv2d(88,20,5)
        #inception
        self.inception1 = Inception(10)
        self.inception2 = Inception(20)
        #activate
        self.activate = nn.ReLU()
        #Pool
        self.pool = nn.MaxPool2d(kernel_size=2)
        #Linear
        self.linear1 = nn.Linear(1408,output_size,bias=False)
    def forward(self,inp):
        batch = inp.size(0)
        inp = self.inception1(self.pool(self.activate(self.conv1(inp))))
        inp = self.inception2(self.pool(self.activate(self.conv2(inp))))
        inp = inp.view(batch,-1)#用=更新inp
        return self.linear1(inp)


def train(epoch):
    for idx,(data,target) in enumerate(train_loader,1):
        data,target = data.to(device),target.to(device)
        # print(data.size(),target.size())
        output = model(data)
        loss = criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (idx) % 100 == 0:
            print(f'Train Epoch: {epoch} , Loss: {loss.item():.4f}')


def test():
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 数据转移
            outputs = model(data)
            _, predicted = torch.max(outputs.data, dim=1)  # 按列从左到右扫描数据返回最大值序号到predicted
            total += target.size(0)
            correct += (predicted == target).sum().item()
        print(f'正确率：{100 * correct / total}')
        print()

# Data
input_size = 1
output_size = 10
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

train_set = datasets.MNIST(root='../MNIST_data',
                           train=True,
                           download=False,
                           transform=transform)
train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True)
test_set = datasets.MNIST(root='../MNIST_data',
                          train=False,
                          download=False,
                          transform=transform)
test_loader = DataLoader(dataset=train_set,
                         batch_size=batch_size,
                         shuffle=False)
#Model
model=Net(input_size,output_size).to(device)

#Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

#train
if __name__ =="__main__":
    for epoch in range(10):
        train(epoch)
        test()