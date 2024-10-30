import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets , transforms
from torchvision.transforms import ToTensor

all_transforms = transforms.Compose([
                                    transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4914, 0.4822, 0.4465] , [0.2023, 0.1994, 0.2010]) # Use standard normalization values for CIFAR-10
                ]) 

train_set = datasets.CIFAR10(train=True , root='./data' , transform=all_transforms , download=True)
test_set = datasets.CIFAR10(train=False , root='./data' , transform=all_transforms , download=True)

train_loader = DataLoader(dataset=train_set , batch_size=50 , shuffle=True)
test_loader = DataLoader(dataset=test_set , batch_size=5000)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels= 3 , out_channels=32 , kernel_size=3),
            nn.Conv2d(in_channels=32 , out_channels=32 , kernel_size=3),
            nn.MaxPool2d(kernel_size=2 , stride=2),
            nn.Conv2d(in_channels=32 , out_channels=64 ,kernel_size=3),
            nn.Conv2d(in_channels=64 , out_channels=64 , kernel_size=3),
            nn.MaxPool2d(kernel_size=2 , stride=2),
        )
        self.sequence_2 = nn.Sequential(
            nn.Linear(1600 , 128),
            nn.ReLU(),
            nn.Linear(128 , 10))

    def forward(self , x):
        out = self.sequence(x)
        out = out.reshape(out.size(0) , -1)
        out = self.sequence_2(out)
        return out

model = Model()
critearion= nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters() , lr=0.001)
num_epochs = 10
def train(epochs):
    for epoch in range(epochs):
        model.train()
        for x,y in train_loader:
            yhat = model(x)
            loss = critearion(yhat , y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], loss {:.4f}',format(
                        epoch+1,num_epochs , loss.item()
        ))

train(1)
def test():
    correct = 0
    model.eval()
    for x,y in test_loader:
        yhat = model(x)
        loss = critearion(yhat , y)
        _ , yhat = torch.max(yhat.data , 1)
        total += y.size(0)
        correct += (yhat == y).sum().item()
        print("Accuracy of data on {} are {}%".format(50000 , 100 * (correct/total)))

test()