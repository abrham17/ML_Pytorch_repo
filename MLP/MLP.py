import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

nObs = 1000
yNormal = torch.distributions.Normal(loc = 0.0 , scale=10.0)
yNoise = yNormal.sample([nObs])

xObs = 10 * torch.rand([nObs]) - 5
yObs = xObs**3 + xObs**2 + 25 * np.sin(2 * xObs) + yNoise
print(yObs.size())
print(xObs.size())
nTest = 200
yTestN = yNormal.sample([nTest])
xTest = 10 * torch.rand([nTest]) - 5 
yTest =  xTest**3 + xTest**2 + 25 * np.sin(2 * xTest) + yTestN

d = pd.DataFrame({'xObs' : xObs, 'yObs' : yObs})
nInput  = 1
nHidden = 10
nOutput = 1

class MLPcondensed(nn.Module):
    def __init__(self, nInput, nHidden, nOutput):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(nInput, nHidden),
            nn.ReLU(),
            nn.Linear(nHidden, nHidden),
            nn.ReLU(),
            nn.Linear(nHidden, nHidden),
            nn.ReLU(),
            nn.Linear(nHidden, nOutput)
        )

    def forward(self, x):
        return(self.layers(x))

model = MLPcondensed(nInput, nHidden, nOutput)
class nonLinearRegressionData(Dataset):
    def __init__(self, xObs, yObs):
        self.xObs = torch.reshape(xObs, (len(xObs), 1))
        self.yObs = torch.reshape(yObs, (len(yObs), 1))

    def __len__(self):
        return(len(self.xObs))

    def __getitem__(self, idx):
        return(xObs[idx], yObs[idx])

# instantiate Dataset object for current training data
d = nonLinearRegressionData(xObs, yObs)
train_dataloader = DataLoader(d, batch_size=25 , shuffle=True)

#Instantiating test dataset object
d_test = nonLinearRegressionData(xTest, yTest)

test_dataloader  = DataLoader(d_test, batch_size=50)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
nTrainSteps = 5000

for epoch in range(0, nTrainSteps):

  # Set current loss value
  current_loss = 0.0

  # Iterate over the DataLoader for training data
  model.train()
  for i, data in enumerate(train_dataloader, 0):
    # Get inputs
    inputs, targets = data
    # Zero the gradients
    optimizer.zero_grad()
    # Perform forward pass (make sure to supply the input in the right way)
    outputs = model(torch.reshape(inputs, (len(inputs), 1))).squeeze()
    # Compute loss
    loss = loss_function(outputs, targets)
    # Perform backward pass
    loss.backward()
    # Perform optimization
    optimizer.step()
    # Print statistics
    current_loss += loss.item()

  if (epoch + 1) % 250 == 0:
      model.eval()
      test_error = 0
      with torch.no_grad():
          for j, test_data in enumerate(test_dataloader,0):
              test_x, test_y = test_data
              test_out = model(torch.reshape(test_x, (len(test_x), 1))).squeeze()
              test_loss = loss_function(test_out, test_y)
              test_error += test_loss.item()          
      print('Loss after epoch %5d: %.3f' %
            (epoch + 1, test_error))
      test_error = 0.0
