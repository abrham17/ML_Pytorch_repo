'''
1) Importing necessary modules and libraries
'''
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

'''
2) The true model and training data
'''


def goalFun(x):
    return(x**3 - x**2 + 25 * np.sin(2*x))

# # create linear sequence (x) and apply goalFun (y)
# x = np.linspace(start = -5, stop =5, num = 1000)
# y = goalFun(x)

# # plot the function
# d = pd.DataFrame({'x' : x, 'y' : y})
# sns.lineplot(data = d, x = 'x', y = 'y')
# plt.show()

##################################################
## generate training data (with noise)
##################################################

nObs = 1000 # number of observations

# get noise around y observations
yNormal = torch.distributions.Normal(loc=0.0, scale=10)
yNoise  = yNormal.sample([nObs])


# get observations
xObs = 10*torch.rand([nObs])-5    # uniform from [-5,5]
yObs = xObs**3 - xObs**2 + 25 * torch.sin(2*xObs) + yNoise

# get observations
n_test = 200
y_testN = yNormal.sample([n_test])
x_test = 10*torch.rand([n_test])-5    # uniform from [-5,5]
y_test = x_test**3 - x_test**2 + 25 * torch.sin(2*x_test) + y_testN

# # plot the data
# d = pd.DataFrame({'xObs' : xObs, 'yObs' : yObs})
# sns.scatterplot(data = d, x = 'xObs', y = 'yObs')
# plt.show()

'''
3) Defining the MLP using PyTorch’s built-in modules
'''

##################################################
## network dimension parameters
##################################################

nInput  = 1
nHidden = 10
nOutput = 1

##################################################
## set up multi-layer perceptron w/ PyTorch
##    -- explicit version --
##################################################

# class MLPexplicit(nn.Module):
#     '''
#     Multi-layer perceptron for non-linear regression.
#     '''
#     def __init__(self, nInput, nHidden, nOutput):
#         super(MLPexplicit, self).__init__()
#         self.nInput  = nInput
#         self.nHidden = nHidden
#         self.nOutput = nOutput
#         self.linear1 = nn.Linear(self.nInput, self.nHidden)
#         self.linear2 = nn.Linear(self.nHidden, self.nHidden)
#         self.linear3 = nn.Linear(self.nHidden, self.nHidden)
#         self.linear4 = nn.Linear(self.nHidden, self.nOutput)
#         self.ReLU    = nn.ReLU()

#     def forward(self, x):
#         h1 = self.ReLU(self.linear1(x))
#         h2 = self.ReLU(self.linear2(h1))
#         h3 = self.ReLU(self.linear3(h2))
#         output = self.linear4(h3)
#         return(output)

# model = MLPexplicit(nInput, nHidden, nOutput)
# #Displaying each network parameters
# for p in model.parameters():
#     print(p.detach().numpy().round(2))

'''
4) More condensed definition of the MLP
'''

##################################################
## set up multi-layer perceptron w/ PyTorch
##    -- condensed version --
##################################################

class MLPcondensed(nn.Module):
    '''
    Multi-layer perceptron for non-linear regression.
    '''
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

# which model to use from here onwards
# model = mlpExplicit
#model = mlpCondensed

'''
5) Preparing the training data
'''

class nonLinearRegressionData(Dataset):
    '''
    Custom 'Dataset' object for our regression data.
    Must implement these functions: __init__, __len__, and __getitem__.
    '''

    def __init__(self, xObs, yObs):
        self.xObs = torch.reshape(xObs, (len(xObs), 1))
        self.yObs = torch.reshape(yObs, (len(yObs), 1))

    def __len__(self):
        return(len(self.xObs))

    def __getitem__(self, idx):
        return(xObs[idx], yObs[idx])

# instantiate Dataset object for current training data
d = nonLinearRegressionData(xObs, yObs)

# instantiate DataLoader
#    we use the 4 batches of 25 observations each (full data  has 100 observations)
#    we also shuffle the data
train_dataloader = DataLoader(d, batch_size=25 , shuffle=True)

#Instantiating test dataset object
d_test = nonLinearRegressionData(x_test, y_test)

#Instantiate test dataloader
test_dataloader  = DataLoader(d_test, batch_size=50)

# # Iterating over the dataloader and showing 
# # the input and output of the dataset
# for i, data in enumerate(train_dataloader, 0):
#     input, target = data
#     print("In: ", input)
#     print("Out:", target,"\n")

'''
6) Training the model
'''
##################################################
## training the model
##################################################

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
nTrainSteps = 5000

# Run the training loop
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

# Process is complete.
print('Training process has finished.')

yPred = np.array([model.forward(torch.tensor([o])).detach().numpy() for o in x_test]).flatten()

# plot the data
d = pd.DataFrame({'xObs' : x_test.detach().numpy(),
                  'yObs' : y_test.detach().numpy(),
                  'yPred': yPred})
dWide = pd.melt(d, id_vars = 'xObs', value_vars= ['yObs', 'yPred'])
sns.scatterplot(data = dWide, x = 'xObs', y = 'value', hue = 'variable', alpha = 0.7)
x = np.linspace(start = -5, stop =5, num = 1000)
y = goalFun(x)
plt.plot(x,y, color='g', alpha = 0.5)
plt.show()
