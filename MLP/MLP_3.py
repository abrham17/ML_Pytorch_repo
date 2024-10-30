import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader , Dataset
torch.manual_seed(42)
# Create data
start = 0.00
end = 1.00
step = 0.01

X = torch.arange(start, end, step).view(-1, 1)  # Reshape to be a 2D tensor
z  = torch.randn(100).view(-1,1)
y = X**3 + X**2 + 25 * torch.sin(2 * X) + z

# Split data into train and test sets
train_split = int(0.7 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


class Data(Dataset):
    def __init__(self , X , y):
        self.X = X
        self.y = y
        self.len = self.X.shape[0]
    def __getitem__(self , index):
        return self.X[index] , self.y[index]
    def __len__(self):
      return self.len

train_Data = Data(X_train , y_train)
test_Data = Data(X_test , y_test)

train_dataloader = DataLoader(dataset=train_Data , batch_size=10 , shuffle=True)
test_dataloader = DataLoader(dataset=test_Data , batch_size=10 , shuffle=True)
def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    
    plt.scatter(test_data, test_labels, c="r", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="g", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


class LinearRegressionModel(nn.Module): 
    def __init__(self, nInput, nHidden, nOutput):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(nInput, nHidden),
            nn.Sigmoid(),
            nn.Linear(nHidden, nHidden),
            nn.Sigmoid(),
            nn.Linear(nHidden, nHidden),
            nn.Sigmoid(),
            nn.Linear(nHidden , nHidden),
            nn.Sigmoid(),
            nn.Linear(nHidden, nOutput)
        )

    def forward(self, x):
        return(self.layers(x))


torch.manual_seed(42)
# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel(1 , 10, 1)

loss_fn = nn.MSELoss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters() , lr=0.02)
# Set the number of epochs (how many times the model will pass over the training data)
epochs = 4000

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []
def train(epochs):
  for epoch in range(epochs):
      model_0.train()
      for x,y in train_dataloader:
        # Trainin

        # 1. Forward pass on train data using the forward() method inside 
        y_pred = model_0(X_train)
        # print(y_pred)

        # 2. Calculate the loss (how different are our models predictions to the ground truth)
        loss = loss_fn(y_pred, y_train)

        # 3. Zero grad of the optimizer
        optimizer.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Progress the optimizer
        optimizer.step()
        """
        if epoch % 500 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
        """

  # Find our model's learned parameters
train(epochs)
def test():
    for x,y in test_dataloader:
          # Testing
            model_0.eval()

            with torch.no_grad():
              # 1. Forward pass on test data
              test_pred = model_0(X_test)

              # 2. Caculate loss on test data
              test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type
          # Print out what's happening
with torch.no_grad():
    predictions = model_0(X_test).numpy()

plot_predictions(X_train, y_train, X_test, y_test, predictions=predictions)
test()