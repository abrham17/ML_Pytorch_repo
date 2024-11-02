import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(1)

# Generate synthetic data
x, y = make_circles(n_samples=1000, noise=0.03, random_state=42)

X = torch.from_numpy(x).type(torch.float)
Y = torch.from_numpy(y).type(torch.float)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def plot_decision_boundary(model, X, y):
    # Create a mesh grid over the feature space
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    # Prepare the input for prediction
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).type(torch.float).to(device)
    model.eval()
    with torch.no_grad():
        # Predict on the grid points
        zz = model(grid).detach().cpu().numpy()
        zz = zz.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, zz, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu, edgecolor='k')
    plt.show()

# Define the model class
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        return torch.sigmoid(self.layer_3(x))  # Use sigmoid here for final output

# Initialize model, loss function, and optimizer
model = Model().to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
epochs = 30000

# Define accuracy function
def accuracy(y_true, y_pred):
    y_pred_rounded = torch.round(y_pred)  # Round the predictions to 0 or 1
    correct = (y_pred_rounded == y_true).sum().item()
    return correct / len(y_true)

# Move data to target device
X_train, y_train = x_train.to(device), y_train.to(device)
X_test, y_test = x_test.to(device), y_test.to(device)

# Training loop
for epoch in range(epochs):
    model.train()
    # 1. Forward pass
    y_logits = model(X_train).squeeze()
    
    # 2. Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy(y_true=y_train, y_pred=y_logits)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model.eval()
    with torch.no_grad():
        # 1. Forward pass
        test_logits = model(X_test).squeeze()
        # 2. Calculate loss/accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy(y_true=y_test, y_pred=test_logits)
    # Print out every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.4f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.4f}%")

# Plot the decision boundary
plot_decision_boundary(model, X.cpu().numpy(), Y.cpu().numpy())
