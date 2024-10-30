import torch
from torch import nn

# Import torchvision 

import gradio as gr
from torchvision import transforms

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader , Dataset

# prepare image data from MNIST
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)

# Set a DataLoader
train_loader = DataLoader(train_data , batch_size = 32 , shuffle = True)
test_loader = DataLoader(test_data , batch_size = 32 , shuffle = False)

# set a device agnostice 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# an image classfication model

class Model(nn.Module):
  def __init__(self , input , hidden , output):
    super().__init__()
    self.block1 = nn.Sequential(
        nn.Conv2d(in_channels = input , out_channels = hidden , kernel_size = 3 , stride = 1 , padding = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels = hidden , out_channels = hidden , kernel_size = 3 , stride = 1 , padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2 , stride = 2)
    )
    self.block2 = nn.Sequential(
        nn.Conv2d(in_channels = hidden , out_channels = hidden , kernel_size = 3 , stride = 1 , padding = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels = hidden , out_channels = hidden , kernel_size = 3 , stride = 1 , padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden*7*7, 
                  out_features=output)
        )
  def forward(self, x: torch.Tensor):
        x = self.block1(x)
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
torch.manual_seed(42)
model_2 = Model(input=1, 
    hidden=10, 
    output=10).to(device)

# create optimizer and loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), 
                             lr=0.1)
#plot the loss function of test data and train data
def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(test_losses)), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()
    plt.show()

# Training and testing function combined
def train_and_evaluate(model, train_loader, test_loader, loss_fn, optimizer, epochs, device):
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model_2.train()
        running_train_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            y_pred = model_2(x)
            loss = loss_fn(y_pred, y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()

        # Average train loss for this epoch
        train_losses.append(running_train_loss / len(train_loader))

        # Testing phase
        model_2.eval()
        running_test_loss = 0.0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                y_pred = model_2(x)
                loss = loss_fn(y_pred, y)
                
                running_test_loss += loss.item()
        
        # Average test loss for this epoch
        test_losses.append(running_test_loss / len(test_loader))

        # Print the losses every epoch
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

    # Plot the training and testing losses
    plot_losses(train_losses, test_losses)

# Hyperparameters
epochs = 100
learning_rate = 0.001


# Train and evaluate the model
train_and_evaluate(model_2, train_loader, test_loader, loss_fn, optimizer, epochs, device)

# Define the mapping from FashionMNIST class indices to their names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Define a transformation to apply to input images to match the format of your training data
transform = transforms.Compose([transforms.Resize((28, 28)),  # Ensure the image is 28x28
                                transforms.ToTensor()])

# Define a function that takes an input image, processes it, and returns the predicted label
def predict(image):
    # Transform the input image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Set the model to evaluation mode and turn off gradients
    model_2.eval()
    with torch.no_grad():
        output = model_2(image)
    
    # Get the predicted class
    predicted_class = torch.argmax(output, dim=1).item()
    
    # Map the predicted class index to the actual class name
    predicted_label = class_names[predicted_class]
    return predicted_label

# Create the Gradio interface
interface = gr.Interface(fn=predict, 
                         inputs=gr.Image(image_mode='L', label="Upload an Image"),
                         outputs="text",
                         title="FashionMNIST Clothing Classifier",
                         description="Upload an image of a fashion item (28x28 grayscale) and the model will predict its type.")

# Launch the interface
interface.launch()
