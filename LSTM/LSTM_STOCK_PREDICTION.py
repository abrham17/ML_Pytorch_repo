import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
ticker = 'ETH-USD'
data = yf.download(ticker, start="2018-01-01", end="2023-01-01")['Close']

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.values.reshape(-1, 1))

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 30
X, y = create_sequences(data, SEQ_LENGTH)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _ , hn = self.lstm(x)
        return self.fc(hn[-1])

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_output = model(X_test)
            test_loss = criterion(test_output, y_test)
            test_losses.append(test_loss.item())

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    return train_losses, test_losses, model


input_size = 1
variants = [
    {"hidden_size": 50, "num_layers": 1},
    {"hidden_size": 100, "num_layers": 1},
    {"hidden_size": 50, "num_layers": 2},
    {"hidden_size": 100, "num_layers": 2}
]

for variant in variants:
    print(f"Training LSTM with hidden_size={variant['hidden_size']} and num_layers={variant['num_layers']}")
    
    model = LSTMModel(input_size=input_size, hidden_size=variant["hidden_size"], num_layers=variant["num_layers"])
    
    train_losses, test_losses, trained_model = train_model(model, X_train, y_train, X_test, y_test)
    
    trained_model.eval()
    with torch.no_grad():
        predictions = trained_model(X_test).numpy()
    mse_loss = nn.MSELoss()
    rmse = torch.sqrt(mse_loss(torch.tensor(predictions), y_test)).item()
    print(f"RMSE for LSTM with hidden_size={variant['hidden_size']} and num_layers={variant['num_layers']}: {rmse:.4f}")
    
    plt.figure(figsize=(10, 4))
    plt.plot(y_test.numpy(), label="True Price")
    plt.plot(predictions, label="Predicted Price")
    plt.title(f"LSTM Prediction with RMSE={rmse:.4f}")
    plt.xlabel("Time")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.show()

