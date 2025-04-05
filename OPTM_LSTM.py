import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------
# Data Preparation with Normalization
# -----------------------

file_path = "datasets/test.csv"
data = pd.read_csv(file_path)

levels = 5
bid_price_cols = [f"bids[{i}].price" for i in range(levels)]
ask_price_cols = [f"asks[{i}].price" for i in range(levels)]
feature_columns = bid_price_cols + ask_price_cols

data["target"] = data["mark_price"].shift(-1)
data.dropna(inplace=True)
data["mid_price"] = (data[bid_price_cols[0]] + data[ask_price_cols[0]]) / 2

scaler_X = StandardScaler()
scaler_y = StandardScaler()
data[feature_columns] = scaler_X.fit_transform(data[feature_columns])
data["target"] = scaler_y.fit_transform(data[["target"]])
data["mid_price"] = scaler_y.fit_transform(data[["mid_price"]])

n = len(data)
train_end = int(0.6 * n)
eval_end = int(0.9 * n)
train_data = data.iloc[:train_end].reset_index(drop=True)
eval_data = data.iloc[eval_end:].reset_index(drop=True)

# -----------------------
# Dataset Definition (Combined Input)
# -----------------------

# Combine LOB features with guarantor (mid_price) as the last element
class HFTDatasetCombined(Dataset):
    def __init__(self, df, feature_cols, guarantor_col, target_col):
        self.X_features = df[feature_cols].values.astype(np.float32)
        self.guarantor = df[guarantor_col].values.astype(np.float32).reshape(-1, 1)
        self.X = np.concatenate([self.X_features, self.guarantor], axis=1)
        self.y = df[target_col].values.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]  # shape: (num_features + 1,)
        y = self.y[idx]
        return x, y

train_dataset = HFTDatasetCombined(train_data, feature_columns, "mid_price", "target")
eval_dataset = HFTDatasetCombined(eval_data, feature_columns, "mid_price", "target")
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# -----------------------
# OPTM-LSTM Cell (PyTorch Implementation)
# -----------------------

class OPTMLSTMCellTorch(nn.Module):
    def __init__(self, input_dim, hidden_size, n_epoch=13, gd_lr=0.0001):
        super(OPTMLSTMCellTorch, self).__init__()
        self.hidden_size = hidden_size
        self.n_epoch = n_epoch
        self.gd_lr = gd_lr
        self.linear = nn.Linear(input_dim, 4 * hidden_size, bias=True)
        self.recurrent_linear = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh

    def forward(self, x, h, c):
        # x: (batch, input_dim + 1) where last element is guarantor
        x_features = x[:, :-1]  # (batch, input_dim)
        guarantor = x[:, -1]    # (batch,)
        
        z = self.linear(x_features) + self.recurrent_linear(h)
        z_i, z_f, z_c, z_o = z.chunk(4, dim=1)
        i = self.sigmoid(z_i)
        f = self.sigmoid(z_f)
        c_t = self.tanh(z_c)
        o = self.sigmoid(z_o)
        c_new = f * c + i * c_t
        h_temp = o * self.tanh(c_new)
        
        gated_vector = torch.cat([i, f, c_t, c_new, o, h_temp], dim=1)  # (batch, 6*hidden_size)
        batch_size = x.shape[0]
        guarantor = guarantor.view(batch_size, 1)
        
        theta = torch.ones(6 * self.hidden_size, 1, device=x.device)
        for _ in range(self.n_epoch):
            y_pred = gated_vector @ theta  # (batch, 1)
            error = y_pred - guarantor       # (batch, 1)
            grad = (2 / batch_size) * (gated_vector.t() @ error)
            theta = theta - self.gd_lr * grad

        theta_parts = torch.chunk(theta, 6, dim=0)
        importance = [torch.mean(torch.abs(part)) for part in theta_parts]
        importance_stack = torch.stack(importance)  # (6,)
        max_idx = torch.argmax(importance_stack)
        
        if max_idx.item() == 0:
            new_h = i
        elif max_idx.item() == 1:
            new_h = f
        elif max_idx.item() == 2:
            new_h = c_t
        elif max_idx.item() == 3:
            new_h = c_new
        elif max_idx.item() == 4:
            new_h = o
        else:
            new_h = h_temp
        
        return new_h, c_new

# -----------------------
# OPTM-LSTM Model
# -----------------------

class OPTMLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(OPTMLSTM, self).__init__()
        self.hidden_size = hidden_size
        # Exclude guarantor from cell's input
        self.cell = OPTMLSTMCellTorch(input_size - 1, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x: (seq_len, batch, input_size)
        seq_len, batch, _ = x.shape
        device = x.device
        h = torch.zeros(batch, self.hidden_size, device=device)
        c = torch.zeros(batch, self.hidden_size, device=device)
        for t in range(seq_len):
            x_t = x[t]  # (batch, input_size)
            h, c = self.cell(x_t, h, c)
        out = self.fc(h)
        return out

# -----------------------
# Standard LSTM Model (for comparison)
# -----------------------

class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(StandardLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[-1]
        out = self.fc(out)
        return out

# -----------------------
# Training and Evaluation Functions with tqdm and Info Matrix
# -----------------------

def evaluate_model(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.unsqueeze(0).to(device)
            y = y.view(-1, 1).to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item() * x.size(1)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    info_matrix = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        for x, y in train_bar:
            x = x.unsqueeze(0).to(device)  # shape: (seq_len=1, batch, input_size)
            y = y.view(-1, 1).to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(1)
            train_bar.set_postfix(loss=f"{loss.item():.6f}")
        
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        val_loss = evaluate_model(model, val_loader)
        info = {'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': val_loss}
        info_matrix.append(info)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
    return model, info_matrix

# -----------------------
# Run Experiments
# -----------------------

input_size_combined = len(feature_columns) + 1  # features plus guarantor
hidden_size = 64

# Train Standard LSTM Model
model_lstm = StandardLSTM(input_size_combined, hidden_size)
print("Training Standard LSTM Model...")
model_lstm, info_matrix_lstm = train_model(model_lstm, train_loader, eval_loader, num_epochs=5, lr=0.0001)
val_loss_lstm = evaluate_model(model_lstm, eval_loader)
print("Standard LSTM Final Val Loss:", val_loss_lstm)

# Train OPTM-LSTM Model
model_optm = OPTMLSTM(input_size_combined, hidden_size)
print("Training OPTM-LSTM Model...")
model_optm, info_matrix_optm = train_model(model_optm, train_loader, eval_loader, num_epochs=5, lr=0.0001)
val_loss_optm = evaluate_model(model_optm, eval_loader)
print("OPTM-LSTM Final Val Loss:", val_loss_optm)

# Plot info matrix for Standard LSTM training
epochs = [info['epoch'] for info in info_matrix_lstm]
train_losses = [info['train_loss'] for info in info_matrix_lstm]
val_losses = [info['val_loss'] for info in info_matrix_lstm]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label="Train Loss (Standard LSTM)")
plt.plot(epochs, val_losses, label="Val Loss (Standard LSTM)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Standard LSTM Loss over Epochs")
plt.legend()
plt.grid()
plt.show()

# (You can similarly plot the info matrix for OPTM-LSTM if desired)
