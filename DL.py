import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------
# Data Preparation
# -----------------------

# Load the dataset (adjust file_path as needed)
file_path = "datasets/test.csv"
data = pd.read_csv(file_path)

# Define the number of levels (e.g., 5) and include only bid and ask prices (ignore amounts)
levels = 5
bid_price_cols = [f"bids[{i}].price" for i in range(levels)]
ask_price_cols = [f"asks[{i}].price" for i in range(levels)]
# Our features now include only the price columns
feature_columns = bid_price_cols + ask_price_cols

# Create target: predict next mark_price
data["target"] = data["mark_price"].shift(-1)
data.dropna(inplace=True)

# Compute mid-price as (best bid + best ask) / 2 (using the first level)
data["mid_price"] = (data[bid_price_cols[0]] + data[ask_price_cols[0]]) / 2

# Normalize feature columns and target using StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

data[feature_columns] = scaler_X.fit_transform(data[feature_columns])
data["target"] = scaler_y.fit_transform(data[["target"]])
data["mid_price"] = scaler_y.fit_transform(data[["mid_price"]])

# Split data: 60% train, 30% test, 10% eval (hidden data)
n = len(data)
train_end = int(0.6 * n)
test_end = int(0.9 * n)
train_data = data.iloc[:train_end].reset_index(drop=True)
test_data = data.iloc[train_end:test_end].reset_index(drop=True)
eval_data = data.iloc[test_end:].reset_index(drop=True)

# -----------------------
# PyTorch Dataset
# -----------------------

class HFTDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, mid_price_col):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)
        self.mid = df[mid_price_col].values.astype(np.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Sequence length = 1
        x = self.X[idx].reshape(1, -1)  # shape: (seq_len, input_size)
        y = self.y[idx]
        mid = self.mid[idx]
        return x, y, mid

train_dataset = HFTDataset(train_data, feature_columns, "target", "mid_price")
# We now use the eval (hidden) data for final evaluation and plotting.
eval_dataset = HFTDataset(eval_data, feature_columns, "target", "mid_price")

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Use batch_size=1 for inference time measurement on the hidden eval set
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

# -----------------------
# Standard LSTM Model
# -----------------------

class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(StandardLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer: (seq_len, batch, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (seq_len, batch, input_size)
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))  # out: (seq_len, batch, hidden_size)
        out = out[-1]  # use output from the last time step (here, only one)
        out = self.fc(out)
        return out

# -----------------------
# OPTM-LSTM Model
# -----------------------

class OPTMLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(OPTMLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        
        # LSTM cell parameters (only for prices now)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Uf = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Ui = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.Wc = nn.Linear(input_size, hidden_size)
        self.Uc = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.Wo = nn.Linear(input_size, hidden_size)
        self.Uo = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Hyperparameters for the internal gradient descent (GD) feature importance mechanism
        self.gd_iters = 7
        self.gd_lr = 0.001
        
    def forward(self, x, hidden, c, mid_price):
        # x: (batch, input_size), hidden and c: (batch, hidden_size)
        mid_price = mid_price.view(-1, 1)
        
        # Standard LSTM computations:
        f_t = torch.sigmoid(self.Wf(x) + self.Uf(hidden))
        i_t = torch.sigmoid(self.Wi(x) + self.Ui(hidden))
        c_tilda = torch.tanh(self.Wc(x) + self.Uc(hidden))
        o_t = torch.sigmoid(self.Wo(x) + self.Uo(hidden))
        c_new = f_t * c + i_t * c_tilda
        h_std = o_t * torch.tanh(c_new)
        
        # Concatenate internal outputs: f_t, i_t, c_tilda, o_t, c_new, h_std
        r_t = torch.cat([f_t, i_t, c_tilda, o_t, c_new, h_std], dim=1)
        
        # Initialize theta for the internal GD loop and update it
        batch_size = r_t.size(0)
        theta = torch.zeros(6 * self.hidden_size, 1, device=x.device)
        for _ in range(self.gd_iters):
            y_pred = r_t.mm(theta)
            error = y_pred - mid_price
            grad = 2 * r_t.t().mm(error) / batch_size
            theta = theta - self.gd_lr * grad
        
        # Partition theta into six parts and compute average absolute weight per part.
        theta_parts = torch.chunk(theta, 6, dim=0)
        importance = [torch.mean(torch.abs(part)) for part in theta_parts]
        max_idx = torch.argmax(torch.stack(importance))
        
        # Select the corresponding slice from r_t as the final hidden state.
        start = max_idx.item() * self.hidden_size
        end = start + self.hidden_size
        h_new = r_t[:, start:end]
        return h_new, c_new

class OPTMLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(OPTMLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell = OPTMLSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x, mid_price):
        # x: (seq_len, batch, input_size); we use seq_len = 1.
        batch_size = x.size(1)
        device = x.device
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        seq_len = x.size(0)
        for t in range(seq_len):
            x_t = x[t]
            h, c = self.cell(x_t, h, c, mid_price)
        out = self.fc(h)
        return out

# -----------------------
# Training and Evaluation Functions
# -----------------------

def train_model(model, loader, num_epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x, y, mid in loader:
            x = x.transpose(0, 1).to(device)  # shape: (seq_len, batch, input_size)
            y = y.view(-1, 1).to(device)
            mid = mid.to(device)
            optimizer.zero_grad()
            if isinstance(model, OPTMLSTM):
                outputs = model(x, mid)
            else:
                outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(1)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(loader.dataset):.6f}")
    return model

def evaluate_model_with_preds(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    preds = []
    actuals = []
    infer_times = []
    with torch.no_grad():
        for x, y, mid in loader:
            x = x.transpose(0, 1).to(device)
            start = time.time()
            if isinstance(model, OPTMLSTM):
                output = model(x, mid.to(device))
            else:
                output = model(x)
            end = time.time()
            infer_times.append((end - start) / x.size(1))
            preds.append(output.cpu().numpy())
            actuals.append(y.reshape(-1, 1))
    preds = np.vstack(preds)
    actuals = np.vstack(actuals)
    
    # Flatten for directional accuracy calculation.
    preds_flat = preds.flatten()
    actuals_flat = actuals.flatten()
    
    mse = np.mean((preds_flat - actuals_flat) ** 2)
    actual_diff = np.sign(np.diff(actuals_flat, axis=0, prepend=actuals_flat[0]))
    pred_diff = np.sign(preds_flat - actuals_flat)
    directional_accuracy = np.mean(actual_diff == pred_diff)
    tol = 0.0001 * np.mean(actuals_flat)
    tol_accuracy = np.mean(np.abs(preds_flat - actuals_flat) <= tol)
    avg_infer_time = np.mean(infer_times)
    
    metrics = {
        "MSE": mse,
        "Directional Accuracy": directional_accuracy,
        "Tolerance Accuracy": tol_accuracy,
        "Average Inference Time per Sample (s)": avg_infer_time
    }
    return metrics, preds, actuals

# -----------------------
# Run Experiments
# -----------------------

input_size = len(feature_columns)
hidden_size = 64  # Example hidden size

# Create and train models using the training data
model_lstm = StandardLSTM(input_size, hidden_size)
model_optm = OPTMLSTM(input_size, hidden_size)

print("Training Standard LSTM Model...")
model_lstm = train_model(model_lstm, train_loader, num_epochs=50, lr=0.0001)
print("Evaluating Standard LSTM Model on Hidden Eval Set...")
metrics_lstm, preds_lstm, actuals_lstm = evaluate_model_with_preds(model_lstm, eval_loader)
print("Standard LSTM Metrics:")
for k, v in metrics_lstm.items():
    print(f"{k}: {v}")

print("\nTraining OPTM-LSTM Model...")
model_optm = train_model(model_optm, train_loader, num_epochs=50, lr=0.0001)
print("Evaluating OPTM-LSTM Model on Hidden Eval Set...")
metrics_optm, preds_optm, actuals_optm = evaluate_model_with_preds(model_optm, eval_loader)
print("OPTM-LSTM Metrics:")
for k, v in metrics_optm.items():
    print(f"{k}: {v}")

# Plot Actual vs. Predicted Prices using all samples from the hidden eval set.
plt.figure(figsize=(12, 6))
plt.plot(actuals_lstm.flatten(), label="Actual Price", color="blue")
plt.plot(preds_lstm.flatten(), label="Predicted (Standard LSTM)", color="orange", linestyle="--")
plt.plot(preds_optm.flatten(), label="Predicted (OPTM-LSTM)", color="green", linestyle=":")
plt.title("Actual vs. Predicted Prices (Hidden Eval Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# Bar plot to compare performance metrics between the two models.
metrics_names = ["MSE", "Directional Accuracy", "Tolerance Accuracy", "Average Inference Time per Sample (s)"]
lstm_metrics = [metrics_lstm[m] for m in metrics_names]
optm_metrics = [metrics_optm[m] for m in metrics_names]

x = np.arange(len(metrics_names))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, lstm_metrics, width, label="Standard LSTM")
plt.bar(x + width/2, optm_metrics, width, label="OPTM-LSTM")
plt.xticks(x, metrics_names, rotation=45)
plt.ylabel("Metric Value")
plt.title("Performance Metrics Comparison on Hidden Eval Set")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()
