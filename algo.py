import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = "datasets/test.csv"  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Parse and preprocess relevant columns for all levels
levels = 5  # Number of bid/ask levels
bid_columns = [f"bids[{i}].price" for i in range(levels)] + [f"bids[{i}].amount" for i in range(levels)]
ask_columns = [f"asks[{i}].price" for i in range(levels)] + [f"asks[{i}].amount" for i in range(levels)]

# Create cumulative metrics for VWAP and volume imbalance
data["VWAP_bid"] = sum(data[f"bids[{i}].price"] * data[f"bids[{i}].amount"] for i in range(levels)) / sum(
    data[f"bids[{i}].amount"] for i in range(levels))
data["VWAP_ask"] = sum(data[f"asks[{i}].price"] * data[f"asks[{i}].amount"] for i in range(levels)) / sum(
    data[f"asks[{i}].amount"] for i in range(levels))

data["cumulative_bid_volume"] = sum(data[f"bids[{i}].amount"] for i in range(levels))
data["cumulative_ask_volume"] = sum(data[f"asks[{i}].amount"] for i in range(levels))
data["volume_imbalance"] = data["cumulative_bid_volume"] / (
    data["cumulative_bid_volume"] + data["cumulative_ask_volume"]
)

# Calculate spread using VWAPs
data["spread"] = data["VWAP_ask"] - data["VWAP_bid"]

# Split the dataset into training (90%) and testing (10%)
split_index = int(len(data) * 0.9)
training_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()

# Calculate thresholds from training data
avg_spread = training_data["spread"].mean()
imbalance_high = 0.6
imbalance_low = 0.4

# Define prediction rules to calculate the next price and signals
def predict_next_price_and_signal(row, current_price):
    delta_price = 0
    signal = "stable"
    # Rule 1: Narrow spread means stable prices
    if row["spread"] < avg_spread:
        delta_price = 0  # Stable
        signal = "stable"
    # Rule 2: High bid volume imbalance (upward movement)
    elif row["volume_imbalance"] > imbalance_high:
        delta_price = row["spread"] * 0.5  # Fraction of spread for upward movement
        signal = "up"
    # Rule 3: High ask volume imbalance (downward movement)
    elif row["volume_imbalance"] < imbalance_low:
        delta_price = -row["spread"] * 0.5  # Fraction of spread for downward movement
        signal = "down"
    # Rule 4: Add momentum
    if pd.notnull(row["momentum"]):  # Add momentum effect
        delta_price += row["momentum"]
    return current_price + delta_price, signal

# Add momentum to test data
test_data["momentum"] = test_data["mark_price"].diff()

# Initialize predicted price with the first actual mark price and create signal column
test_data["predicted_price"] = None
test_data["signal"] = None
test_data.iloc[0, test_data.columns.get_loc("predicted_price")] = test_data.iloc[0]["mark_price"]

# Calculate predicted prices and signals iteratively
for i in range(1, len(test_data)):
    current_price = test_data.iloc[i - 1]["predicted_price"]
    predicted_price, signal = predict_next_price_and_signal(test_data.iloc[i], current_price)
    test_data.iloc[i, test_data.columns.get_loc("predicted_price")] = predicted_price
    test_data.iloc[i, test_data.columns.get_loc("signal")] = signal

# Evaluate performance (Mean Squared Error) based on mark price
mse = mean_squared_error(test_data["mark_price"], test_data["predicted_price"])
print(f"Mean Squared Error: {mse:.2f}")

# Visualize the actual vs. predicted price along with signals
plt.figure(figsize=(12, 6))
plt.plot(test_data["mark_price"], label="Actual Price (Mark Price)", color="blue")
plt.plot(test_data["predicted_price"], label="Predicted Price", color="orange", linestyle="--")
plt.title("Actual vs Predicted Prices with Signals")
plt.xlabel("Index")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# Add signal distribution visualization
signal_counts = test_data["signal"].value_counts()
signal_counts.plot(kind='bar', title="Signal Distribution", color=["blue", "orange", "green"])
plt.xlabel("Signals")
plt.ylabel("Frequency")
plt.show()
