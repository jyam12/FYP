import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error

# Load the dataset
# file_path = "datasets/Processed_Dataset_adding_priceMovementLabel.csv"  # Replace with the path to your CSV file
# data = pd.read_csv(file_path)

class hueristic_Method:
    # Parse and preprocess relevant columns for all levels
    def __init__(self,X_train,X_eval, levels):
        # Instance variables unique to each instance
        # Create cumulative metrics for VWAP and volume imbalance
        self.levels=levels
        self.X_train = X_train.copy(deep=True)
        self.X_eval=X_eval.copy(deep=True)
        self.imbalance_high = 0.75
        self.imbalance_low = 0.4

        self.bid_columns = [f"bids[{i}].price" for i in range(levels)] + [f"bids[{i}].amount" for i in range(levels)]
        self.ask_columns = [f"asks[{i}].price" for i in range(levels)] + [f"asks[{i}].amount" for i in range(levels)]

        self.X_train["VWAP_bid"] = sum(self.X_train[f"bids[{i}].price"] * self.X_train[f"bids[{i}].amount"] for i in range(self.levels)) / sum(
            self.X_train[f"bids[{i}].amount"] for i in range(self.levels))
        self.X_train["VWAP_ask"] = sum(self.X_train[f"asks[{i}].price"] * self.X_train[f"asks[{i}].amount"] for i in range(self.levels)) / sum(
            self.X_train[f"asks[{i}].amount"] for i in range(self.levels))
        # Calculate spread using VWAPs
        self.X_train["spread"] = self.X_train["VWAP_ask"] - self.X_train["VWAP_bid"]
        # Calculate thresholds from training data
        self.avg_spread = self.X_train["spread"].mean()

        # self.X_eval["VWAP_bid"] = sum(self.X_eval[f"bids[{i}].price"] * self.X_eval[f"bids[{i}].amount"] for i in range(self.levels)) / sum(
        #     self.X_eval[f"bids[{i}].amount"] for i in range(self.levels))
        # self.X_eval["VWAP_ask"] = sum(self.X_eval[f"asks[{i}].price"] * self.X_eval[f"asks[{i}].amount"] for i in range(self.levels)) / sum(
        #     self.X_eval[f"asks[{i}].amount"] for i in range(self.levels))
        # # Calculate spread using VWAPs
        # self.X_eval["spread"] = self.X_eval["VWAP_ask"] - self.X_eval["VWAP_bid"]


    # Define prediction rules to calculate the next price and signals
    def predict_next_price_and_signal(self,row,current_price):
        delta_price = 0
        signal = "stable"
        # Rule 1: Narrow spread means stable prices
        if row["spread"] < self.avg_spread:
            delta_price = 0  # Stable
            signal = 0.0
        # Rule 2: High bid volume imbalance (upward movement)
        elif row["volume_imbalance"] > self.imbalance_high:
            delta_price = row["spread"] * 0.2  # Fraction of spread for upward movement
            signal = 1.0
        # Rule 3: High ask volume imbalance (downward movement)
        elif row["volume_imbalance"] < self.imbalance_low:
            delta_price = -row["spread"] * 0.2  # Fraction of spread for downward movement
            signal = -1.0
        # Rule 4: Add momentum
        if pd.notnull(row["momentum"]):  # Add momentum effect
            delta_price += row["momentum"]
        return current_price + delta_price, signal
    
    # # Initialize predicted price with the first actual mark price and create signal column
    # test_data["predicted_price"] = None
    # test_data["predict_signal"] = None
    # test_data.iloc[0, test_data.columns.get_loc("predicted_price")] = test_data.iloc[0]["mark_price"]

    # Calculate predicted prices and signals iteratively
    def predict(self):

        self.X_eval["VWAP_bid"] = sum(self.X_eval[f"bids[{i}].price"] * self.X_eval[f"bids[{i}].amount"] for i in range(self.levels)) / sum(
            self.X_eval[f"bids[{i}].amount"] for i in range(self.levels))
        self.X_eval["VWAP_ask"] = sum(self.X_eval[f"asks[{i}].price"] * self.X_eval[f"asks[{i}].amount"] for i in range(self.levels)) / sum(
            self.X_eval[f"asks[{i}].amount"] for i in range(self.levels))
        # Calculate spread using VWAPs
        self.X_eval["spread"] = self.X_eval["VWAP_ask"] - self.X_eval["VWAP_bid"]

        self.X_eval["cumulative_bid_volume"] = sum(self.X_eval[f"bids[{i}].amount"] for i in range(self.levels))
        self.X_eval["cumulative_ask_volume"] = sum(self.X_eval[f"asks[{i}].amount"] for i in range(self.levels))
        self.X_eval["volume_imbalance"] = self.X_eval["cumulative_bid_volume"] / (
            self.X_eval["cumulative_bid_volume"] + self.X_eval["cumulative_ask_volume"])

        # Add momentum to eval data
        self.X_eval["momentum"] = self.X_eval["mark_price"].diff()
        self.X_eval["predicted_heuristic"] = None
        self.X_eval["predicted_direction_heuristic"] = None

        current_price= self.X_eval.iloc[0]["mark_price"]
        predicted_price, signal = self.predict_next_price_and_signal(self.X_eval.iloc[0], current_price)
        self.X_eval.iloc[0, self.X_eval.columns.get_loc("predicted_heuristic")] = predicted_price
        self.X_eval.iloc[0, self.X_eval.columns.get_loc("predicted_direction_heuristic")] = signal
        for i in range(1,len(self.X_eval)):
            current_price = self.X_eval.iloc[i - 1]["mark_price"]
            predicted_price, signal = self.predict_next_price_and_signal(self.X_eval.iloc[i], current_price)
            self.X_eval.iloc[i, self.X_eval.columns.get_loc("predicted_heuristic")] = predicted_price
            self.X_eval.iloc[i, self.X_eval.columns.get_loc("predicted_direction_heuristic")] = signal

        return self.X_eval[["predicted_heuristic","predicted_direction_heuristic"]]


# # Parse and preprocess relevant columns for all levels
# levels = 5  # Number of bid/ask levels
# bid_columns = [f"bids[{i}].price" for i in range(levels)] + [f"bids[{i}].amount" for i in range(levels)]
# ask_columns = [f"asks[{i}].price" for i in range(levels)] + [f"asks[{i}].amount" for i in range(levels)]

# # Create cumulative metrics for VWAP and volume imbalance
# data["VWAP_bid"] = sum(data[f"bids[{i}].price"] * data[f"bids[{i}].amount"] for i in range(levels)) / sum(
#     data[f"bids[{i}].amount"] for i in range(levels))
# data["VWAP_ask"] = sum(data[f"asks[{i}].price"] * data[f"asks[{i}].amount"] for i in range(levels)) / sum(
#     data[f"asks[{i}].amount"] for i in range(levels))

# data["cumulative_bid_volume"] = sum(data[f"bids[{i}].amount"] for i in range(levels))
# data["cumulative_ask_volume"] = sum(data[f"asks[{i}].amount"] for i in range(levels))
# data["volume_imbalance"] = data["cumulative_bid_volume"] / (
#     data["cumulative_bid_volume"] + data["cumulative_ask_volume"]
# )

# # Calculate spread using VWAPs
# data["spread"] = data["VWAP_ask"] - data["VWAP_bid"]

# # Split the dataset into training (90%) and testing (10%)
# split_index = int(len(data) * 0.9)
# training_data = data.iloc[:split_index].copy()
# test_data = data.iloc[split_index:].copy()

# # Calculate thresholds from training data
# avg_spread = training_data["spread"].mean()
# imbalance_high = 0.75
# imbalance_low = 0.4

# # Define prediction rules to calculate the next price and signals
# def predict_next_price_and_signal(row, current_price):
#     delta_price = 0
#     signal = "stable"
#     # Rule 1: Narrow spread means stable prices
#     if row["spread"] < avg_spread:
#         delta_price = 0  # Stable
#         signal = "stable"
#     # Rule 2: High bid volume imbalance (upward movement)
#     elif row["volume_imbalance"] > imbalance_high:
#         delta_price = row["spread"] * 0.2  # Fraction of spread for upward movement
#         signal = "up"
#     # Rule 3: High ask volume imbalance (downward movement)
#     elif row["volume_imbalance"] < imbalance_low:
#         delta_price = -row["spread"] * 0.2  # Fraction of spread for downward movement
#         signal = "down"
#     # Rule 4: Add momentum
#     if pd.notnull(row["momentum"]):  # Add momentum effect
#         delta_price += row["momentum"]
#     return current_price + delta_price, signal

# # Add momentum to test data
# test_data["momentum"] = test_data["mark_price"].diff()

# # Initialize predicted price with the first actual mark price and create signal column
# test_data["predicted_price"] = None
# test_data["predict_signal"] = None
# test_data.iloc[0, test_data.columns.get_loc("predicted_price")] = test_data.iloc[0]["mark_price"]

# # Calculate predicted prices and signals iteratively
# def Heuristic_Predict():
#     for i in range(1, len(test_data)):
#         current_price = test_data.iloc[i - 1]["predicted_price"]
#         predicted_price, signal = predict_next_price_and_signal(test_data.iloc[i], current_price)
#         test_data.iloc[i, test_data.columns.get_loc("predicted_price")] = predicted_price
#         test_data.iloc[i, test_data.columns.get_loc("predict_signal")] = signal
#     return test_data

# # Evaluate performance (Mean Squared Error) based on mark price
# mse = mean_squared_error(test_data["mark_price"], test_data["predicted_price"])
# print(f"Mean Squared Error: {mse:.2f}")

# # Visualize the actual vs. predicted price along with signals
# plt.figure(figsize=(12, 6))
# plt.plot(test_data["mark_price"], label="Actual Price (Mark Price)", color="blue")
# plt.plot(test_data["predicted_price"], label="Predicted Price", color="orange", linestyle="--")
# plt.title("Actual vs Predicted Prices with Signals")
# plt.xlabel("Index")
# plt.ylabel("Price")
# plt.legend()
# plt.grid()
# plt.show()

# # Add signal distribution visualization
# signal_counts = test_data["signal"].value_counts()
# signal_counts.plot(kind='bar', title="Signal Distribution", color=["blue", "orange", "green"])
# plt.xlabel("Signals")
# plt.ylabel("Frequency")
# plt.show()
