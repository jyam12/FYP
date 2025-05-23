import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib


# Load the dataset
file_path = "datasets/Processed_Dataset_adding_priceMovementLabel.csv"  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Parse relevant columns for all bid and ask levels
levels = 5  # Number of bid/ask levels
bid_columns = [f"bids[{i}].price" for i in range(levels)] + [f"bids[{i}].amount" for i in range(levels)]
ask_columns = [f"asks[{i}].price" for i in range(levels)] + [f"asks[{i}].amount" for i in range(levels)]
feature_columns = bid_columns + ask_columns

# Create target variable (predict next mark_price)
data["target"] = data["mark_price"].shift(-1)  # Predict next mark_price (shift by -1)

# Drop rows with NaN values due to lagging
data.dropna(inplace=True)

# Sequentially split data into train (60%), test (30%), and eval (10%)
train_size = int(len(data) * 0.6)
test_size = int(len(data) * 0.3)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:train_size + test_size]
eval_data = data.iloc[train_size + test_size:].copy()  # Copy to avoid SettingWithCopyWarning

# Features and target
X_train = train_data[feature_columns]
y_train = train_data["target"]
X_test = test_data[feature_columns]
y_test = test_data["target"]
X_eval = eval_data[feature_columns]
y_eval = eval_data["target"]

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train AdaBoost Regressor
adaboost_model = AdaBoostRegressor(n_estimators=50, random_state=42)
adaboost_model.fit(X_train, y_train)

# Save the models
joblib.dump(lr_model, "lr_model.pkl") 
joblib.dump(adaboost_model, "adaboost_model.pkl") 

# # Predict on the evaluation set
# eval_data.loc[:, "predicted_lr"] = lr_model.predict(X_eval)
# eval_data.loc[:, "predicted_adaboost"] = adaboost_model.predict(X_eval)

# # Evaluate performance using MSE on evaluation set
# mse_lr = mean_squared_error(y_eval, eval_data["predicted_lr"])
# mse_adaboost = mean_squared_error(y_eval, eval_data["predicted_adaboost"])

# print(f"Linear Regression MSE (Eval Set): {mse_lr:.2f}")
# print(f"AdaBoost MSE (Eval Set): {mse_adaboost:.2f}")

# # Directional Accuracy
# eval_data.loc[:, "actual_direction"] = np.sign(y_eval.diff().fillna(0))
# eval_data.loc[:, "predicted_direction_lr"] = np.sign(eval_data["predicted_lr"] - y_eval)
# eval_data.loc[:, "predicted_direction_adaboost"] = np.sign(eval_data["predicted_adaboost"] - y_eval)

# # Calculate directional accuracy
# directional_accuracy_lr = (eval_data["actual_direction"] == eval_data["predicted_direction_lr"]).mean()
# directional_accuracy_adaboost = (eval_data["actual_direction"] == eval_data["predicted_direction_adaboost"]).mean()

# print(f"Directional Accuracy (Linear Regression): {directional_accuracy_lr:.2%}")
# print(f"Directional Accuracy (AdaBoost): {directional_accuracy_adaboost:.2%}")

# # Tolerance-Based Accuracy
# tolerance = 0.0001 * y_eval.mean()  # Set tolerance to 0.01% of average price
# eval_data.loc[:, "tolerance_lr"] = abs(eval_data["predicted_lr"] - y_eval) <= tolerance
# eval_data.loc[:, "tolerance_adaboost"] = abs(eval_data["predicted_adaboost"] - y_eval) <= tolerance

# # Calculate tolerance-based accuracy
# tolerance_accuracy_lr = eval_data["tolerance_lr"].mean()
# tolerance_accuracy_adaboost = eval_data["tolerance_adaboost"].mean()

# print(f"Tolerance-Based Accuracy (Linear Regression): {tolerance_accuracy_lr:.2%}")
# print(f"Tolerance-Based Accuracy (AdaBoost): {tolerance_accuracy_adaboost:.2%}")

# # Add symbol columns based on comparison with the last mark price
# symbol_tolerance = 0.0001 * y_eval.mean()  # Adjust as needed
# eval_data.loc[:, "symbol_lr"] = None
# eval_data.loc[:, "symbol_adaboost"] = None

# for i in range(1, len(eval_data)):
#     # Get the last mark price
#     last_mark_price = y_eval.iloc[i - 1]

#     # Linear Regression symbol logic
#     if eval_data["predicted_lr"].iloc[i] > last_mark_price + symbol_tolerance:
#         eval_data.loc[eval_data.index[i], "symbol_lr"] = "up"
#     elif eval_data["predicted_lr"].iloc[i] < last_mark_price - symbol_tolerance:
#         eval_data.loc[eval_data.index[i], "symbol_lr"] = "down"
#     else:
#         eval_data.loc[eval_data.index[i], "symbol_lr"] = "stable"

#     # AdaBoost symbol logic
#     if eval_data["predicted_adaboost"].iloc[i] > last_mark_price + symbol_tolerance:
#         eval_data.loc[eval_data.index[i], "symbol_adaboost"] = "up"
#     elif eval_data["predicted_adaboost"].iloc[i] < last_mark_price - symbol_tolerance:
#         eval_data.loc[eval_data.index[i], "symbol_adaboost"] = "down"
#     else:
#         eval_data.loc[eval_data.index[i], "symbol_adaboost"] = "stable"

# # Save the final evaluation DataFrame to a CSV file
# eval_data.to_csv("eval_data_with_predictions.csv", index=False)
# print("Evaluation DataFrame saved as 'eval_data_with_predictions.csv'")

# # Visualize symbol distributions
# symbol_counts_lr = eval_data["symbol_lr"].value_counts()
# symbol_counts_adaboost = eval_data["symbol_adaboost"].value_counts()

# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# symbol_counts_lr.plot(kind="bar", ax=axes[0], title="Symbol Distribution (Linear Regression)", color=["green", "red", "blue"])
# symbol_counts_adaboost.plot(kind="bar", ax=axes[1], title="Symbol Distribution (AdaBoost)", color=["green", "red", "blue"])
# axes[0].set_xlabel("Symbol")
# axes[0].set_ylabel("Count")
# axes[1].set_xlabel("Symbol")
# axes[1].set_ylabel("Count")
# plt.tight_layout()
# plt.show()

# # Plot actual vs predicted prices
# plt.figure(figsize=(12, 6))
# plt.plot(eval_data.index, y_eval, label="Actual Price", color="blue")
# plt.plot(eval_data.index, eval_data["predicted_lr"], label="Predicted Price (Linear Regression)", color="orange", linestyle="--")
# plt.plot(eval_data.index, eval_data["predicted_adaboost"], label="Predicted Price (AdaBoost)", color="green", linestyle="--")
# plt.title("Actual vs Predicted Prices (Eval Set)")
# plt.xlabel("Index")
# plt.ylabel("Price")
# plt.legend()
# plt.grid()
# plt.show()

# # Plot directional accuracy
# labels = ["Linear Regression", "AdaBoost"]
# values = [directional_accuracy_lr, directional_accuracy_adaboost]
# plt.figure(figsize=(8, 5))
# plt.bar(labels, values, color=["orange", "green"])
# plt.title("Directional Accuracy Comparison")
# plt.ylabel("Accuracy")
# plt.ylim(0, 1)
# plt.grid(axis="y")
# plt.show()

# # Plot tolerance-based accuracy
# labels = ["Linear Regression", "AdaBoost"]
# values = [tolerance_accuracy_lr, tolerance_accuracy_adaboost]
# plt.figure(figsize=(8, 5))
# plt.bar(labels, values, color=["orange", "green"])
# plt.title("Tolerance-Based Accuracy Comparison")
# plt.ylabel("Accuracy")
# plt.ylim(0, 1)
# plt.grid(axis="y")
# plt.show()
