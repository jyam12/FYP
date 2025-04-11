import pandas as pd
import numpy as np

# Specify the path to the CSV file
# Processed_Dataset_path = './datasets/Samplized_Dataset.csv'
Processed_Dataset_path = './datasets/Sim_Samplized_Dataset.csv'


# Read the compressed CSV file using pandas
Processed_Dataset = pd.read_csv(Processed_Dataset_path)


def generate_next_mark_price():
    # Create target variable (predict next mark_price)
    Processed_Dataset["next_1st_mark_price"] = Processed_Dataset["mark_price"].shift(-1)  # Predict next mark_price (shift by -1)
    Processed_Dataset["next_2nd_mark_price"] = Processed_Dataset["mark_price"].shift(-2)  # Predict next mark_price (shift by -1)
    # Drop rows with NaN values due to last data do not have next mark_price
    Processed_Dataset.dropna(inplace=True)

generate_next_mark_price()

# Add a new column 'Price Movement' to determine if the mark price is up, down, or stable comparing to the previous timestamp
def generate_actual_label():
    Processed_Dataset['1st_Price_Movement'] = Processed_Dataset.apply(lambda row: 'Up' if row['next_1st_mark_price'] > row['mark_price'] else 'Down' if row['next_1st_mark_price'] < row['mark_price'] else 'Stable', axis=1)
    Processed_Dataset['2nd_Price_Movement'] = Processed_Dataset.apply(lambda row: 'Up' if row['next_2nd_mark_price'] > row['mark_price'] else 'Down' if row['next_2nd_mark_price'] < row['mark_price'] else 'Stable', axis=1)

generate_actual_label()

def gerenrate_actual_direction():
    Processed_Dataset["1st_actual_direction"] = np.sign(Processed_Dataset["next_1st_mark_price"] - Processed_Dataset["mark_price"])
    Processed_Dataset["2nd_actual_direction"] = np.sign(Processed_Dataset["next_2nd_mark_price"] - Processed_Dataset["mark_price"])


gerenrate_actual_direction()

# Processed_Dataset.to_csv('./datasets/Pre_Processed_Dataset.csv', index=False)
Processed_Dataset.to_csv('./datasets/Sim_Pre_Processed_Dataset.csv', index=False)


