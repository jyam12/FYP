import pandas as pd
import numpy as np

# Specify the path to the CSV file
Processed_Dataset_path = 'datasets/Processed_Dataset.csv'


# Read the compressed CSV file using pandas
Processed_Dataset = pd.read_csv(Processed_Dataset_path)


# Add a new column 'Price Movement' to determine if the mark price is up, down, or stable comparing to the previous timestamp
def generate_actual_label():
    Processed_Dataset['Price Movement'] = Processed_Dataset["mark_price"].diff().apply(lambda x: 'Up' if x > 0 else 'Down' if x < 0 else 'Stable')

generate_actual_label()

Processed_Dataset.to_csv('./datasets/Processed_Dataset_adding_priceMovementLabel.csv', index=False)