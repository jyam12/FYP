import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Specify the path to the CSV file
Processed_Dataset_path = './datasets/PreProcessed_Dataset.csv'

# Read the compressed CSV file using pandas
Processed_Dataset = pd.read_csv(Processed_Dataset_path)

# Sequentially split data into train (60%), test (30%), and eval (10%)
#207359
train_size = int(len(Processed_Dataset) * 0.6)
#103679
test_size = int(len(Processed_Dataset) * 0.3)
#34559
eval_size = int(len(Processed_Dataset) * 0.1)

train_data = Processed_Dataset.iloc[:train_size+1].copy(deep=True)
test_data = Processed_Dataset.iloc[train_size:train_size + test_size+1].copy(deep=True)
eval_data = Processed_Dataset.iloc[train_size+test_size+2:].copy(deep=True)

print("Size of data",len(Processed_Dataset))
print("Size of train data",len(train_data))
print("Size of test data",len(test_data))
print("Size of eval data",len(eval_data))

train_data.to_csv('./datasets/Train_Pre_Processed_Dataset.csv', index=False)
test_data.to_csv('./datasets/Test_Pre_Processed_Dataset.csv', index=False)
eval_data.to_csv('./datasets/Eval_Pre_Processed_Dataset.csv', index=False)


