import pandas as pd
import numpy as np



# Specify the path to the CSV file
book_snapshot_5_path = 'datasets/deribit_book_snapshot_5_2024-09-01_BTC-PERPETUAL.csv.gz'
derivative_ticker_path = 'datasets/deribit_derivative_ticker_2024-09-01_BTC-PERPETUAL.csv.gz'

# Read the compressed CSV file using pandas
book_snapshot_5 = pd.read_csv(book_snapshot_5_path, compression='gzip')
derivative_ticker= pd.read_csv(derivative_ticker_path, compression='gzip')

# Define Sampling time interval
Sampling_time_interval="250ms"
# Define the attribute selected of Processed_dataframe
selected_data = {
    'timestamp': pd.date_range(start='2024-09-01 00:00:00', end='2024-09-01 23:59:59.999', freq=Sampling_time_interval).astype('int64') // 10**3,
    "exchange": np.nan,
    "symbol":   np.nan,
    "DT_timestamp": np.nan,
    "BS5_timestamp": np.nan,
    "last_price": np.nan,
    "index_price": np.nan,
    "mark_price": np.nan,
    "asks[0].price": np.nan,
    "asks[0].amount": np.nan,
    "bids[0].price": np.nan,
    "bids[0].amount": np.nan,    
    "asks[1].price": np.nan,
    "asks[1].amount": np.nan,
    "bids[1].price": np.nan,
    "bids[1].amount": np.nan,    
    "asks[2].price": np.nan,
    "asks[2].amount": np.nan,
    "bids[2].price": np.nan,
    "bids[2].amount": np.nan,    
    "asks[3].price": np.nan,
    "asks[3].amount": np.nan,
    "bids[3].price": np.nan,
    "bids[3].amount": np.nan,    
    "asks[4].price": np.nan,
    "asks[4].amount": np.nan,
    "bids[4].price": np.nan,
    "bids[4].amount": np.nan,
    
}
P_data = pd.DataFrame(selected_data)
P_data['exchange'] = P_data['exchange'].astype(str)
P_data['symbol'] = P_data['symbol'].astype(str)
# P_data['DT_timestamp'] = P_data['DT_timestamp'].astype('int64')
# P_data['BS5_timestamp'] = P_data['BS5_timestamp'].astype('int64')
# print("Size of data",len(P_data))
# P_data.head()

# Rename column name in the datasets
book_snapshot_5 = book_snapshot_5.rename(columns={'timestamp': 'BS5_timestamp'})
derivative_ticker = derivative_ticker.rename(columns={'timestamp': 'DT_timestamp'})

# Define the attribute selected
derivative_ticker_selected_attribute=["exchange", "symbol", "DT_timestamp", "last_price", "index_price", "mark_price"]
book_snapshot_5_selected_attribute=["BS5_timestamp","asks[0].price","asks[0].amount", "bids[0].price","bids[0].amount",
                           "asks[1].price","asks[1].amount","bids[1].price","bids[1].amount",
                           "asks[2].price","asks[2].amount","bids[2].price","bids[2].amount",
                           "asks[3].price","asks[3].amount","bids[3].price","bids[3].amount",
                           "asks[4].price","asks[4].amount","bids[4].price","bids[4].amount"]

# Function to find nearest timestamp in dataframe for a given timestamp 
def find_nearest_timestamp(ts, df, attribute,name):
    # return the index of the first occurrence of this minimum value.
    idx = (df[name] - ts).abs().argmin()
    return df.loc[idx, attribute]

# Fill df based on nearest timestamp from P_derivative_ticker
for idx in P_data.index:
# for idx in range(345600):
    P_data.loc[idx,derivative_ticker_selected_attribute] = find_nearest_timestamp(P_data['timestamp'].iloc[idx], derivative_ticker,derivative_ticker_selected_attribute,"DT_timestamp")
    P_data.loc[idx,book_snapshot_5_selected_attribute] = find_nearest_timestamp(P_data['timestamp'].iloc[idx], book_snapshot_5,book_snapshot_5_selected_attribute,"BS5_timestamp")

P_data['DT_timestamp'] = P_data['DT_timestamp'].astype('int64')
P_data['BS5_timestamp'] = P_data['BS5_timestamp'].astype('int64')
P_data.to_csv('./datasets/Processed_Dataset.csv', index=False)
