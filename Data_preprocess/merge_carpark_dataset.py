# -*- coding: utf-8 -*-
"""mergeCarparkDataset.ipynb

#1. IMPORTS and Mount Drive
"""

import pandas as pd
import numpy as np

import os
from google.colab import drive
drive.mount('/content/gdrive')

# Change working directory to be current folder
# os.chdir('/content/gdrive/My Drive/Your Folder Name/Your sub Folder Name')
os.chdir('/content/gdrive/My Drive/Proj5006')
!ls

"""#2. Read CSVs"""

df_eswar = pd.read_csv("eswar/lta_data.carpark_availability.csv")  # Replace with your actual file path
df_ian = pd.read_csv("ian/lta_data.carpark_availability.csv")  # Replace with your actual file path

"""#2. Filter out for minute grid"""

min_grid = {'min': [0,5,10,15,20,25,30,35,40,45,50,55,]}

df_ian["timestamp"] = pd.to_datetime(df_ian["timestamp"])
df_ian['timestamp'] = df_ian['timestamp'].dt.round('min')
df_ian['min'] = df_ian['timestamp'].dt.minute
df_ian = df_ian[df_ian['min'].isin(min_grid['min'])]
df_ian = df_ian.reset_index(drop=True)

#df_ian["time_diff"] = df_ian['timestamp'].diff()
#df_ian["time_diff_min"] = df_ian["time_diff"].astype('int') // 10**9 // 60
#print(df_ian["time_diff_min"].value_counts())

df_eswar["timestamp"] = pd.to_datetime(df_eswar["timestamp"])
df_eswar['timestamp'] = df_eswar['timestamp'].dt.round('min')
df_eswar['min'] = df_eswar['timestamp'].dt.minute
df_eswar = df_eswar[df_eswar['min'].isin(min_grid['min'])]
df_eswar = df_eswar.reset_index(drop=True)

#df_eswar["time_diff"] = df_eswar['timestamp'].diff()
#df_eswar["time_diff_min"] = df_eswar["time_diff"].astype('int') // 10**9 // 60
#print(df_eswar["time_diff_min"].value_counts())

"""# 2b. Filter out only for cars and create unique id to search for"""

df_ian = df_ian[df_ian['LotType'] == 'C']
df_eswar = df_eswar[df_eswar['LotType'] == 'C']

df_ian['unique_id'] = df_ian['CarParkID'] + '_' + df_ian['timestamp'].astype(str)
df_eswar['unique_id'] = df_eswar['CarParkID'] + '_' + df_eswar['timestamp'].astype(str)

print(df_ian.shape)
unique_values = df_ian['unique_id'].unique()
print(unique_values.shape)

"""#3. Define Data Start to End Time and filter Dataframe too"""

#decide the start and end time
start_time = pd.to_datetime('2025-02-16 00:00:00').tz_localize('UTC')
end_time = pd.to_datetime('2025-03-16 00:00:00').tz_localize('UTC')

df_ian = df_ian[df_ian['timestamp'].between(start_time, end_time)]
df_eswar = df_eswar[df_eswar['timestamp'].between(start_time, end_time)]

all_timestamps = pd.date_range(start=start_time, end=end_time, freq='5min')
all_timestamps

"""# 3b. ALL possible Unique IDs"""

unique_cp = df_ian['CarParkID'].unique()
all_timestamps = all_timestamps.astype(str)
print(unique_cp)
print(all_timestamps)

all_UID = [a + "_" + b for a in unique_cp for b in all_timestamps]

#print(all_UID)

type(all_timestamps)

all_UID = pd.Index(all_UID)
type(all_UID)
all_UID

"""#4. Combine the datasets"""

missing_UID = all_UID[~all_UID.isin(df_ian['unique_id'])]
matching_rows = df_eswar[df_eswar['unique_id'].isin(missing_UID)]
# Step 2: Append the matching rows to df_ian
df_combined = pd.concat([df_ian, matching_rows], ignore_index=True)
# Step 3: Sort the new dataframe by timestamp
df_combined = df_combined.sort_values(by=['timestamp','unique_id'])
# Print the final result
df_combined = df_combined.reset_index(drop=True)
df_combined

df_combined["time_diff"] = df_combined['timestamp'].diff()
df_combined["time_diff_min"] = df_combined["time_diff"].astype('int') // 10**9 // 60
print(df_combined["time_diff_min"].value_counts())

"""# 5. Save to CSV"""

# Define the file path in Google Drive
file_path = '/content/gdrive/My Drive/Proj5006/concat/carpark_availability.csv'

# Save DataFrame to CSV
df_combined.to_csv(file_path, index=False)

print(f"CSV file saved at: {file_path}")

"""# 6. Explore large gaps

#6a. Explore in original DFs
"""

df_ian["time_diff"] = df_ian['timestamp'].diff()
df_ian["time_diff_min"] = df_ian["time_diff"].astype('int') // 10**9 // 60
print(df_ian["time_diff_min"].value_counts())

df_eswar["time_diff"] = df_eswar['timestamp'].diff()
df_eswar["time_diff_min"] = df_eswar["time_diff"].astype('int') // 10**9 // 60
print(df_eswar["time_diff_min"].value_counts())

gaps_i = df_ian[(df_ian["time_diff_min"] != 5) & (df_ian["time_diff_min"] != 0)]
gaps_i

gaps_e = df_eswar[(df_eswar["time_diff_min"] != 5) & (df_eswar["time_diff_min"] != 0)]
gaps_e

"""#6b Explore in combined DFs"""

gaps = df_combined[(df_combined["time_diff_min"] != 5) & (df_combined["time_diff_min"] != 0) ].index
gaps

df_combined.loc[gaps]

gaps = gaps[gaps != 0]  # Remove 0

# Step 2: Add (value - 1) for each remaining value
gaps = gaps.tolist()  # Convert to list if it's an Index object
gaps_extended = set(gaps + [x - 1 for x in gaps])  # Use set to avoid duplicates

# Step 3: Sort the final list
gaps_sorted = sorted(gaps_extended)

# Print the final result
print(gaps_sorted)

df_missing = df_combined.loc[gaps_sorted]
df_missing

"""#6c Save to CSV"""

# Define the file path in Google Drive
file_path = '/content/gdrive/My Drive/Proj5006/concat/missing_carpark_availability.csv'

# Save DataFrame to CSV
df_missing.to_csv(file_path, index=False)

print(f"CSV file saved at: {file_path}")