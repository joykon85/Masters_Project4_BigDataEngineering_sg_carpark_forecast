# -*- coding: utf-8 -*-
"""impute_carpark_missing.ipynb

# 1. Imports and load data
"""

import pandas as pd
import numpy as np

df_cp = pd.read_csv("concat/16febto15mar_carpark_availability.csv")
df_cp_missing = pd.read_csv("concat/16febto30mar_missing_carpark_availability.csv")

"""# 2. Find the missing and generate the missing date times"""

df_cp_missing = df_cp_missing[df_cp_missing['time_diff_min']!= 0]

df_cp_missing['missing_times'] = (df_cp_missing['time_diff_min'] - 5) / 5

df_cp_missing

def generate_and_sort_previous_timestamps(timestamp_str, x):
    """
    Generates a list of x timestamps at 5-minute intervals before the given timestamp,
    and returns them in chronological (ascending) order.

    Args:
        timestamp_str: A string representing the timestamp (e.g., "2025-02-17 00:30:00+00:00").
        x: The number of timestamps to generate.

    Returns:
        A sorted list of timestamp strings, or an error message if the input is invalid.
    """
    try:
        timestamp = pd.to_datetime(timestamp_str)
        previous_timestamps = []
        for i in range(1, x + 1):
            previous_timestamp = timestamp - pd.Timedelta(minutes=5 * i)
            previous_timestamps.append(str(previous_timestamp))

        # Sort the timestamps chronologically (ascending order)
        sorted_timestamps = sorted(previous_timestamps)
        return sorted_timestamps

    except ValueError:
        return "Invalid timestamp format"
    except TypeError:
        return "Invalid number of timestamps requested"

results = []

df_cp_missing['missing_times'] = df_cp_missing['missing_times'].astype(int)

for index, row in df_cp_missing.iterrows():
    timestamp_str = row['timestamp']
    x = row['missing_times']
    result = generate_and_sort_previous_timestamps(timestamp_str, x)
    results.append(result)

results

"""# 3. Filter out URA and create CARPARK DF Dictionary"""

df_cp_URA = df_cp[df_cp['Agency'] == 'URA']

df_cp_URA.shape

# List of columns to drop
columns_to_drop = ['_id', 'Area', 'Development', 'Location', 'LotType', 'Agency' ]

# Drop the columns
df_cp_URA = df_cp_URA.drop(columns=columns_to_drop)

df_cp_URA

df_cp_URA.reset_index(drop=True, inplace=True)

df_cp_URA.isna().sum()

dfs = {carpark: df_cp_URA[df_cp_URA["CarParkID"] == carpark] for carpark in df_cp_URA["CarParkID"].unique()}

cp_list = df_cp_URA["CarParkID"].unique()

def append_missing_timestamps_to_dataframes_all(dataframes_dict, missing_timestamps_list):
    """
    Appends the same list of lists of missing timestamps to every DataFrame in the dictionary.

    Args:
        dataframes_dict: A dictionary where keys are IDs and values are pandas DataFrames.
        missing_timestamps_list: A list of lists, where each inner list contains missing timestamps (strings).

    Returns:
        A dictionary of DataFrames with appended missing timestamps.
    """

    modified_dataframes_dict = dataframes_dict.copy()

    for id_val, df in modified_dataframes_dict.items():
        df_copy = df.copy()  # Create a copy to avoid modifying the original DataFrame
        # Convert 'timestamp' column to datetime objects before appending new timestamps
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])

        for missing_times in missing_timestamps_list:
            if missing_times: #Check if the list is empty
                missing_df = pd.DataFrame({
                    'CarParkID': df_copy['CarParkID'].iloc[0],
                    'timestamp': pd.to_datetime(missing_times), # Convert to Timestamp here
                    'AvailableLots': pd.NA
                })

                df_copy = pd.concat([df_copy, missing_df], ignore_index=True)

        modified_dataframes_dict[id_val] = df_copy.sort_values(by='timestamp').reset_index(drop=True)

    return modified_dataframes_dict

def available_lots_interpolate(dataframes_dict):
    """
    Converts the 'AvailableLots' column to numeric for each DataFrame in the dictionary.

    Args:
        dataframes_dict: A dictionary where keys are IDs and values are pandas DataFrames.

    Returns:
        A dictionary of DataFrames with 'AvailableLots' converted to numeric.
    """
    modified_dataframes_dict = dataframes_dict.copy()

    for id_val, df in modified_dataframes_dict.items():
        if 'AvailableLots' in df.columns:  # Check if the column exists
            df['AvailableLots'] = pd.to_numeric(df['AvailableLots'], errors='coerce')
            df['AvailableLots'] = df['AvailableLots'].interpolate(method='linear')
            modified_dataframes_dict[id_val] = df

    return modified_dataframes_dict

"""# START HERE?"""

dfs_ts = append_missing_timestamps_to_dataframes_all(dfs, results)
dfs_inter = available_lots_interpolate(dfs_ts)

dfs_inter['A0007']

dfs_inter['A0007'].isna().sum()

def flatten_dataframe_dict(dataframes_dict):
    """
    Flattens a dictionary of DataFrames into a single DataFrame.

    Args:
        dataframes_dict: A dictionary where keys are IDs and values are pandas DataFrames.

    Returns:
        A single pandas DataFrame containing all the data from the dictionary.
    """
    all_dataframes = []
    for id_val, df in dataframes_dict.items():
        df_copy = df.copy() #make copy to avoid modifying original.
        all_dataframes.append(df_copy)

    if all_dataframes: #Check if the list is empty
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame() #return empty dataframe if the dictionary is empty

df_done = flatten_dataframe_dict(dfs_inter)

"""# Ensure DF _DONE is done

"""

# Step 3: Sort the new dataframe by timestamp
df_done = df_done.sort_values(by=['timestamp','unique_id'])
# Print the final result
df_done = df_done.reset_index(drop=True)

df_done["time_diff"] = df_done['timestamp'].diff()
df_done["time_diff_min"] = df_done["time_diff"].astype('int') // 10**9 // 60
print(df_done["time_diff_min"].value_counts())

df_done.isna().sum()

"""# SAVE TO CSV

"""

df_done

# List of columns to drop
columns_to_drop = ['min', 'time_diff', 'time_diff_min']

# Drop the columns
df_done_drop = df_done.drop(columns=columns_to_drop)

df_done_drop

df_done_drop['CarParkID'].value_counts()

# Define the file path in Google Drive
file_path = '/content/gdrive/My Drive/Proj5006/concat/16febto30mar_carpark_availability_imputed.csv'

# Save DataFrame to CSV
df_done_drop.to_csv(file_path, index=False)

print(f"CSV file saved at: {file_path}")

