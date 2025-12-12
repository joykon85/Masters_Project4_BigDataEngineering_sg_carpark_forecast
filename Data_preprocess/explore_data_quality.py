# -*- coding: utf-8 -*-
"""ExploreMissingValuesCP.ipynb
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

import matplotlib.pyplot as plt

df_cp = pd.read_csv("concat/16febto30mar_carpark_availability.csv")

df_cp_missing = pd.read_csv("concat/16febto30mar_missing_carpark_availability.csv")

gaps = df_cp[(df_cp["time_diff_min"] != 5) & (df_cp["time_diff_min"] != 0) ].index
gaps

df_cp.loc[gaps]

df_cp['timestamp'].unique().size

df_cp['timestamp'].value_counts()

df_cp['CarParkID'].unique().size

value_counts= df_cp['CarParkID'].value_counts()

# Define the file path in Google Drive
file_path = '/content/gdrive/My Drive/Proj5006/concat/16febto30mar_carparks_value_count.csv'

# Save DataFrame to CSV
value_counts.to_csv(file_path)

print(f"CSV file saved at: {file_path}")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(value_counts)
