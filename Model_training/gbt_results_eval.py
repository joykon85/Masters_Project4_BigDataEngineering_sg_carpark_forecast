import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle

import os

# ===============================
# 🔹 Step 1: Load & Preprocess Data
# ===============================
df = pd.read_csv("16febto15mar_carpark_availability_final.csv")  # replace with your actual file name
#df = pd.read_csv("5thApril.csv")

pred_df = pd.read_csv("predictions_GBT_25Apr.csv")

pred_df

# Load MinMaxScaler
with open('4APR_LSTM_Model/min_max_scaler_AvailableLots.pkl', 'rb') as file:
    scaler_AL = pickle.load(file)

pred_df['label_actual'] = scaler_AL.inverse_transform(pred_df[['label']])

pred_df['pred_actual'] = scaler_AL.inverse_transform(pred_df[['prediction']])

from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

MSE_N = mean_squared_error(pred_df['label'], pred_df['prediction'])
#print(MSE_N)
MSE = mean_squared_error(pred_df['label_actual'], pred_df['pred_actual'])
print("MSE Value is: ", MSE)
RMSE_N = root_mean_squared_error(pred_df['label'], pred_df['prediction'])
#print(RMSE_N)
RMSE = root_mean_squared_error(pred_df['label_actual'], pred_df['pred_actual'])
print("RMSE Value is: ",RMSE)

y_test_t = pred_df['label_actual']
preds_t = pred_df['pred_actual']

y_test_t = y_test_t.to_numpy()
preds_t = preds_t.to_numpy()

"""# Plot predictions

"""
num_CP= len(df['CarParkID'].unique())

CP = df['CarParkID'].unique()

y_test_t_reshape = y_test_t.reshape(y_test_t.shape[0],)
y_test_t_reshape = y_test_t_reshape.reshape(num_CP, int(y_test_t.shape[0]/num_CP))
y_test_t_reshape.shape

preds_t_reshape = preds_t.reshape(preds_t.shape[0],)
preds_t_reshape = preds_t_reshape.reshape(num_CP, int(preds_t.shape[0]/num_CP))
preds_t_reshape.shape

def plot_test_vs_pred(y_test, y_pred, title="Test vs. Predicted"):
    """
    Plots y_test and y_pred in the same graph for comparison.

    Args:
        y_test (array-like): Test target values.
        y_pred (array-like): Predicted values.
        title (str): Title of the plot.
    """

    plt.figure(figsize=(12, 6))  # Adjust figure size as needed

    plt.plot(y_test, label='y_test', marker='o')  # Plot y_test with markers
    plt.plot(y_pred, label='y_pred', marker='x')  # Plot y_pred with markers

    plt.title(title)
    plt.xlabel('Data Points (Test Set)')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()

for i in range(num_CP):
  plot_test_vs_pred(pd.Series(y_test_t_reshape[i]).tail(400), pd.Series(preds_t_reshape[i]).tail(400),CP[i])
