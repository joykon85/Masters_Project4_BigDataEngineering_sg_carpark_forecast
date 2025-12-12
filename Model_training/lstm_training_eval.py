
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Concatenate, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

import tensorflow as tf
print(tf.__version__)

import matplotlib.pyplot as plt
import pickle

import os

# ===============================
# 🔹 Step 1: Load & Preprocess Data
# ===============================
df = pd.read_csv("16febto30mar_carpark_availability_final.csv")  # replace with your actual file name

# Convert to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract features
df['Min'] = df['timestamp'].dt.minute
df['Hour'] = df['timestamp'].dt.hour
df['DayOfWeek'] = df['timestamp'].dt.dayofweek


# Encode CarParkID
le = LabelEncoder()
df['CarParkEncoded'] = le.fit_transform(df['CarParkID'])

# Sort for time sequence modeling
df.sort_values(['CarParkEncoded', 'timestamp'], inplace=True)


# ===============================
# 🔹 Step 2: Normalize Features
# ===============================
scaler_AL = MinMaxScaler()
scalar_CP = MinMaxScaler()
scaler_temp = MinMaxScaler()
#df_f[['Hour', 'DayOfWeek', 'AvailableLots']] = scaler.fit_transform(df_f[['Hour', 'DayOfWeek', 'AvailableLots']])
df['AvailableLots_MM'] = scaler_AL.fit_transform(df[['AvailableLots']])
df['CarParkEncoded_MM'] = scalar_CP.fit_transform(df[['CarParkEncoded']])
df[['Min_MM', 'Hour_MM', 'DayOfWeek_MM']] = scaler_temp.fit_transform(df[['Min','Hour', 'DayOfWeek']])


# Save LabelEncoder
with open('label_encoder_Carparks.pkl', 'wb') as file:
    pickle.dump(le, file)

# Save MinMaxScaler
with open('min_max_scaler_AvailableLots.pkl', 'wb') as file:
    pickle.dump(scaler_AL, file)

with open('min_max_scaler_Carparks.pkl', 'wb') as file:
    pickle.dump(scalar_CP, file)

with open('min_max_scaler_Min_Hour_Day.pkl', 'wb') as file:
    pickle.dump(scaler_temp, file)

print("LabelEncoder and MinMaxScaler saved successfully!")


def check_value_counts_equal(series):
    """
    Checks if all values in a Pandas Series' value_counts() are equal.

    Args:
        series: A Pandas Series.

    Returns:
        True if all value counts are equal, False otherwise.
    """
    value_counts = series.value_counts()
    if len(value_counts) <= 1: #if there is only one unique value, or zero, then they are equal.
        return True
    else:
        return (value_counts == value_counts.iloc[0]).all() #compare all to the first value.

check_value_counts_equal(df['CarParkID'])

df['CarParkID'].value_counts().iloc[0]

"""# Put into dictionaries so that train test set will contain every carpark"""

dfs = {carpark: df[df["CarParkID"] == carpark] for carpark in df["CarParkID"].unique()}

'''
# ===============================
# 🔹 Step 3: Create Sequences for LSTM
# ===============================

def create_sequences_per_location(data_dict, sequence_length=5, test_ratio=0.2):
    X_train, y_train = [], []
    X_test, y_test = [], []

    for location, df in data_dict.items():
        # Assuming df is sorted by time already
        values = df[['CarParkEncoded_MM', 'Min_MM', 'Hour_MM', 'DayOfWeek_MM', 'AvailableLots_MM']].values

        X_loc, y_loc = [], []

        for i in range(len(values) - sequence_length):
            seq = values[i:i + sequence_length]
            label = values[i + sequence_length][4]  # AvailableLots
            X_loc.append(seq)
            y_loc.append(label)

        # Split within this location
        split_idx = int(len(X_loc) * (1 - test_ratio))
        X_train += X_loc[:split_idx]
        y_train += y_loc[:split_idx]
        X_test += X_loc[split_idx:]
        y_test += y_loc[split_idx:]

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
'''

# ===============================
# 🔹 Step 3: Create Sequences for LSTM
# ===============================

"""# 30 mins version"""
def create_sequences_per_location_30minlag(data_dict, sequence_length=5, test_ratio=0.2):
    X_train, y_train = [], []
    X_test, y_test = [], []

    for location, df in data_dict.items():
        # Assuming df is sorted by time already
        values = df[['CarParkEncoded_MM', 'Min_MM', 'Hour_MM', 'DayOfWeek_MM', 'AvailableLots_MM']].values

        X_loc, y_loc = [], []

        for i in range(len(values) - sequence_length - 5): # need to compenstate here with minus 5
            seq = values[i:i + sequence_length]
            label = values[i + sequence_length + 5][4]  # AvailableLots (plus 5 is 30mins vs + 0 is 5mins)
            X_loc.append(seq)
            y_loc.append(label)

        # Split within this location
        split_idx = int(len(X_loc) * (1 - test_ratio))
        X_train += X_loc[:split_idx]
        y_train += y_loc[:split_idx]
        X_test += X_loc[split_idx:]
        y_test += y_loc[split_idx:]

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def create_sequences_per_location_45minlag(data_dict, sequence_length=5, test_ratio=0.2):
    X_train, y_train = [], []
    X_test, y_test = [], []

    for location, df in data_dict.items():
        # Assuming df is sorted by time already
        values = df[['CarParkEncoded_MM', 'Min_MM', 'Hour_MM', 'DayOfWeek_MM', 'AvailableLots_MM']].values

        X_loc, y_loc = [], []

        for i in range(len(values) - sequence_length - 8): # need to compenstate here with minus 5
            seq = values[i:i + sequence_length]
            label = values[i + sequence_length + 8][4]  # AvailableLots (plus 5 is 30mins vs + 0 is 5mins)
            X_loc.append(seq)
            y_loc.append(label)

        # Split within this location
        split_idx = int(len(X_loc) * (1 - test_ratio))
        X_train += X_loc[:split_idx]
        y_train += y_loc[:split_idx]
        X_test += X_loc[split_idx:]
        y_test += y_loc[split_idx:]

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def create_sequences_per_location_60minlag(data_dict, sequence_length=5, test_ratio=0.2):
    X_train, y_train = [], []
    X_test, y_test = [], []

    for location, df in data_dict.items():
        # Assuming df is sorted by time already
        values = df[['CarParkEncoded_MM', 'Min_MM', 'Hour_MM', 'DayOfWeek_MM', 'AvailableLots_MM']].values

        X_loc, y_loc = [], []

        for i in range(len(values) - sequence_length - 11): # need to compenstate here with minus 5
            seq = values[i:i + sequence_length]
            label = values[i + sequence_length + 11[4]  # AvailableLots (plus 5 is 30mins vs + 0 is 5mins)
            X_loc.append(seq)
            y_loc.append(label)

        # Split within this location
        split_idx = int(len(X_loc) * (1 - test_ratio))
        X_train += X_loc[:split_idx]
        y_train += y_loc[:split_idx]
        X_test += X_loc[split_idx:]
        y_test += y_loc[split_idx:]

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


X_train, X_test, y_train, y_test = create_sequences_per_location_30minlag(dfs, sequence_length=5, test_ratio=0.2)

print(X_train.shape)
print (y_train.shape)
print(X_test.shape)
print (y_test.shape)


# ===============================
# 🔹 Step 4: Build & Train LSTM Model
# ===============================
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer='adam', loss='mse')
model.summary()

hist = model.fit(X_train, y_train, epochs=1000, batch_size=5096, validation_data=(X_test, y_test))

def plot_loss(history, start_epoch=0, end_epoch=None):
    """Plots the training and validation loss from a Keras history object for a specified epoch range."""

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    if end_epoch is None:
        end_epoch = len(loss)  # Plot all epochs if end_epoch is not specified

    epochs = range(start_epoch, end_epoch)

    plt.plot(epochs, loss[start_epoch:end_epoch], label='loss')
    plt.plot(epochs, val_loss[start_epoch:end_epoch], label='val_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(hist, 100)

preds = model.predict(X_test)

len(y_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

y_test = y_test.reshape(len(y_test), 1)

preds_t = scaler_AL.inverse_transform(preds)
y_test_t = scaler_AL.inverse_transform(y_test)

MSE_N = mean_squared_error(y_test, preds)
#print(MSE_N)
MSE = mean_squared_error(y_test_t, preds_t)
print("MSE value is: ", MSE)
RMSE_N = root_mean_squared_error(y_test, preds)
#print(RMSE_N)
RMSE = root_mean_squared_error(y_test_t, preds_t)
print("RMSE value is: ", RMSE)

"""# BASELINE MODEL"""


# Extract the last element of the inner (5, 5) arrays
y_benchmark = X_test[:, -1, -1]

# Reshape the result to (143379, 1)
y_benchmark = y_benchmark.reshape(-1, 1)

y_benchmark_t = scaler_AL.inverse_transform(y_benchmark)

MSE = mean_squared_error(y_test_t, y_benchmark_t)
print(MSE)

RMSE = root_mean_squared_error(y_test_t, y_benchmark_t)
print(RMSE)


"""# Plot predictions

"""

num_CP= len(df['CarParkEncoded_MM'].unique())

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

CP = df['CarParkID'].unique()

for i in range(num_CP):
  plot_test_vs_pred(pd.Series(y_test_t_reshape[i]).tail(400), pd.Series(preds_t_reshape[i]).tail(400),CP[i])


"""# Save Model"""

model.save('LSTM_Model_26APR_30mins.keras')

