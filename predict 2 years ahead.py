import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the data
file_path = "/content/air quality.csv"
data = pd.read_csv(file_path)

# Data Cleaning and Preprocessing
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
data = data.dropna(subset=['PM2.5 (µg/m³)'])

# Select relevant columns and set the timestamp as the index
data.set_index('Timestamp', inplace=True)
data = data[['PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO (µg/m³)', 'NO2 (µg/m³)', 
             'NOx (ppb)', 'SO2 (µg/m³)', 'CO (mg/m³)', 'AT (°C)', 
             'RH (%)', 'WS (m/s)', 'WD (deg)', 'RF (mm)', 
             'TOT-RF (mm)', 'SR (W/mt2)', 'BP (mmHg)']]

# Scale the features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Function to create sequences for LSTM
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 0])  # PM2.5 is the first column
    return np.array(X), np.array(y)

# Create sequences
time_steps = 24  # Using 24 hours of data for prediction
X, y = create_sequences(data_scaled, time_steps)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# Predicting future values
def predict_future(model, last_data, future_steps):
    predictions = []
    current_input = last_data
    for _ in range(future_steps):
        pred = model.predict(current_input.reshape(1, time_steps, current_input.shape[1]))
        predictions.append(pred[0][0])
        # Update the input for the next prediction
        current_input = np.append(current_input[1:], [[pred]], axis=0)
    return np.array(predictions)

# Predict PM2.5 for the next 2 years (assuming hourly data)
future_steps = 2 * 365 * 24  # 2 years ahead
last_data = data_scaled[-time_steps:]  # Get the last available data

future_predictions = predict_future(model, last_data, future_steps)

# Inverse transform the predictions to original scale
future_predictions_inv = scaler.inverse_transform(np.concatenate((future_predictions.reshape(-1, 1), 
                                                               np.zeros((future_predictions.size, data.shape[1] - 1))), 
                                                              axis=1))[:, 0]

# Print future predictions
for i, pred in enumerate(future_predictions_inv):
    print(f"Prediction for hour {i + 1}: PM2.5 = {pred:.2f} µg/m³")
