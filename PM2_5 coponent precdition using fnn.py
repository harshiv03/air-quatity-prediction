import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the data
file_path = "/content/air quality.csv"
data = pd.read_csv(file_path)

# Data Cleaning
# Convert Timestamp to DateTime
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

# Drop columns with too many missing values (more than 50% missing)
data = data.drop(columns=['NH3 (µg/m³)', 'Ozone (µg/m³)', 'Benzene (µg/m³)',
                          'Toluene (µg/m³)', 'Xylene (µg/m³)', 'O Xylene (µg/m³)',
                          'Eth-Benzene (µg/m³)', 'MP-Xylene (µg/m³)', 'VWS (m/s)'])

# Drop rows with missing values in target variable (PM2.5)
data = data.dropna(subset=['PM2.5 (µg/m³)'])

# Separate Timestamp for later merging
timestamp = data['Timestamp']

# Select only numeric columns for imputation
numeric_data = data.select_dtypes(include=[np.number])

# Fill missing values with median strategy for numeric columns
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data),
                            columns=numeric_data.columns)

# Merge the Timestamp back
data = pd.concat([timestamp, data_imputed], axis=1)

# Feature Engineering
# Extract hour, day of the week, and month from Timestamp
data['hour'] = data['Timestamp'].dt.hour
data['dayofweek'] = data['Timestamp'].dt.dayofweek
data['month'] = data['Timestamp'].dt.month

# Drop the original Timestamp column
data = data.drop(columns=['Timestamp'])

# Define features (X) and target (y)
X = data.drop(columns=['PM2.5 (µg/m³)'])
y = data['PM2.5 (µg/m³)']

# Impute missing values in the target variable before splitting
y = pd.Series(SimpleImputer(strategy='median').fit_transform(y.values.reshape(-1, 1)).flatten())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Store the column names before scaling
column_names = X_train.columns

# Check for missing values in features and handle if necessary
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=column_names)
X_test = pd.DataFrame(imputer.transform(X_test), columns=column_names)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a deep learning model (Feedforward Neural Network)
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # Regression output layer

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# User Input for New Data
print("Enter the values for the new data point:")
hour = int(input("Hour of the day (0-23): "))
dayofweek = int(input("Day of the week (0=Monday, 6=Sunday): "))
month = int(input("Month (1-12): "))
PM10 = float(input("PM10 (µg/m³): "))
NO = float(input("NO (µg/m³): "))
NO2 = float(input("NO2 (µg/m³): "))
NOx = float(input("NOx (ppb): "))
SO2 = float(input("SO2 (µg/m³): "))
CO = float(input("CO (mg/m³): "))
AT = float(input("AT (°C): "))
RH = float(input("RH (%)): "))
WS = float(input("WS (m/s): "))
WD = float(input("WD (deg): "))
RF = float(input("RF (mm): "))
TOT_RF = float(input("TOT-RF (mm): "))
SR = float(input("SR (W/mt2): "))
BP = float(input("BP (mmHg): "))

# Create a new DataFrame with the user input
new_data = pd.DataFrame({
    'hour': [hour],
    'dayofweek': [dayofweek],
    'month': [month],
    'PM10 (µg/m³)': [PM10],
    'NO (µg/m³)': [NO],
    'NO2 (µg/m³)': [NO2],
    'NOx (ppb)': [NOx],
    'SO2 (µg/m³)': [SO2],
    'CO (mg/m³)': [CO],
    'AT (°C)': [AT],
    'RH (%)': [RH],
    'WS (m/s)': [WS],
    'WD (deg)': [WD],
    'RF (mm)': [RF],
    'TOT-RF (mm)': [TOT_RF],
    'SR (W/mt2)': [SR],
    'BP (mmHg)': [BP],
})

# Ensure the new data has the same columns as the training data
new_data = new_data[column_names]

# Ensure the new data is processed in the same way as training data
new_data_imputed = pd.DataFrame(imputer.transform(new_data), columns=column_names)
new_data_scaled = scaler.transform(new_data_imputed)

# Predict PM2.5 for the new data
new_prediction = model.predict(new_data_scaled)

print(f"Predicted PM2.5: {new_prediction[0][0]}")
