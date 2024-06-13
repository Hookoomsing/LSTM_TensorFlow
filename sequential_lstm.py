import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# File paths
file_paths = {
    'X_train': '/Users/hanshookoomsing/Documents/undergraduate_project/LSTM_TensorFlow/X_train.npy',
    'X_val': '/Users/hanshookoomsing/Documents/undergraduate_project/LSTM_TensorFlow/X_val.npy',
    'X_test': '/Users/hanshookoomsing/Documents/undergraduate_project/LSTM_TensorFlow/X_test.npy',
    'y_train': '/Users/hanshookoomsing/Documents/undergraduate_project/LSTM_TensorFlow/y_train.npy',
    'y_val': '/Users/hanshookoomsing/Documents/undergraduate_project/LSTM_TensorFlow/y_val.npy',
    'y_test': '/Users/hanshookoomsing/Documents/undergraduate_project/LSTM_TensorFlow/y_test.npy'
}

# Error handling
def load_data(file_path):
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

# Load datasets
X_train = load_data(file_paths['X_train'])
X_val = load_data(file_paths['X_val'])
X_test = load_data(file_paths['X_test'])
y_train = load_data(file_paths['y_train'])
y_val = load_data(file_paths['y_val'])
y_test = load_data(file_paths['y_test'])

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    # Predict stock price
    model.add(Dense(1))  
    return model

# Build model
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_lstm_model(input_shape)
model.summary()

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), verbose=1)

# Evaluate model on TEST set
test_loss, test_mae, test_mape = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Loss: {test_loss}')
print(f'Test MAE: {test_mae}')
print(f'Test MAPE: {test_mape}')

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plot training & validation MAE values
plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plot training & validation MAPE values
plt.figure(figsize=(12, 6))
plt.plot(history.history['mape'], label='Train MAPE')
plt.plot(history.history['val_mape'], label='Validation MAPE')
plt.title('Model MAPE')
plt.ylabel('MAPE')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()