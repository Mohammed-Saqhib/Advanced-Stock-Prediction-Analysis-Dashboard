import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from utils import get_stock_data

def get_stock_data(ticker, start_date, end_date):
    """
    Download stock data for a given ticker and date range
    """
    from utils import get_stock_data as utils_get_stock_data
    return utils_get_stock_data(ticker, start_date, end_date)

def create_sequences(data, target_column, sequence_length=60):
    """
    Create sequences for LSTM model
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        seq = data.iloc[i:i+sequence_length]
        target = data.iloc[i+sequence_length][target_column]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def prepare_lstm_data(data, sequence_length=60, target_column='Close', test_size=0.2):
    """
    Prepare data for LSTM model
    """
    # Select features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = data[features].copy()
    
    # Add technical indicators
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    # Scale data
    scaler_dict = {}
    for column in df.columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
        scaler_dict[column] = scaler
    
    # Create sequences
    X, y = create_sequences(df, target_column, sequence_length)
    
    # Reshape X to be 3D: [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], len(df.columns))
    
    # Split into train and test sets
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler_dict, df

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI)
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def build_lstm_model(input_shape, neurons=50, dropout_rate=0.2, learning_rate=0.001):
    """
    Build LSTM model
    """
    model = Sequential()
    
    # First LSTM layer with dropout
    model.add(LSTM(neurons, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer with dropout
    model.add(LSTM(neurons, return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    # Dense output layer
    model.add(Dense(1))
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

def train_lstm_model(X_train, y_train, X_test, y_test, batch_size=32, epochs=100, patience=20, model_path='models/lstm_model.h5'):
    """
    Train LSTM model with early stopping
    """
    # Create directory if not exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    # Model checkpoint to save best model
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return model, history

def evaluate_lstm_model(model, X_test, y_test, scaler_dict, target_column='Close'):
    """
    Evaluate LSTM model
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    scaler = scaler_dict[target_column]
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return y_test_inv, y_pred_inv, mse, rmse, mae, r2

def predict_future_lstm(model, data, scaler_dict, sequence_length=60, days=30, target_column='Close'):
    """
    Predict future stock prices using LSTM model
    """
    # Get the last sequence from the data
    last_sequence = data[-sequence_length:].values
    
    # Scale the sequence
    scaled_sequence = np.zeros_like(last_sequence)
    for i, column in enumerate(data.columns):
        scaled_sequence[:, i] = scaler_dict[column].transform(last_sequence[:, i].reshape(-1, 1)).flatten()
    
    # Initialize future predictions
    future_prices = []
    current_sequence = scaled_sequence.copy()
    
    # Generate predictions for the next 'days' days
    for _ in range(days):
        # Reshape sequence for prediction
        current_sequence_reshaped = current_sequence.reshape(1, sequence_length, len(data.columns))
        
        # Predict the next day's price
        next_price_scaled = model.predict(current_sequence_reshaped)[0, 0]
        
        # Inverse transform the prediction
        next_price = scaler_dict[target_column].inverse_transform(
            np.array([[next_price_scaled]])
        )[0, 0]
        
        # Add to future prices
        future_prices.append(next_price)
        
        # Update the sequence for the next prediction
        # This is a simplified approach; ideally we'd update all features
        new_row = current_sequence[-1].copy()
        new_row[data.columns.get_loc(target_column)] = next_price_scaled
        
        # Roll the window
        current_sequence = np.vstack([current_sequence[1:], [new_row]])
    
    # Generate future dates
    last_date = data.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(days)]
    
    return future_dates, future_prices

def plot_lstm_results(data, y_test_inv, y_pred_inv, future_dates=None, future_prices=None, ticker="STOCK"):
    """
    Plot LSTM results
    """
    plt.figure(figsize=(14, 8))
    
    # Plot actual prices
    plt.plot(data.index[-100:], data['Close'][-100:], color='black', label='Actual Prices')
    
    # Calculate the dates for test predictions
    test_dates = data.index[-len(y_test_inv):]
    
    # Plot test predictions
    plt.plot(test_dates, y_pred_inv, color='blue', label='LSTM Predictions')
    
    # Plot future predictions
    if future_dates is not None and future_prices is not None:
        plt.plot(future_dates, future_prices, 'r--', linewidth=2, label='Future Predictions')
    
    plt.title(f'{ticker} Stock Price Prediction - LSTM Model')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_lstm_prediction.png')
    plt.show()

def plot_training_history(history):
    """
    Plot training history
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('lstm_training_history.png')
    plt.show()

def main():
    # Set parameters
    ticker = "AAPL"
    start_date = "2015-01-01"
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    sequence_length = 60
    
    # Get data
    print(f"Downloading {ticker} stock data from {start_date} to {end_date}...")
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    # Prepare data
    print("Preparing data for LSTM...")
    X_train, X_test, y_train, y_test, scaler_dict, prepared_data = prepare_lstm_data(
        stock_data, sequence_length=sequence_length
    )
    
    # Build and train model
    print("Training LSTM model...")
    model, history = train_lstm_model(
        X_train, y_train, X_test, y_test, 
        batch_size=32, epochs=100, patience=15
    )
    
    # Evaluate model
    print("Evaluating model...")
    y_test_inv, y_pred_inv, mse, rmse, mae, r2 = evaluate_lstm_model(
        model, X_test, y_test, scaler_dict
    )
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    # Predict future prices
    print("Predicting future prices...")
    future_dates, future_prices = predict_future_lstm(
        model, prepared_data, scaler_dict, sequence_length=sequence_length, days=30
    )
    
    # Plot results
    print("Plotting results...")
    plot_lstm_results(
        stock_data, y_test_inv, y_pred_inv, 
        future_dates, future_prices, ticker
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
