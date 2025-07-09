import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utils import get_stock_data
import datetime

def get_stock_data(ticker, start_date, end_date):
    """
    Download stock data for a given ticker and date range
    """
    from utils import get_stock_data as utils_get_stock_data
    return utils_get_stock_data(ticker, start_date, end_date)

def prepare_data(data):
    """
    Prepare data for linear regression model
    """
    # Reset index to make Date a column
    data = data.reset_index()
    
    # Convert Date to numeric for model training
    data['Date'] = mdates.date2num(data['Date'])
    
    # Create features and target
    X = data[['Date']]  # Features
    y = data['Close']   # Target (closing price)
    
    return X, y

def train_model(X, y):
    """
    Train a linear regression model and evaluate its performance
    """
    # Split data into training and testing sets (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate model
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    print(f"Train R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Model Accuracy: {test_r2 * 100:.2f}%")
    
    return model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred

def predict_future(model, data, days=30):
    """
    Predict stock prices for future dates
    """
    last_date = mdates.num2date(data['Date'].iloc[-1])
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days+1)]
    future_dates_num = mdates.date2num(future_dates)
    
    # Make predictions
    future_predictions = model.predict(future_dates_num.reshape(-1, 1))
    
    return future_dates, future_predictions


def visualize_results(data, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, 
                     future_dates=None, future_predictions=None, ticker="STOCK"):
    # Create a figure
    plt.figure(figsize=(12, 6))
    
    # Get train and test dates from the respective DataFrames
    train_dates = X_train.index
    test_dates = X_test.index
    
    # Plot training data
    plt.scatter(train_dates, y_train, color='blue', label='Training Data', alpha=0.3)
    plt.plot(train_dates, y_train_pred, color='blue', label='Training Predictions')
    
    # Plot testing data
    plt.scatter(test_dates, y_test, color='green', label='Testing Data', alpha=0.3)
    plt.plot(test_dates, y_test_pred, color='green', label='Testing Predictions')
    
    # Plot future predictions if available
    if future_dates is not None and future_predictions is not None:
        plt.plot(future_dates, future_predictions, color='red', linestyle='--', label='Future Predictions')
    
    plt.title(f'{ticker} Stock Price Prediction', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Price ($)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_linear_prediction.png')
    plt.show()

def main():
    # Set parameters
    ticker = "AAPL"  # Apple stock
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    
    # Get data
    print(f"Downloading {ticker} stock data from {start_date} to {end_date}...")
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    # Prepare data
    X, y = prepare_data(stock_data)
    
    # Train model and evaluate
    print("Training model...")
    model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = train_model(X, y)
    
    # Predict future prices
    print("Predicting future prices...")
    future_dates, future_predictions = predict_future(model, X, days=30)
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(stock_data.reset_index(), X_train, X_test, y_train, y_test, 
                     y_train_pred, y_test_pred, future_dates, future_predictions, ticker)
    
    print("Done!")

if __name__ == "__main__":
    main()
