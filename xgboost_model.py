import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import yfinance as yf
import datetime
from joblib import dump, load
import os
from utils import get_stock_data, calculate_metrics, ensure_dir_exists, logger, timer_decorator

@timer_decorator
def prepare_features(data, window_sizes=[5, 10, 20, 50]):
    """
    Prepare features for XGBoost model
    
    Parameters:
    - data: DataFrame with OHLCV data
    - window_sizes: List of window sizes for technical indicators
    
    Returns:
    - X: Features DataFrame
    - y: Target Series
    - prepared_data: DataFrame with all features
    """
    df = data.copy()
    
    # Technical indicators
    # Moving averages
    for window in window_sizes:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA_Ratio{window}'] = df['Close'] / df[f'MA{window}']
        
    # Volatility
    for window in window_sizes:
        df[f'Volatility{window}'] = df['Close'].pct_change().rolling(window=window).std()
        
    # Price momentum
    for window in window_sizes:
        df[f'Return{window}'] = df['Close'].pct_change(periods=window)
        
    # Volume features
    for window in window_sizes:
        df[f'Volume_MA{window}'] = df['Volume'].rolling(window=window).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    # RSI
    for window in [7, 14, 21]:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, 0.001)  # Avoid division by zero
        df[f'RSI{window}'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    # Target variable - next day's closing price
    df['Next_Close'] = df['Close'].shift(-1)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    # Features and target
    X = df.drop(['Next_Close'], axis=1)
    y = df['Next_Close']
    
    return X, y, df

@timer_decorator
def train_xgboost_model(X, y, test_size=0.2, random_state=42):
    """
    Train XGBoost model
    
    Parameters:
    - X: Features DataFrame
    - y: Target Series
    - test_size: Proportion of data to use for testing
    - random_state: Random seed for reproducibility
    
    Returns:
    - Dictionary with model and results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Print results
    logger.info("XGBoost Model Results:")
    logger.info(f"Train R²: {train_metrics['r2']:.4f}")
    logger.info(f"Test R²: {test_metrics['r2']:.4f}")
    logger.info(f"Train RMSE: {train_metrics['rmse']:.4f}")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
    
    # Save model
    ensure_dir_exists("models")
    dump(model, "models/xgboost_model.joblib")
    dump(scaler, "models/xgboost_scaler.joblib")
    
    return {
        'model': model,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'train_r2': train_metrics['r2'],
        'test_r2': test_metrics['r2'],
        'train_rmse': train_metrics['rmse'],
        'test_rmse': test_metrics['rmse'],
        'train_mae': train_metrics['mae'],
        'test_mae': test_metrics['mae'],
        'train_mse': train_metrics['mse'],
        'test_mse': test_metrics['mse']
    }

def predict_future_xgboost(model_result, data, days=30):
    """
    Predict future stock prices using XGBoost model
    
    Parameters:
    - model_result: Dictionary with model and results from train_xgboost_model
    - data: Original stock data
    - days: Number of days to predict
    
    Returns:
    - future_dates: Array of future dates
    - future_prices: Array of predicted prices
    """
    model = model_result['model']
    scaler = model_result['scaler']
    
    # Get last date from data
    last_date = data.index[-1]
    
    # Create future dates
    future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(days)]
    
    # Get last available data for prediction
    last_data = data.copy()
    future_prices = []
    
    for i in range(days):
        # Prepare features for prediction
        X_last, _, _ = prepare_features(last_data)
        X_last_scaled = scaler.transform(X_last.iloc[-1:])
        
        # Predict next day's price
        next_price = model.predict(X_last_scaled)[0]
        future_prices.append(next_price)
        
        # Update data with prediction for next iteration
        new_row = pd.DataFrame(
            index=[future_dates[i]],
            data={
                'Open': next_price,
                'High': next_price * 1.01,  # Estimate
                'Low': next_price * 0.99,   # Estimate
                'Close': next_price,
                'Volume': last_data['Volume'].iloc[-1]  # Use last volume as estimate
            }
        )
        
        # Append to the data for next iteration
        last_data = pd.concat([last_data, new_row])
    
    return future_dates, future_prices

def plot_xgboost_results(data, model_result, future_dates=None, future_prices=None, ticker="STOCK"):
    """
    Plot XGBoost results
    
    Parameters:
    - data: Original stock data
    - model_result: Dictionary with model and results
    - future_dates: List of future dates for prediction
    - future_prices: List of predicted prices
    - ticker: Stock ticker symbol
    """
    plt.figure(figsize=(14, 8))
    
    # Plot original data
    plt.plot(data.index[-100:], data['Close'][-100:], color='black', label='Actual Prices')
    
    # Plot test predictions
    y_test = model_result['y_test']
    y_test_pred = model_result['y_test_pred']
    test_dates = model_result['X_test'].index
    
    plt.scatter(test_dates, y_test, color='blue', alpha=0.3, label='Test Data')
    plt.plot(test_dates, y_test_pred, color='blue', label='Test Predictions')
    
    # Plot future predictions if available
    if future_dates is not None and future_prices is not None:
        plt.plot(future_dates, future_prices, 'r--', linewidth=2, label='Future Predictions')
    
    # Add title and labels
    plt.title(f'{ticker} Stock Price Prediction - XGBoost Model')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_xgboost_prediction.png')
    plt.show()

def plot_feature_importance(model_result, top_n=20):
    """
    Plot feature importance from XGBoost model
    
    Parameters:
    - model_result: Dictionary with model and results
    - top_n: Number of top features to display
    """
    model = model_result['model']
    feature_names = model_result['X_train'].columns
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for plotting
    feat_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(12, 8))
    plt.barh(feat_importance['Feature'][:top_n][::-1], feat_importance['Importance'][:top_n][::-1])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Features for XGBoost Model')
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png')
    plt.show()

def main():
    # Set parameters
    ticker = "AAPL"
    start_date = "2018-01-01"
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Get data
    logger.info(f"Downloading {ticker} stock data from {start_date} to {end_date}...")
    stock_data = get_stock_data(ticker, start=start_date, end=end_date)
    
    if stock_data.empty:
        logger.error("Failed to download stock data. Exiting.")
        return
    
    # Prepare features
    logger.info("Preparing features...")
    X, y, prepared_data = prepare_features(stock_data)
    
    # Train model
    logger.info("Training XGBoost model...")
    model_result = train_xgboost_model(X, y)
    
    # Predict future prices
    logger.info("Predicting future prices...")
    future_dates, future_prices = predict_future_xgboost(model_result, stock_data, days=30)
    
    # Plot results
    logger.info("Plotting results...")
    plot_xgboost_results(stock_data, model_result, future_dates, future_prices, ticker)
    plot_feature_importance(model_result)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
