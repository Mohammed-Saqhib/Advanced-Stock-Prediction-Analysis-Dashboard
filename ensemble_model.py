import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import datetime
from joblib import dump, load
from utils import get_stock_data

def get_stock_data(ticker, start_date, end_date):
    """
    Download stock data for a given ticker and date range
    """
    from utils import get_stock_data as utils_get_stock_data
    return utils_get_stock_data(ticker, start_date, end_date)

def prepare_features(data, window_size=60):
    """
    Prepare features with more technical indicators
    """
    df = data.copy()
    
    # Technical indicators
    # Moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential moving averages
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)
    df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA20']
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Price momentum
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
    df['Price_Change_10d'] = df['Close'].pct_change(periods=10)
    
    # Volume features
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
    
    # Volatility
    df['High_Low_Range'] = df['High'] - df['Low']
    df['Close_Open_Range'] = abs(df['Close'] - df['Open'])
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)  # Annualized
    
    # Target variable: next day's closing price
    df['Next_Close'] = df['Close'].shift(-1)
    
    # Add previous n days of closing prices as features
    for i in range(1, window_size + 1):
        df[f'Close_lag_{i}'] = df['Close'].shift(i)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    # Features and target
    columns_to_drop = ['Next_Close']
    if 'Adj Close' in df.columns:
        columns_to_drop.append('Adj Close')
    
    X = df.drop(columns_to_drop, axis=1)
    y = df['Next_Close']
    
    return X, y, df

def build_ensemble_model(X, y):
    """
    Build and train an ensemble of models
    """
    # Use time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Define models
    models = {
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ]),
        'Lasso': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=0.1))
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'GradientBoosting': Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]),
        'SVR': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=1.0, epsilon=0.1))
        ])
    }
    
    # Train each model and collect results
    trained_models = {}
    predictions = {}
    results = {}
    
    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Evaluate model
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print(f"{name} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")
        
        # Store model and predictions
        trained_models[name] = pipeline
        predictions[name] = y_test_pred
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'test_mae': test_mae
        }
    
    # Create simple ensemble (average of all predictions)
    ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    print(f"\nEnsemble - Test R²: {ensemble_r2:.4f}, Test MSE: {ensemble_mse:.4f}, Test MAE: {ensemble_mae:.4f}")
    
    # Create weighted ensemble based on performance
    weights = {name: results[name]['test_r2'] for name in models.keys()}
    # Normalize weights (ensure they sum to 1)
    total_weight = sum(weights.values())
    weights = {name: weight/total_weight for name, weight in weights.items()}
    
    weighted_ensemble_pred = np.zeros_like(ensemble_pred)
    for name, weight in weights.items():
        weighted_ensemble_pred += weight * predictions[name]
    
    weighted_ensemble_r2 = r2_score(y_test, weighted_ensemble_pred)
    weighted_ensemble_mse = mean_squared_error(y_test, weighted_ensemble_pred)
    weighted_ensemble_mae = mean_absolute_error(y_test, weighted_ensemble_pred)
    
    print(f"Weighted Ensemble - Test R²: {weighted_ensemble_r2:.4f}, Test MSE: {weighted_ensemble_mse:.4f}, Test MAE: {weighted_ensemble_mae:.4f}")
    
    # Save the best performing models
    for name, model in trained_models.items():
        dump(model, f'models/{name}_model.joblib')
    
    # Return everything we need for visualization and prediction
    return {
        'trained_models': trained_models,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'predictions': predictions,
        'ensemble_pred': ensemble_pred,
        'weighted_ensemble_pred': weighted_ensemble_pred,
        'weights': weights,
        'results': results,
        'ensemble_r2': ensemble_r2,
        'weighted_ensemble_r2': weighted_ensemble_r2
    }

def predict_future_ensemble(model_results, data, days=30):
    """
    Make future predictions using the ensemble model
    """
    models = model_results['trained_models']
    weights = model_results['weights']
    last_data = data.copy()
    
    # Prepare the feature set for prediction
    X_last, _, _ = prepare_features(last_data, window_size=60)
    last_row = X_last.iloc[-1:].copy()
    
    future_dates = [last_data.index[-1] + datetime.timedelta(days=i+1) for i in range(days)]
    future_prices = []
    future_prices_by_model = {name: [] for name in models.keys()}
    
    for i in range(days):
        # Get predictions from all models
        day_predictions = {}
        
        for name, model in models.items():
            try:
                pred = model.predict(last_row)[0]
                day_predictions[name] = pred
                future_prices_by_model[name].append(pred)
            except Exception as e:
                print(f"Error in {name} model prediction: {e}")
        
        # Calculate weighted ensemble prediction
        weighted_pred = sum(day_predictions[name] * weights[name] for name in day_predictions.keys())
        future_prices.append(weighted_pred)
        
        # Update the last row for next prediction (this is a simplified approach)
        # For a real implementation, we would need to update all features properly
        if i < days - 1:
            # Create a new row for the next prediction
            new_row = last_row.copy()
            new_row.loc[new_row.index[0], 'Close'] = weighted_pred
            new_row.loc[new_row.index[0], 'Open'] = weighted_pred
            new_row.loc[new_row.index[0], 'High'] = weighted_pred * 1.01
            new_row.loc[new_row.index[0], 'Low'] = weighted_pred * 0.99
            
            # Shift the lag features
            for j in range(60, 1, -1):
                if f'Close_lag_{j}' in new_row.columns and f'Close_lag_{j-1}' in new_row.columns:
                    new_row.loc[new_row.index[0], f'Close_lag_{j}'] = new_row.loc[new_row.index[0], f'Close_lag_{j-1}']
            
            if 'Close_lag_1' in new_row.columns:
                new_row.loc[new_row.index[0], 'Close_lag_1'] = last_row.loc[last_row.index[0], 'Close']
                
            last_row = new_row
    
    return future_dates, future_prices, future_prices_by_model

def visualize_ensemble_results(data, model_results, future_dates=None, future_prices=None, future_prices_by_model=None, ticker="STOCK"):
    """
    Visualize the results of the ensemble model
    """
    plt.figure(figsize=(14, 8))
    
    # Plot actual prices
    plt.plot(data.index[-100:], data['Close'][-100:], color='black', label='Actual Prices')
    
    # Plot test predictions
    y_test = model_results['y_test']
    weighted_ensemble_pred = model_results['weighted_ensemble_pred']
    X_test_idx = model_results['X_test'].index
    
    plt.plot(X_test_idx, weighted_ensemble_pred, color='blue', label='Ensemble Predictions')
    
    # Plot individual model predictions for comparison
    for name, preds in model_results['predictions'].items():
        plt.plot(X_test_idx, preds, alpha=0.3, linestyle='--', label=f'{name} Predictions')
    
    # Plot future predictions
    if future_dates is not None and future_prices is not None:
        plt.plot(future_dates, future_prices, 'r--', linewidth=2, label='Future Predictions (Ensemble)')
        
        # Plot individual model future predictions
        if future_prices_by_model is not None:
            for name, preds in future_prices_by_model.items():
                plt.plot(future_dates, preds, alpha=0.2, linestyle=':', label=f'Future {name}')
    
    plt.title(f'{ticker} Stock Price Prediction - Ensemble Model')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_ensemble_prediction.png')
    plt.show()

def main():
    # Set parameters
    ticker = "AAPL"
    start_date = "2015-01-01"  # More historical data
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Create models directory if not exists
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Get data
    print(f"Downloading {ticker} stock data from {start_date} to {end_date}...")
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    # Prepare features
    print("Preparing features...")
    X, y, prepared_data = prepare_features(stock_data)
    
    # Build and train ensemble model
    print("Building ensemble model...")
    model_results = build_ensemble_model(X, y)
    
    # Predict future prices
    print("Predicting future prices...")
    future_dates, future_prices, future_prices_by_model = predict_future_ensemble(model_results, stock_data, days=30)
    
    # Visualize results
    print("Visualizing results...")
    visualize_ensemble_results(stock_data, model_results, future_dates, future_prices, future_prices_by_model, ticker)
    
    print("Done!")

if __name__ == "__main__":
    main()
