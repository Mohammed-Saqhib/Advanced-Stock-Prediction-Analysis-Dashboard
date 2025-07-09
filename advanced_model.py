import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import datetime
from utils import get_stock_data

def get_stock_data(ticker, start_date, end_date):
    """
    Download stock data for a given ticker and date range
    """
    from utils import get_stock_data as utils_get_stock_data
    return utils_get_stock_data(ticker, start_date, end_date)

def prepare_advanced_features(data):
    """
    Prepare advanced features for the model with enhanced technical indicators
    """
    df = data.copy()
    
    # Technical indicators
    # Moving averages
    for window in [5, 10, 20, 50, 200]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
    
    # Exponential moving averages
    for window in [5, 12, 26, 50]:
        df[f'EMA{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    for window in [20]:
        df[f'STD{window}'] = df['Close'].rolling(window=window).std()
        df[f'Upper_Band{window}'] = df[f'MA{window}'] + (df[f'STD{window}'] * 2)
        df[f'Lower_Band{window}'] = df[f'MA{window}'] - (df[f'STD{window}'] * 2)
        df[f'BB_Width{window}'] = (df[f'Upper_Band{window}'] - df[f'Lower_Band{window}']) / df[f'MA{window}']
        df[f'BB_Position{window}'] = (df['Close'] - df[f'Lower_Band{window}']) / (df[f'Upper_Band{window}'] - df[f'Lower_Band{window}'])
    
    # Relative Strength Index (RSI)
    for window in [7, 14, 21]:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df[f'RSI{window}'] = 100 - (100 / (1 + rs))
    
    # Price momentum
    for window in [1, 5, 10, 21]:
        df[f'Price_Change_{window}d'] = df['Close'].pct_change(periods=window)
    
    # Volatility
    df['High_Low_Range'] = df['High'] - df['Low']
    df['Close_Open_Range'] = abs(df['Close'] - df['Open'])
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Historical volatility
    for window in [10, 21, 63]:  # 2 weeks, 1 month, 3 months
        df[f'Volatility_{window}d'] = df['Daily_Return'].rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    # Volume indicators
    df['Volume_Change'] = df['Volume'].pct_change()
    for window in [5, 10, 20]:
        df[f'Volume_MA{window}'] = df['Volume'].rolling(window=window).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Average True Range (ATR)
    for window in [14]:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'ATR{window}'] = tr.rolling(window=window).mean()
    
    # Target variable: next day's closing price
    df['Next_Close'] = df['Close'].shift(-1)
    
    # Feature interaction and polynomial features
    df['MA5_MA20_Ratio'] = df['MA5'] / df['MA20']
    df['RSI14_change'] = df['RSI14'].diff()
    
    # Add day of week and month (calendar effects)
    df['day_of_week'] = pd.to_datetime(df.index).dayofweek
    df['month'] = pd.to_datetime(df.index).month
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    # Features and target
    columns_to_drop = ['Next_Close']
    if 'Adj Close' in df.columns:
        columns_to_drop.append('Adj Close')
    
    X = df.drop(columns_to_drop, axis=1)
    y = df['Next_Close']
    
    return X, y

def train_advanced_model(X, y):
    """
    Train multiple regression models and select the best one
    """
    # Check if we have enough data
    if len(X) < 5:  # Arbitrary small number as minimum
        raise ValueError(f"Not enough data points ({len(X)}) for modeling")
        
    # Calculate appropriate test size to ensure at least 3 samples in test set
    min_test_samples = 3
    test_size = min(0.25, max(min_test_samples / len(X), 0.1))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Create pipelines for different models
    pipelines = {
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge())
        ]),
        'Lasso': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso())
        ])
    }
    
    # Train and evaluate each model
    results = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        results[name] = {
            'model': pipeline,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        print(f"{name} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['test_r2'])
    print(f"\nBest model: {best_model[0]} with Test R²: {best_model[1]['test_r2']:.4f}")
    
    return best_model[1]

def visualize_advanced_results(data, result_dict, ticker="STOCK"):
    """
    Visualize the results of the advanced model
    """
    model = result_dict['model']
    X_test = result_dict['X_test']
    y_test = result_dict['y_test']
    y_test_pred = result_dict['y_test_pred']
    
    plt.figure(figsize=(12, 6))
    
    # Get dates from index
    test_dates = X_test.index
    
    # Plot actual vs predicted prices
    plt.plot(test_dates, y_test, color='black', label='Actual Prices')
    plt.plot(test_dates, y_test_pred, color='red', label='Predicted Prices')
    
    # Calculate future predictions (next 10 days)
    last_data = data.iloc[-50:].copy()  # Get more historical data for calculations
    last_row = data.iloc[-1:].copy()
    future_preds = []
    future_dates = [data.index[-1] + datetime.timedelta(days=i+1) for i in range(10)]
    
    print(f"Training features: {len(X_test.columns)}")
    print(f"Feature names: {list(X_test.columns)}")
    
    for i in range(10):
        # Create a new row with the same structure as X_test
        # This handles MultiIndex columns properly
        new_row = pd.DataFrame(index=[future_dates[i]], columns=X_test.columns)
        
        # Use previous prediction as the current price
        current_price = last_row['Close'].values[0] if i == 0 else future_preds[-1]
        
        # Check if we have MultiIndex columns
        is_multiindex = isinstance(X_test.columns, pd.MultiIndex)
        
        # Populate all necessary features
        for col in X_test.columns:
            col_name = col[0] if is_multiindex else col  # Get base column name
            
            # Basic price data
            if col_name == 'Open':
                new_row[col] = current_price
            elif col_name == 'High':
                new_row[col] = current_price * 1.01
            elif col_name == 'Low':
                new_row[col] = current_price * 0.99
            elif col_name == 'Close':
                new_row[col] = current_price
            elif col_name == 'Volume':
                new_row[col] = last_row['Volume'].values[0]
            # Moving averages
            elif col_name == 'MA5':
                if i == 0:
                    new_row[col] = (current_price + data['Close'].iloc[-4:].sum()) / 5
                else:
                    recent_prices = [p for p in future_preds[-4:]] if len(future_preds) >= 4 else \
                                   [p for p in future_preds] + list(data['Close'].iloc[-(5-len(future_preds)-1):].values)
                    new_row[col] = (current_price + sum(recent_prices)) / 5
            elif col_name == 'MA20':
                if i == 0:
                    new_row[col] = (current_price + data['Close'].iloc[-19:].sum()) / 20
                else:
                    recent_prices = [p for p in future_preds[-19:]] if len(future_preds) >= 19 else \
                                   [p for p in future_preds] + list(data['Close'].iloc[-(20-len(future_preds)-1):].values)
                    new_row[col] = (current_price + sum(recent_prices)) / 20
            elif col_name == 'MA50':
                if i == 0:
                    new_row[col] = (current_price + data['Close'].iloc[-49:].sum()) / 50
                else:
                    recent_prices = [p for p in future_preds[-49:]] if len(future_preds) >= 49 else \
                                   [p for p in future_preds] + list(data['Close'].iloc[-(50-len(future_preds)-1):].values)
                    new_row[col] = (current_price + sum(recent_prices)) / 50
            # Price changes
            elif col_name == 'Price_Change':
                prev_close = last_row['Close'].values[0] if i == 0 else future_preds[-1]
                new_row[col] = (current_price / prev_close) - 1
            elif col_name == 'Price_Change_5d':
                prev_close_5d = data['Close'].iloc[-5] if i == 0 else \
                              (data['Close'].iloc[-5+i] if i < 5 else future_preds[i-5])
                new_row[col] = (current_price / prev_close_5d) - 1
            # Volume features
            elif col_name == 'Volume_Change':
                new_row[col] = 0  # Assuming no change for simplicity
            elif col_name == 'Volume_MA5':
                new_row[col] = last_data['Volume'].iloc[-5:].mean()
            # Volatility metrics
            elif col_name == 'High_Low_Range':
                high_val = current_price * 1.01
                low_val = current_price * 0.99
                new_row[col] = high_val - low_val
            elif col_name == 'Close_Open_Range':
                new_row[col] = abs(current_price - current_price)  # Same for simplicity
                
        # Verify we have no NaN values
        if new_row.isnull().any().any():
            print(f"Warning: NaN values in prediction data for day {i+1}")
            print("Columns with NaN:", new_row.columns[new_row.isnull().any()].tolist())
            # Fill any remaining NaNs with 0 to avoid errors
            new_row.fillna(0, inplace=True)
        
        # Make the prediction
        try:
            next_pred = model.predict(new_row)
            future_preds.append(next_pred[0])
            last_row = new_row.copy()
        except Exception as e:
            print(f"Error in prediction at day {i+1}: {e}")
            print(f"Expected features: {list(X_test.columns)}")
            print(f"Actual features: {list(new_row.columns)}")
            print(f"Missing: {set(X_test.columns) - set(new_row.columns)}")
            print(f"Extra: {set(new_row.columns) - set(X_test.columns)}")
            raise
    
    # Plot future predictions
    plt.plot(future_dates, future_preds, 'b--', label='Future Predictions')
    
    plt.title(f'{ticker} Stock Price Prediction - Advanced Model')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_advanced_prediction.png')
    plt.show()

def feature_importance(model_result):
    """
    Display feature importance if available
    """
    try:
        # Get the model from pipeline
        model = model_result['model'].named_steps['model']
        
        # Get feature names
        feature_names = model_result['X_train'].columns
        
        # For linear models we can use coefficients as importance
        if hasattr(model, 'coef_'):
            coefficients = model.coef_
            
            # Create DataFrame with feature names and coefficients
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(coefficients)
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'], importance_df['Importance'])
            plt.xlabel('Absolute Coefficient Value')
            plt.ylabel('Feature')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.show()
            
            return importance_df
        else:
            print("Model doesn't have attribute 'coef_'")
            return None
    except Exception as e:
        print(f"Error displaying feature importance: {e}")
        return None

def main():
    # Set parameters
    ticker = "AAPL"
    start_date = "2018-01-01"
    end_date = "2023-01-01"
    
    # Get data
    print(f"Downloading {ticker} stock data from {start_date} to {end_date}...")
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    # Prepare advanced features
    print("Preparing advanced features...")
    X, y = prepare_advanced_features(stock_data)
    
    # Train and select best model
    print("Training models...")
    best_model_result = train_advanced_model(X, y)
    
    # Visualize results
    print("Visualizing results...")
    visualize_advanced_results(stock_data, best_model_result, ticker)
    
    # Display feature importance
    print("Analyzing feature importance...")
    feature_importance(best_model_result)
    
    print("Done!")

if __name__ == "__main__":
    main()
