import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils import get_stock_data

def get_stock_data(ticker, start_date, end_date):
    """
    Download stock data for a given ticker and date range
    """
    from utils import get_stock_data as utils_get_stock_data
    return utils_get_stock_data(ticker, start_date, end_date)

def prepare_prophet_data(data):
    """
    Prepare data for Prophet model
    """
    # Prophet requires a specific dataframe format with 'ds' and 'y' columns
    df = data.reset_index()
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    
    # Add additional regressors
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    
    return df

def train_prophet_model(data, test_size=0.2):
    """
    Train Prophet model with additional regressors
    """
    # Split data into training and testing sets
    train_size = int(len(data) * (1 - test_size))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # Initialize and train Prophet model
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0
    )
    
    # Add additional regressors
    model.add_regressor('day_of_week')
    model.add_regressor('month')
    
    # Fit the model
    model.fit(train_data)
    
    # Make predictions for test data
    forecast = model.predict(test_data)
    
    # Calculate metrics
    y_true = test_data['y'].values
    y_pred = forecast['yhat'].values
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Prophet Model - Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return model, forecast, train_data, test_data, mse, rmse, mae, r2

def predict_future_prophet(model, data, days=30):
    """
    Predict future stock prices using Prophet model
    """
    # Create a dataframe for future dates
    future = model.make_future_dataframe(periods=days)
    
    # Add additional regressors to future dataframe
    future['day_of_week'] = future['ds'].dt.dayofweek
    future['month'] = future['ds'].dt.month
    
    # Make predictions
    forecast = model.predict(future)
    
    # Get only future predictions
    future_forecast = forecast.iloc[-days:]
    future_dates = future_forecast['ds'].dt.date
    future_prices = future_forecast['yhat'].values
    lower_bound = future_forecast['yhat_lower'].values
    upper_bound = future_forecast['yhat_upper'].values
    
    return future_dates, future_prices, lower_bound, upper_bound, forecast

def visualize_prophet_results(model, forecast, train_data, test_data, future_dates=None, future_prices=None, ticker="STOCK"):
    """
    Visualize Prophet model results
    """
    # Plot the forecast
    fig1 = model.plot(forecast)
    plt.title(f'{ticker} Stock Price Forecast - Prophet Model')
    plt.tight_layout()
    plt.savefig(f'{ticker}_prophet_forecast.png')
    
    # Plot the components
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig(f'{ticker}_prophet_components.png')
    
    return fig1, fig2

def main():
    # Set parameters
    ticker = "AAPL"
    start_date = "2018-01-01"
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Get data
    print(f"Downloading {ticker} stock data from {start_date} to {end_date}...")
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    # Prepare data for Prophet
    print("Preparing data for Prophet...")
    prophet_data = prepare_prophet_data(stock_data)
    
    # Train Prophet model
    print("Training Prophet model...")
    model, forecast, train_data, test_data, mse, rmse, mae, r2 = train_prophet_model(prophet_data)
    
    # Predict future prices
    print("Predicting future prices...")
    future_dates, future_prices, lower_bound, upper_bound, full_forecast = predict_future_prophet(model, prophet_data, days=30)
    
    # Visualize results
    print("Visualizing results...")
    visualize_prophet_results(model, full_forecast, train_data, test_data, future_dates, future_prices, ticker)
    
    print("Done!")

if __name__ == "__main__":
    main()
