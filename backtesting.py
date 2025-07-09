import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import datetime
from tqdm import tqdm
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import logging
from pathlib import Path
import seaborn as sns
from matplotlib.dates import MonthLocator, DateFormatter
import warnings

# Import project modules
import stock_prediction
import advanced_model
import ensemble_model
try:
    import lstm_model
    has_tensorflow = True
except ImportError:
    has_tensorflow = False
try:
    import prophet_model
    has_prophet = True
except ImportError:
    has_prophet = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("backtest")

# Create cache directory
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download stock data for a given ticker and date range with caching
    
    Parameters:
    - ticker: Stock symbol
    - start_date: Start date in YYYY-MM-DD format
    - end_date: End date in YYYY-MM-DD format
    
    Returns:
    - DataFrame with stock data
    """
    # Use our utility function instead
    from utils import get_stock_data as get_stock_data_util
    return get_stock_data_util(ticker, start=start_date, end=end_date)

def calculate_additional_metrics(actual: List[float], predicted: List[float]) -> Dict[str, float]:
    """
    Calculate additional performance metrics
    
    Parameters:
    - actual: List of actual prices
    - predicted: List of predicted prices
    
    Returns:
    - Dictionary with additional metrics
    """
    if len(actual) < 2 or len(predicted) < 2:
        return {
            'directional_accuracy': float('nan'),
            'profit_factor': float('nan'),
            'sharpe_ratio': float('nan')
        }
    
    # Actual price changes
    actual_changes = [actual[i+1] - actual[i] for i in range(len(actual)-1)]
    
    # Predicted price changes
    predicted_changes = [predicted[i+1] - predicted[i] for i in range(len(predicted)-1)]
    
    # Directional accuracy (how often the model predicts the correct direction)
    correct_direction = sum(1 for i in range(len(actual_changes)) 
                          if (actual_changes[i] > 0 and predicted_changes[i] > 0) 
                          or (actual_changes[i] < 0 and predicted_changes[i] < 0))
    
    directional_accuracy = correct_direction / len(actual_changes) if len(actual_changes) > 0 else 0
    
    # Trading simulation metrics
    gains = sum(actual_changes[i] for i in range(len(actual_changes)) if predicted_changes[i] > 0)
    losses = sum(-actual_changes[i] for i in range(len(actual_changes)) if predicted_changes[i] < 0 and actual_changes[i] < 0)
    profit_factor = gains / losses if losses != 0 else float('inf')
    
    # Sharpe ratio (simplified - assuming daily predictions)
    returns = [actual_changes[i] for i in range(len(actual_changes)) if predicted_changes[i] > 0]
    if returns and len(returns) > 1:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualize
    else:
        sharpe_ratio = 0
    
    return {
        'directional_accuracy': directional_accuracy,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio
    }

def backtest_window(args: Tuple) -> Dict:
    """
    Backtest a single window - used for parallel processing
    """
    (ticker, model_name, train_data, test_data, prediction_days, window_index) = args
    
    try:
        # Run the selected model
        if model_name == 'linear':
            X, y = stock_prediction.prepare_data(train_data)
            model, _, _, _, _, _, _ = stock_prediction.train_model(X, y)
            future_dates, future_predictions = stock_prediction.predict_future(
                model, X, days=min(prediction_days, len(test_data))
            )
            
        elif model_name == 'advanced':
            X, y = advanced_model.prepare_advanced_features(train_data)
            model_result = advanced_model.train_advanced_model(X, y)
            model = model_result['model']
            
            # Get future predictions
            last_data = train_data.copy()
            future_dates = test_data.index[:min(prediction_days, len(test_data))]
            future_predictions = []
            
            # Generate predictions using advanced_model's prediction logic
            for i in range(min(prediction_days, len(test_data))):
                # Create a new sample based on last available data
                # This is a simplified version - in production, you should recreate
                # the feature engineering steps properly
                new_sample = advanced_model.prepare_prediction_sample(last_data, future_dates[i])
                pred = model.predict(new_sample)[0]
                future_predictions.append(pred)
                # Update last data with prediction for next iteration
                new_row = pd.DataFrame(index=[future_dates[i]], 
                                      data={'Close': pred, 'Open': pred, 'High': pred*1.01, 
                                            'Low': pred*0.99, 'Volume': last_data['Volume'].iloc[-1]})
                last_data = pd.concat([last_data, new_row])
            
        elif model_name == 'ensemble':
            X, y, _ = ensemble_model.prepare_features(train_data)
            model_results = ensemble_model.build_ensemble_model(X, y)
            future_dates, future_predictions, _ = ensemble_model.predict_future_ensemble(
                model_results, train_data, days=min(prediction_days, len(test_data))
            )
            
        elif model_name == 'lstm' and has_tensorflow:
            sequence_length = 60
            X_train, X_test, y_train, y_test, scaler_dict, prepared_data = lstm_model.prepare_lstm_data(
                train_data, sequence_length=sequence_length
            )
            model, _ = lstm_model.train_lstm_model(
                X_train, y_train, X_test, y_test, 
                batch_size=32, epochs=50, patience=5
            )
            future_dates, future_predictions = lstm_model.predict_future_lstm(
                model, prepared_data, scaler_dict, sequence_length=sequence_length, days=min(prediction_days, len(test_data))
            )
            
        elif model_name == 'prophet' and has_prophet:
            prophet_data = prophet_model.prepare_prophet_data(train_data)
            model, _, _, _, _, _, _, _ = prophet_model.train_prophet_model(prophet_data, test_size=0.1)
            future_dates, future_predictions, _, _, _ = prophet_model.predict_future_prophet(
                model, prophet_data, days=min(prediction_days, len(test_data))
            )
            future_dates = [datetime.datetime.combine(d, datetime.time()) for d in future_dates]
        
        else:
            return {
                'success': False,
                'error': f"Unknown model {model_name}",
                'window_index': window_index
            }
            
        # Compare predictions to actual prices
        actual = []
        predicted = []
        dates = []
        
        for j, date in enumerate(future_dates):
            if j >= len(test_data):
                break
                
            # Find closest date in test data
            idx = test_data.index.get_indexer([date], method='nearest')[0]
            if idx < len(test_data):
                actual_price = test_data['Close'].iloc[idx]
                pred_price = future_predictions[j]
                
                dates.append(date)
                actual.append(actual_price)
                predicted.append(pred_price)
        
        # Calculate metrics for this window
        if len(actual) > 0 and len(predicted) > 0:
            window_mse = mean_squared_error(actual, predicted)
            window_r2 = r2_score(actual, predicted) if len(actual) > 1 else 0
            window_mae = mean_absolute_error(actual, predicted)
            
            # Calculate additional metrics
            additional_metrics = calculate_additional_metrics(actual, predicted)
            
            return {
                'success': True,
                'window_index': window_index,
                'dates': dates,
                'actual': actual,
                'predicted': predicted,
                'mse': window_mse,
                'r2': window_r2,
                'mae': window_mae,
                **additional_metrics
            }
        else:
            return {
                'success': False,
                'error': "No predictions could be compared to actual prices",
                'window_index': window_index
            }
            
    except Exception as e:
        logger.error(f"Error in window {window_index}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'window_index': window_index
        }

def backtest_model(ticker: str, model_name: str, start_date: str, end_date: str, 
                   window_size: int = 252, step_size: int = 21, prediction_days: int = 30,
                   parallel: bool = False, max_workers: Optional[int] = None) -> Dict:
    """
    Backtest a model over a time period using rolling windows
    
    Parameters:
    - ticker: Stock symbol
    - model_name: Model to use ('linear', 'advanced', 'ensemble', 'lstm', 'prophet')
    - start_date: Start date for backtesting
    - end_date: End date for backtesting
    - window_size: Size of each training window in days (default: 252, about 1 year of trading days)
    - step_size: Days to move forward for each test (default: 21, about 1 month)
    - prediction_days: Number of days to predict for each window
    - parallel: Whether to use parallel processing (default: False)
    - max_workers: Maximum number of parallel workers (default: None - use CPU count)
    
    Returns:
    - Dictionary with backtesting results
    """
    logger.info(f"Backtesting {model_name} model for {ticker} from {start_date} to {end_date}")
    
    # Get full data
    full_data = get_stock_data(ticker, start_date, end_date)
    if full_data.empty:
        logger.error(f"Error: No data found for {ticker}")
        return None
    
    # Initialize lists to store results
    dates = []
    actual_prices = []
    predicted_prices = []
    mse_scores = []
    r2_scores = []
    mae_scores = []
    directional_accuracy_scores = []
    profit_factor_scores = []
    sharpe_ratio_scores = []
    failed_windows = []
    
    # Generate window ranges
    total_days = len(full_data)
    windows = []
    
    for start_idx in range(0, total_days - window_size - prediction_days, step_size):
        end_idx = start_idx + window_size
        test_end_idx = min(end_idx + prediction_days, total_days)
        windows.append((start_idx, end_idx, test_end_idx))
    
    logger.info(f"Created {len(windows)} backtest windows")
    
    if parallel and len(windows) > 1:
        logger.info(f"Running backtests in parallel with {max_workers or 'auto'} workers")
        backtest_args = []
        
        for i, (start_idx, end_idx, test_end_idx) in enumerate(windows):
            # Get training data
            train_data = full_data.iloc[start_idx:end_idx]
            # Get test data (actual future prices)
            test_data = full_data.iloc[end_idx:test_end_idx]
            
            if len(test_data) == 0:
                continue
                
            backtest_args.append((ticker, model_name, train_data, test_data, prediction_days, i))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_window = {executor.submit(backtest_window, arg): arg for arg in backtest_args}
            
            for future in tqdm(as_completed(future_to_window), total=len(backtest_args), desc="Backtesting windows"):
                result = future.result()
                
                if result['success']:
                    # Collect results
                    dates.extend(result['dates'])
                    actual_prices.extend(result['actual'])
                    predicted_prices.extend(result['predicted'])
                    
                    mse_scores.append(result['mse'])
                    r2_scores.append(result['r2'])
                    mae_scores.append(result['mae'])
                    directional_accuracy_scores.append(result['directional_accuracy'])
                    profit_factor_scores.append(result['profit_factor'])
                    sharpe_ratio_scores.append(result['sharpe_ratio'])
                else:
                    failed_windows.append({
                        'window': result['window_index'],
                        'error': result.get('error', 'Unknown error'),
                        'traceback': result.get('traceback', '')
                    })
    else:
        # Sequential processing
        for i, (start_idx, end_idx, test_end_idx) in enumerate(tqdm(windows, desc="Backtesting windows")):
            # Get training data
            train_data = full_data.iloc[start_idx:end_idx]
            # Get test data (actual future prices)
            test_data = full_data.iloc[end_idx:test_end_idx]
            
            if len(test_data) == 0:
                continue
                
            result = backtest_window((ticker, model_name, train_data, test_data, prediction_days, i))
            
            if result['success']:
                # Collect results
                dates.extend(result['dates'])
                actual_prices.extend(result['actual'])
                predicted_prices.extend(result['predicted'])
                
                mse_scores.append(result['mse'])
                r2_scores.append(result['r2'])
                mae_scores.append(result['mae'])
                directional_accuracy_scores.append(result['directional_accuracy'])
                profit_factor_scores.append(result['profit_factor'])
                sharpe_ratio_scores.append(result['sharpe_ratio'])
            else:
                failed_windows.append({
                    'window': i,
                    'error': result.get('error', 'Unknown error'),
                    'traceback': result.get('traceback', '')
                })
    
    # Filter out NaN values for mean calculation
    mse_scores_filtered = [x for x in mse_scores if not np.isnan(x)]
    r2_scores_filtered = [x for x in r2_scores if not np.isnan(x)]
    mae_scores_filtered = [x for x in mae_scores if not np.isnan(x)]
    directional_accuracy_filtered = [x for x in directional_accuracy_scores if not np.isnan(x)]
    profit_factor_filtered = [x for x in profit_factor_scores if not np.isnan(x) and not np.isinf(x)]
    sharpe_ratio_filtered = [x for x in sharpe_ratio_scores if not np.isnan(x)]
    
    # Calculate overall metrics
    results = {
        'ticker': ticker,
        'model': model_name,
        'dates': dates,
        'actual_prices': actual_prices,
        'predicted_prices': predicted_prices,
        'mse': np.mean(mse_scores_filtered) if mse_scores_filtered else float('nan'),
        'rmse': np.sqrt(np.mean(mse_scores_filtered)) if mse_scores_filtered else float('nan'),
        'r2': np.mean(r2_scores_filtered) if r2_scores_filtered else float('nan'),
        'mae': np.mean(mae_scores_filtered) if mae_scores_filtered else float('nan'),
        'directional_accuracy': np.mean(directional_accuracy_filtered) if directional_accuracy_filtered else float('nan'),
        'profit_factor': np.mean(profit_factor_filtered) if profit_factor_filtered else float('nan'),
        'sharpe_ratio': np.mean(sharpe_ratio_filtered) if sharpe_ratio_filtered else float('nan'),
        'window_mse': mse_scores,
        'window_r2': r2_scores,
        'window_mae': mae_scores,
        'window_directional_accuracy': directional_accuracy_scores,
        'window_profit_factor': profit_factor_scores,
        'window_sharpe_ratio': sharpe_ratio_scores,
        'total_windows': len(windows),
        'successful_windows': len(mse_scores),
        'failed_windows': failed_windows,
        'start_date': start_date,
        'end_date': end_date,
        'window_size': window_size,
        'step_size': step_size,
        'prediction_days': prediction_days
    }
    
    logger.info(f"Backtesting completed. Overall metrics:")
    logger.info(f"MSE: {results['mse']:.4f}")
    logger.info(f"RMSE: {results['rmse']:.4f}")
    logger.info(f"R²: {results['r2']:.4f}")
    logger.info(f"MAE: {results['mae']:.4f}")
    logger.info(f"Directional Accuracy: {results['directional_accuracy']:.4f}")
    logger.info(f"Profit Factor: {results['profit_factor']:.4f}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    logger.info(f"Successful windows: {results['successful_windows']}/{results['total_windows']}")
    
    if failed_windows:
        logger.warning(f"Failed windows: {len(failed_windows)}")
        for fail in failed_windows[:5]:  # Show first 5 failures
            logger.warning(f"Window {fail['window']}: {fail['error']}")
        if len(failed_windows) > 5:
            logger.warning(f"... and {len(failed_windows) - 5} more failures")
    
    return results

def visualize_backtest_results(results: Dict, output_dir: str = 'results') -> None:
    """
    Visualize backtesting results with enhanced charts
    
    Parameters:
    - results: Results dictionary from backtest_model
    - output_dir: Directory to save output files
    """
    ticker = results['ticker']
    model_name = results['model']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set Seaborn style for nicer plots
    sns.set(style="whitegrid")
    
    dates = results['dates']
    actual_prices = results['actual_prices']
    predicted_prices = results['predicted_prices']
    
    # 1. Time series plot with actual vs predicted prices
    plt.figure(figsize=(16, 8))
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Date': dates,
        'Actual': actual_prices,
        'Predicted': predicted_prices
    }).set_index('Date')
    
    # Plot with better styling
    ax = df.plot(figsize=(16, 8), linewidth=2)
    plt.title(f'{ticker} Stock Price - Actual vs Predicted ({model_name} Model)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Price ($)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add bands showing prediction error
    plt.fill_between(df.index, df['Actual'], df['Predicted'], color='gray', alpha=0.2, label='Error')
    
    # Format x-axis dates
    ax.xaxis.set_major_locator(MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{ticker}_{model_name}_prediction_time_series.png', dpi=300)
    
    # 2. Scatter plot with regression line
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    ax = plt.subplot(111)
    ax.scatter(actual_prices, predicted_prices, alpha=0.5)
    
    # Add regression line
    z = np.polyfit(actual_prices, predicted_prices, 1)
    p = np.poly1d(z)
    plt.plot(sorted(actual_prices), p(sorted(actual_prices)), "r--", linewidth=2)
    
    # Add perfect prediction line
    min_val = min(min(actual_prices), min(predicted_prices))
    max_val = max(max(actual_prices), max(predicted_prices))
    plt.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=1, alpha=0.3)
    
    plt.title(f'{ticker} - Actual vs Predicted Price ({model_name} Model)', fontsize=16)
    plt.xlabel('Actual Price', fontsize=14)
    plt.ylabel('Predicted Price', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(actual_prices, predicted_prices)[0, 1]
    plt.annotate(f"Correlation: {corr:.4f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                 fontsize=12, ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # Add R² value
    r2 = results['r2']
    plt.annotate(f"R²: {r2:.4f}", xy=(0.05, 0.90), xycoords='axes fraction', 
                 fontsize=12, ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{ticker}_{model_name}_scatter_plot.png', dpi=300)
    
    # 3. Error distribution
    plt.figure(figsize=(14, 8))
    
    errors = np.array(predicted_prices) - np.array(actual_prices)
    percentage_errors = errors / np.array(actual_prices) * 100
    
    plt.subplot(2, 1, 1)
    sns.histplot(errors, kde=True, bins=30)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.title(f'Error Distribution - {model_name} Model', fontsize=16)
    plt.xlabel('Prediction Error ($)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    
    # Add statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.annotate(f"Mean: {mean_error:.4f}\nStd: {std_error:.4f}", 
                 xy=(0.95, 0.95), xycoords='axes fraction', 
                 fontsize=12, ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    plt.subplot(2, 1, 2)
    sns.histplot(percentage_errors, kde=True, bins=30)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.title('Percentage Error Distribution', fontsize=16)
    plt.xlabel('Prediction Error (%)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    
    # Add percentage error statistics
    mean_pct_error = np.mean(percentage_errors)
    std_pct_error = np.std(percentage_errors)
    plt.annotate(f"Mean: {mean_pct_error:.2f}%\nStd: {std_pct_error:.2f}%", 
                 xy=(0.95, 0.95), xycoords='axes fraction', 
                 fontsize=12, ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{ticker}_{model_name}_error_distribution.png', dpi=300)
    
    # 4. Window metrics over time
    plt.figure(figsize=(16, 14))
    
    # Store window indices for plotting
    window_indices = list(range(len(results['window_mse'])))
    
    plt.subplot(3, 2, 1)
    plt.plot(window_indices, results['window_mse'], 'o-', linewidth=2)
    plt.axhline(y=results['mse'], color='r', linestyle='--', alpha=0.7)
    plt.title('MSE by Window', fontsize=16)
    plt.ylabel('MSE', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 2)
    plt.plot(window_indices, results['window_r2'], 'o-', linewidth=2)
    plt.axhline(y=results['r2'], color='r', linestyle='--', alpha=0.7)
    plt.title('R² by Window', fontsize=16)
    plt.ylabel('R²', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 3)
    plt.plot(window_indices, results['window_mae'], 'o-', linewidth=2)
    plt.axhline(y=results['mae'], color='r', linestyle='--', alpha=0.7)
    plt.title('MAE by Window', fontsize=16)
    plt.ylabel('MAE', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 4)
    plt.plot(window_indices, results['window_directional_accuracy'], 'o-', linewidth=2)
    plt.axhline(y=results['directional_accuracy'], color='r', linestyle='--', alpha=0.7)
    plt.title('Directional Accuracy by Window', fontsize=16)
    plt.ylabel('Directional Accuracy', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 5)
    # Filter out inf values for plotting
    profit_factor_plot = [min(x, 10) if not np.isnan(x) and not np.isinf(x) else np.nan 
                         for x in results['window_profit_factor']]
    plt.plot(window_indices, profit_factor_plot, 'o-', linewidth=2)
    plt.axhline(y=min(results['profit_factor'], 10), color='r', linestyle='--', alpha=0.7)
    plt.title('Profit Factor by Window (capped at 10)', fontsize=16)
    plt.xlabel('Window Index', fontsize=14)
    plt.ylabel('Profit Factor', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 6)
    plt.plot(window_indices, results['window_sharpe_ratio'], 'o-', linewidth=2)
    plt.axhline(y=results['sharpe_ratio'], color='r', linestyle='--', alpha=0.7)
    plt.title('Sharpe Ratio by Window', fontsize=16)
    plt.xlabel('Window Index', fontsize=14)
    plt.ylabel('Sharpe Ratio', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{ticker}_{model_name}_window_metrics.png', dpi=300)
    
    # 5. Summary dashboard
    plt.figure(figsize=(16, 10))
    
    # Add a text box with summary metrics
    summary_text = f"""
    MODEL PERFORMANCE SUMMARY
    -------------------------
    Ticker: {ticker}
    Model: {model_name}
    Period: {results['start_date']} to {results['end_date']}
    
    Accuracy Metrics:
    - MSE: {results['mse']:.4f}
    - RMSE: {results['rmse']:.4f}
    - MAE: {results['mae']:.4f}
    - R²: {results['r2']:.4f}
    
    Trading Metrics:
    - Directional Accuracy: {results['directional_accuracy']:.4f} ({results['directional_accuracy']*100:.1f}%)
    - Profit Factor: {results['profit_factor']:.4f}
    - Sharpe Ratio: {results['sharpe_ratio']:.4f}
    
    Backtest Configuration:
    - Window Size: {results['window_size']} days
    - Step Size: {results['step_size']} days
    - Prediction Days: {results['prediction_days']} days
    - Total Windows: {results['total_windows']}
    - Successful Windows: {results['successful_windows']} ({results['successful_windows']/results['total_windows']*100:.1f}%)
    """
    
    plt.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=14,
             bbox=dict(boxstyle='round,pad=1', facecolor='white', alpha=0.8),
             family='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{ticker}_{model_name}_summary.png', dpi=300)
    
    logger.info(f"All visualizations saved to {output_dir} directory")

def compare_backtest_models(ticker: str, start_date: str, end_date: str, models: Optional[List[str]] = None,
                           window_size: int = 252, step_size: int = 21, prediction_days: int = 30,
                           parallel: bool = False, output_dir: str = 'results') -> Tuple[pd.DataFrame, Dict]:
    """
    Compare backtesting results for multiple models
    
    Parameters:
    - ticker: Stock symbol
    - start_date: Start date for backtesting
    - end_date: End date for backtesting
    - models: List of models to compare (default: all available models)
    - window_size: Size of each training window in days
    - step_size: Days to move forward for each test
    - prediction_days: Number of days to predict for each window
    - parallel: Whether to use parallel processing
    - output_dir: Directory to save output files
    
    Returns:
    - Tuple of (comparison DataFrame, dictionary of results by model)
    """
    if models is None:
        models = ['linear', 'advanced', 'ensemble']
        if has_tensorflow:
            models.append('lstm')
        if has_prophet:
            models.append('prophet')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    for model_name in models:
        logger.info(f"\nBacktesting {model_name} model...")
        model_results = backtest_model(
            ticker, model_name, start_date, end_date,
            window_size=window_size, step_size=step_size,
            prediction_days=prediction_days, parallel=parallel
        )
        if model_results:
            results[model_name] = model_results
            visualize_backtest_results(model_results, output_dir)
    
    # Compare metrics
    comparison = {
        'Model': [],
        'MSE': [],
        'RMSE': [],
        'MAE': [],
        'R²': [],
        'Directional Accuracy': [],
        'Profit Factor': [],
        'Sharpe Ratio': []
    }
    
    for model_name, model_results in results.items():
        comparison['Model'].append(model_name)
        comparison['MSE'].append(model_results['mse'])
        comparison['RMSE'].append(model_results['rmse'])
        comparison['MAE'].append(model_results['mae'])
        comparison['R²'].append(model_results['r2'])
        comparison['Directional Accuracy'].append(model_results['directional_accuracy'])
        comparison['Profit Factor'].append(model_results['profit_factor'])
        comparison['Sharpe Ratio'].append(model_results['sharpe_ratio'])
    
    comparison_df = pd.DataFrame(comparison)
    
    # Format the numeric columns
    for col in comparison_df.columns:
        if col != 'Model':
            comparison_df[col] = comparison_df[col].map('{:.4f}'.format)
    
    logger.info("\nModel Comparison:")
    logger.info(f"\n{comparison_df.to_string()}")
    
    # Save comparison to CSV
    comparison_df.to_csv(f'{output_dir}/{ticker}_model_comparison.csv', index=False)
    
    # Create bar plots comparing different metrics
    plt.figure(figsize=(16, 14))
    
    metrics = ['MSE', 'RMSE', 'MAE', 'R²', 'Directional Accuracy', 'Profit Factor', 'Sharpe Ratio']
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 3, i+1)
        
        # Convert string back to float for plotting
        values = [float(val) for val in comparison[metric]]
        
        # Sort models by performance (ascending for error metrics, descending for others)
        sort_ascending = metric in ['MSE', 'RMSE', 'MAE']
        sorted_indices = np.argsort(values)
        if not sort_ascending:
            sorted_indices = sorted_indices[::-1]
            
        sorted_models = [comparison['Model'][i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        # Create color map - green for best, red for worst
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(sorted_values)))
        if sort_ascending:
            colors = colors[::-1]  # Reverse for error metrics
            
        bars = plt.bar(sorted_models, sorted_values, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                     f'{height:.4f}', ha='center', va='bottom', rotation=0, fontsize=9)
        
        plt.title(f'{metric} Comparison', fontsize=14)
        plt.ylabel(metric, fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        # For R² and accuracy metrics, set y-axis from 0 to 1 for better visualization
        if metric in ['R²', 'Directional Accuracy']:
            plt.ylim(0, 1)
            
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{ticker}_model_comparison_metrics.png', dpi=300)
    
    # Create combined time series plot
    plt.figure(figsize=(16, 10))
    
    # Get common date range
    all_dates = set()
    for model_name, model_results in results.items():
        all_dates.update(model_results['dates'])
    
    common_dates = sorted(all_dates)
    
    # Create DataFrame for actual prices
    actual_df = pd.DataFrame(index=common_dates, columns=['Actual'])
    
    # Add predicted prices for each model
    for model_name, model_results in results.items():
        dates_dict = {d: i for i, d in enumerate(model_results['dates'])}
        
        for date in common_dates:
            if date in dates_dict:
                idx = dates_dict[date]
                if 'actual_prices' in model_results and idx < len(model_results['actual_prices']):
                    actual_df.loc[date, 'Actual'] = model_results['actual_prices'][idx]
                if 'predicted_prices' in model_results and idx < len(model_results['predicted_prices']):
                    actual_df.loc[date, model_name] = model_results['predicted_prices'][idx]
    
    # Plot
    ax = actual_df.plot(figsize=(16, 10), linewidth=2)
    
    plt.title(f'{ticker} Stock Price Predictions - Model Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Price ($)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Format x-axis dates
    ax.xaxis.set_major_locator(MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{ticker}_all_models_comparison.png', dpi=300)
    
    # Generate HTML report
    generate_html_report(ticker, results, comparison_df, output_dir)
    
    return comparison_df, results

def generate_html_report(ticker: str, results: Dict, comparison_df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate an HTML report summarizing the backtest results
    
    Parameters:
    - ticker: Stock symbol
    - results: Dictionary of results by model
    - comparison_df: DataFrame with model comparisons
    - output_dir: Directory to save the report
    """
    # Get current date for the report
    report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{ticker} Stock Prediction Model Comparison</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: right;
            }}
            th {{
                background-color: #f2f2f2;
                text-align: center;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            .model-name {{
                text-align: left;
                font-weight: bold;
            }}
            .best-value {{
                background-color: #dff0d8;
                font-weight: bold;
            }}
            .section {{
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }}
            .image-container {{
                margin: 20px 0;
                text-align: center;
            }}
            img {{
                max-width: 100%;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                margin-bottom: 10px;
            }}
            .caption {{
                font-style: italic;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="section">
            <h1>{ticker} Stock Prediction Model Comparison</h1>
            <p><strong>Report generated on:</strong> {report_date}</p>
            <p><strong>Backtest period:</strong> {results[list(results.keys())[0]]['start_date']} to {results[list(results.keys())[0]]['end_date']}</p>
            <p><strong>Window size:</strong> {results[list(results.keys())[0]]['window_size']} days</p>
            <p><strong>Step size:</strong> {results[list(results.keys())[0]]['step_size']} days</p>
            <p><strong>Prediction days:</strong> {results[list(results.keys())[0]]['prediction_days']} days</p>
        </div>
        
        <div class="section">
            <h2>Model Performance Comparison</h2>
            <table>
                <tr>
                    <th>Model</th>
    """
    
    # Add metric columns
    metrics = ['MSE', 'RMSE', 'MAE', 'R²', 'Directional Accuracy', 'Profit Factor', 'Sharpe Ratio']
    for metric in metrics:
        html_content += f"<th>{metric}</th>\n"
    
    html_content += "</tr>\n"
    
    # Find best values for each metric
    best_values = {}
    for metric in metrics:
        values = comparison_df[metric].astype(float)
        if metric in ['MSE', 'RMSE', 'MAE']:
            best_values[metric] = values.min()
        else:
            best_values[metric] = values.max()
    
    # Add rows for each model
    for _, row in comparison_df.iterrows():
        html_content += f"<tr>\n<td class='model-name'>{row['Model']}</td>\n"
        
        for metric in metrics:
            value = float(row[metric])
            is_best = abs(value - best_values[metric]) < 1e-6
            cell_class = "best-value" if is_best else ""
            html_content += f"<td class='{cell_class}'>{value:.4f}</td>\n"
        
        html_content += "</tr>\n"
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Comparison Charts</h2>
            
            <div class="image-container">
                <img src="{ticker}_model_comparison_metrics.png" alt="Model Metrics Comparison">
                <p class="caption">Comparison of performance metrics across different models</p>
            </div>
            
            <div class="image-container">
                <img src="{ticker}_all_models_comparison.png" alt="All Models Time Series">
                <p class="caption">Time series comparison of all model predictions</p>
            </div>
        </div>
    """.format(ticker=ticker)
    
    # Add sections for individual model results
    html_content += "<h2>Individual Model Results</h2>\n"
    
    for model_name in results.keys():
        html_content += f"""
        <div class="section">
            <h3>{model_name.capitalize()} Model</h3>
            
            <div class="image-container">
                <img src="{ticker}_{model_name}_prediction_time_series.png" alt="{model_name} Time Series">
                <p class="caption">Time series of actual vs predicted prices for {model_name} model</p>
            </div>
            
            <div class="image-container">
                <img src="{ticker}_{model_name}_error_distribution.png" alt="{model_name} Error Distribution">
                <p class="caption">Error distribution for {model_name} model predictions</p>
            </div>
            
            <div class="image-container">
                <img src="{ticker}_{model_name}_window_metrics.png" alt="{model_name} Window Metrics">
                <p class="caption">Performance metrics across different backtest windows for {model_name} model</p>
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML content to file
    report_path = os.path.join(output_dir, f"{ticker}_backtest_report.html")
    with open(report_path, "w") as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated: {report_path}")

def backtest_multiple_tickers(tickers: List[str], model_name: str, start_date: str, end_date: str,
                             window_size: int = 252, step_size: int = 21, prediction_days: int = 30,
                             parallel: bool = False, output_dir: str = 'results') -> Dict:
    """
    Backtest a model on multiple tickers
    
    Parameters:
    - tickers: List of stock symbols to backtest
    - model_name: Model to use ('linear', 'advanced', 'ensemble', 'lstm', 'prophet')
    - start_date: Start date for backtesting
    - end_date: End date for backtesting
    - window_size: Size of each training window in days
    - step_size: Days to move forward for each test
    - prediction_days: Number of days to predict for each window
    - parallel: Whether to use parallel processing
    - output_dir: Directory to save output files
    
    Returns:
    - Dictionary with results for each ticker
    """
    logger.info(f"Backtesting {model_name} model for {len(tickers)} tickers")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    metric_summary = {
        'Ticker': [],
        'MSE': [],
        'RMSE': [],
        'MAE': [],
        'R²': [],
        'Directional Accuracy': [],
        'Profit Factor': [],
        'Sharpe Ratio': []
    }
    
    for ticker in tickers:
        logger.info(f"\nBacktesting {ticker}...")
        
        # Create ticker-specific output directory
        ticker_dir = os.path.join(output_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        try:
            result = backtest_model(
                ticker, model_name, start_date, end_date,
                window_size=window_size, step_size=step_size,
                prediction_days=prediction_days, parallel=parallel
            )
            
            if result:
                all_results[ticker] = result
                visualize_backtest_results(result, ticker_dir)
                
                # Add to summary
                metric_summary['Ticker'].append(ticker)
                metric_summary['MSE'].append(result['mse'])
                metric_summary['RMSE'].append(result['rmse'])
                metric_summary['MAE'].append(result['mae'])
                metric_summary['R²'].append(result['r2'])
                metric_summary['Directional Accuracy'].append(result['directional_accuracy'])
                metric_summary['Profit Factor'].append(result['profit_factor'])
                metric_summary['Sharpe Ratio'].append(result['sharpe_ratio'])
        
        except Exception as e:
            logger.error(f"Error backtesting {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(metric_summary)
    
    # Calculate average metrics
    avg_row = {
        'Ticker': 'AVERAGE',
        'MSE': summary_df['MSE'].mean(),
        'RMSE': summary_df['RMSE'].mean(),
        'MAE': summary_df['MAE'].mean(),
        'R²': summary_df['R²'].mean(),
        'Directional Accuracy': summary_df['Directional Accuracy'].mean(),
        'Profit Factor': summary_df['Profit Factor'].mean(),
        'Sharpe Ratio': summary_df['Sharpe Ratio'].mean()
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])])
    
    # Save summary to CSV
    summary_df.to_csv(f'{output_dir}/{model_name}_multi_ticker_summary.csv', index=False)
    
    logger.info("\nMulti-ticker Backtest Summary:")
    logger.info(f"\n{summary_df.to_string()}")
    
    # Create comparative bar charts
    plt.figure(figsize=(16, 14))
    
    metrics = ['MSE', 'RMSE', 'MAE', 'R²', 'Directional Accuracy', 'Profit Factor', 'Sharpe Ratio']
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 3, i+1)
        
        # Sort tickers by metric
        sort_ascending = metric in ['MSE', 'RMSE', 'MAE']
        sorted_df = summary_df[summary_df['Ticker'] != 'AVERAGE'].sort_values(metric, ascending=sort_ascending)
        
        # Create color map - green for best, red for worst
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(sorted_df)))
        if sort_ascending:
            colors = colors[::-1]  # Reverse for error metrics
            
        bars = plt.bar(sorted_df['Ticker'], sorted_df[metric], color=colors)
        
        # Add average line
        plt.axhline(y=avg_row[metric], color='r', linestyle='--', alpha=0.7, label='Average')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                     f'{height:.4f}', ha='center', va='bottom', rotation=45, fontsize=8)
        
        plt.title(f'{metric} Comparison', fontsize=14)
        plt.ylabel(metric, fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        if metric in ['R²', 'Directional Accuracy']:
            plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_multi_ticker_comparison.png', dpi=300)
    
    return all_results

def main():
    """
    Main function to run the backtesting module
    """
    # Set parameters
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    
    # Create results directory
    output_dir = "results/backtest"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse command line arguments if any
    import argparse
    parser = argparse.ArgumentParser(description="Backtest stock prediction models")
    parser.add_argument("--ticker", type=str, default=ticker, help="Stock ticker symbol")
    parser.add_argument("--start", type=str, default=start_date, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=end_date, help="End date (YYYY-MM-DD)")
    parser.add_argument("--window", type=int, default=252, help="Training window size in days")
    parser.add_argument("--step", type=int, default=21, help="Step size in days")
    parser.add_argument("--predict", type=int, default=30, help="Prediction days")
    parser.add_argument("--model", type=str, default=None, help="Specific model to test")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated list of tickers")
    
    args = parser.parse_args()
    
    if args.tickers:
        ticker_list = args.tickers.split(',')
        if args.model:
            logger.info(f"Backtesting {args.model} model for multiple tickers: {ticker_list}")
            backtest_multiple_tickers(
                ticker_list, args.model, args.start, args.end,
                window_size=args.window, step_size=args.step,
                prediction_days=args.predict, parallel=args.parallel,
                output_dir=output_dir
            )
        else:
            for ticker in ticker_list:
                logger.info(f"Running backtest comparison for {ticker}")
                compare_backtest_models(
                    ticker, args.start, args.end,
                    window_size=args.window, step_size=args.step, 
                    prediction_days=args.predict, parallel=args.parallel,
                    output_dir=os.path.join(output_dir, ticker)
                )
    else:
        if args.model:
            logger.info(f"Backtesting {args.model} model for {args.ticker}")
            results = backtest_model(
                args.ticker, args.model, args.start, args.end,
                window_size=args.window, step_size=args.step,
                prediction_days=args.predict, parallel=args.parallel
            )
            if results:
                visualize_backtest_results(results, output_dir)
        else:
            logger.info(f"Running backtest comparison for {args.ticker}")
            compare_backtest_models(
                args.ticker, args.start, args.end,
                window_size=args.window, step_size=args.step, 
                prediction_days=args.predict, parallel=args.parallel,
                output_dir=output_dir
            )
    
    logger.info("Backtesting completed!")

if __name__ == "__main__":
    main()
