import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import logging
from functools import wraps
import time
import random
import pickle
from pathlib import Path
import requests
import pandas_datareader as pdr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('utils')

# Create cache directory
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def timer_decorator(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def exponential_backoff(attempt, base_delay=1, max_delay=60, jitter=True):
    """Calculate exponential backoff time with optional jitter"""
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter:
        delay = delay * (0.5 + random.random())
    return delay

def download_with_retry(ticker, start_date, end_date, max_retries=3, backoff_factor=2, initial_wait=1, **kwargs):
    """
    Download stock data with exponential backoff retry
    
    Parameters:
    - ticker: Stock symbol or list of symbols
    - start_date: Start date 
    - end_date: End date
    - max_retries: Maximum number of retry attempts
    - backoff_factor: Multiplier for the wait time between retries
    - initial_wait: Initial wait time in seconds
    
    Returns:
    - DataFrame with stock data
    """
    attempt = 0
    wait_time = initial_wait
    
    while attempt < max_retries:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, **kwargs)
            if not data.empty:
                return data
            logger.warning(f"Empty data returned for {ticker}. Retrying...")
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
        
        # Wait before retrying with exponential backoff
        time.sleep(wait_time + random.uniform(0, 1))  # Add jitter
        wait_time *= backoff_factor
        attempt += 1
    
    logger.error(f"Failed to download data for {ticker} after {max_retries} attempts")
    return pd.DataFrame()

def get_stock_data(ticker, start=None, end=None, auto_adjust=True, max_retries=5, cache_data=True):
    """
    Download stock data for a given ticker and date range with enhanced rate limit handling
    
    Parameters:
    - ticker: Stock symbol or list of symbols
    - start: Start date (string or datetime)
    - end: End date (string or datetime)
    - auto_adjust: Whether to adjust OHLC automatically
    - max_retries: Maximum number of retry attempts
    - cache_data: Whether to cache downloaded data
    
    Returns:
    - DataFrame with stock data
    """
    # Convert ticker to list if it's a string
    tickers = [ticker] if isinstance(ticker, str) else ticker
    
    # Generate cache key
    cache_key = f"{'-'.join(tickers)}_{start}_{end}_{auto_adjust}"
    cache_file = CACHE_DIR / f"{cache_key.replace(':', '_').replace('/', '_')}.pkl"
    
    # Try to load from cache first if caching is enabled
    if cache_data and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                logger.info(f"Loaded data for {tickers} from cache")
                return cached_data
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
    
    # Create a session for better connection management
    session = requests.Session()
    
    logger.info(f"Getting stock data for {tickers} from {start} to {end}")
    
    # First try Yahoo Finance
    for attempt in range(max_retries):
        try:
            if isinstance(ticker, str):
                # Single ticker
                data = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust, session=session)
            else:
                # Multiple tickers
                data = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust, group_by='column', session=session)
            
            if not data.empty:
                # If successful, save to cache if enabled
                if cache_data:
                    try:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(data, f)
                            logger.info(f"Cached data for {tickers}")
                    except Exception as e:
                        logger.warning(f"Failed to cache data: {e}")
                        
                return data
            
            logger.warning(f"Empty data returned for {tickers} from Yahoo Finance. Retrying...")
            delay = exponential_backoff(attempt)
            logger.info(f"Waiting {delay:.2f} seconds before retry {attempt+1}/{max_retries}")
            time.sleep(delay)
                
        except Exception as e:
            if "Rate limit" in str(e) or "Too Many Requests" in str(e):
                # Rate limit error - use exponential backoff
                delay = exponential_backoff(attempt)
                logger.warning(f"Rate limit hit for {tickers}. Waiting {delay:.2f} seconds before retry {attempt+1}/{max_retries}")
                time.sleep(delay)
            else:
                logger.warning(f"Error downloading data from Yahoo Finance for {tickers}: {e}")
                break  # Break out of retry loop and try Alpha Vantage
    
    # If Yahoo Finance failed, try Financial Modeling Prep
    logger.info(f"Trying Financial Modeling Prep for {tickers}")
    try:
        fmp_data = get_data_fmp(ticker, start, end, auto_adjust)
        if fmp_data is not None and not fmp_data.empty:
            logger.info(f"Successfully retrieved data from Financial Modeling Prep for {tickers}")
            
            # Cache the data if enabled
            if cache_data:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(fmp_data, f)
                        logger.info(f"Cached Financial Modeling Prep data for {tickers}")
                except Exception as e:
                    logger.warning(f"Failed to cache Financial Modeling Prep data: {e}")
            
            return fmp_data
    except Exception as e:
        logger.warning(f"Failed to retrieve data from Financial Modeling Prep: {str(e)}")
    
    # If Financial Modeling Prep failed, try Alpha Vantage
    logger.info(f"Trying Alpha Vantage for {tickers}")
    try:
        alpha_data = get_data_av(ticker, start, end, auto_adjust)
        if alpha_data is not None and not alpha_data.empty:
            logger.info(f"Successfully retrieved data from Alpha Vantage for {tickers}")
            
            # Cache the data if enabled
            if cache_data:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(alpha_data, f)
                        logger.info(f"Cached Alpha Vantage data for {tickers}")
                except Exception as e:
                    logger.warning(f"Failed to cache Alpha Vantage data: {e}")
            
            return alpha_data
    except Exception as e:
        logger.warning(f"Failed to retrieve data from Alpha Vantage: {str(e)}")
    
    # If Alpha Vantage failed, try Marketstack
    logger.info(f"Trying Marketstack for {tickers}")
    try:
        marketstack_data = get_data_marketstack(ticker, start, end, auto_adjust)
        if marketstack_data is not None and not marketstack_data.empty:
            logger.info(f"Successfully retrieved data from Marketstack for {tickers}")
            
            # Cache the data if enabled
            if cache_data:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(marketstack_data, f)
                        logger.info(f"Cached Marketstack data for {tickers}")
                except Exception as e:
                    logger.warning(f"Failed to cache Marketstack data: {e}")
            
            return marketstack_data
    except Exception as e:
        logger.warning(f"Failed to retrieve data from Marketstack: {str(e)}")
    
    # If Marketstack also fails, try pandas-datareader as a last resort
    logger.info(f"Trying pandas-datareader for {tickers}")
    try:
        pdr_data = get_data_pandas_datareader(ticker, start, end, auto_adjust)
        if pdr_data is not None and not pdr_data.empty:
            logger.info(f"Successfully retrieved data from pandas-datareader for {tickers}")
            
            # Cache the data if enabled
            if cache_data:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(pdr_data, f)
                        logger.info(f"Cached pandas-datareader data for {tickers}")
                except Exception as e:
                    logger.warning(f"Failed to cache pandas-datareader data: {e}")
            
            return pdr_data
    except Exception as e:
        logger.warning(f"Failed to retrieve data from pandas-datareader: {str(e)}")
    
    logger.error(f"Failed to retrieve data for {tickers} from all sources")
    return pd.DataFrame()

@timer_decorator
def get_data_av(ticker, start, end, auto_adjust=True):
    """
    Get data from Alpha Vantage API
    
    Parameters:
    - ticker: Stock symbol or list of symbols
    - start: Start date (string or datetime)
    - end: End date (string or datetime)
    - auto_adjust: Whether to use adjusted prices
    
    Returns:
    - DataFrame with stock data
    
    Note:
    - Requires Alpha Vantage API key set as environment variable ALPHA_VANTAGE_API_KEY
    - Free tier has a limit of 5 API calls per minute and 500 calls per day
    """
    # Get API key from environment variable
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    
    if not api_key:
        logger.warning("Alpha Vantage API key not found in environment variables. "
                      "Set ALPHA_VANTAGE_API_KEY environment variable.")
        return None
    
    # Handle list of tickers
    if isinstance(ticker, list):
        # For multiple tickers, we need to fetch them one by one
        all_data = []
        for t in ticker:
            data = get_single_ticker_av(t, start, end, api_key, auto_adjust)
            if data is not None:
                all_data.append(data)
            # Add delay to respect rate limits (free tier: 5 calls per minute)
            time.sleep(12.1)  # ~5 calls per minute
        
        if not all_data:
            return None
            
        # Combine data frames for multiple tickers
        if len(all_data) == 1:
            return all_data[0]
        else:
            # Create hierarchical columns for multiple tickers
            combined = pd.concat(all_data, axis=1, keys=ticker)
            return combined
    else:
        return get_single_ticker_av(ticker, start, end, api_key, auto_adjust)

@timer_decorator
def get_single_ticker_av(ticker, start, end, api_key, auto_adjust=True):
    """
    Get data for a single ticker from Alpha Vantage
    
    Parameters:
    - ticker: Stock symbol
    - start: Start date (string or datetime)
    - end: End date (string or datetime)
    - api_key: Alpha Vantage API key
    - auto_adjust: Whether to use adjusted prices
    
    Returns:
    - DataFrame with stock data
    """
    # Determine which function to call based on auto_adjust
    function = 'TIME_SERIES_DAILY_ADJUSTED' if auto_adjust else 'TIME_SERIES_DAILY'
    
    # Alpha Vantage API URL
    url = f'https://www.alphavantage.co/query?function={function}&symbol={ticker}&outputsize=full&apikey={api_key}'
    
    # Make API request
    logger.info(f"Requesting data from Alpha Vantage for {ticker}")
    r = requests.get(url)
    
    # Check if request was successful
    if r.status_code != 200:
        logger.warning(f"Alpha Vantage API request failed with status code: {r.status_code}")
        return None
    
    # Parse JSON response
    try:
        data = r.json()
    except Exception as e:
        logger.warning(f"Failed to parse Alpha Vantage JSON response: {str(e)}")
        return None
    
    # Check for error messages
    if 'Error Message' in data:
        logger.warning(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
        return None
    
    if 'Note' in data and 'API call frequency' in data['Note']:
        logger.warning(f"Alpha Vantage API rate limit hit: {data['Note']}")
        return None
    
    # Extract time series data
    time_series_key = 'Time Series (Daily)'
    if time_series_key not in data:
        logger.warning(f"No time series data found in Alpha Vantage response for {ticker}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()
    
    # Filter by date
    if start:
        start_date = pd.Timestamp(start)
        df = df[df.index >= start_date]
    if end:
        end_date = pd.Timestamp(end)
        df = df[df.index <= end_date]
    
    # Rename columns to match Yahoo Finance format
    if auto_adjust:
        columns = {
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '5. adjusted close': 'Close',
            '6. volume': 'Volume',
            '5. adjusted close': 'Adj Close'  # Also map to Adj Close for consistency
        }
    else:
        columns = {
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }
    
    # Rename and select columns
    df = df.rename(columns=columns)
    df = df[[col for col in columns.values() if col in df.columns]]
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    # Add Adj Close if it doesn't exist and auto_adjust is True
    if 'Adj Close' not in df.columns and auto_adjust:
        df['Adj Close'] = df['Close']
    
    logger.info(f"Successfully retrieved Alpha Vantage data for {ticker}: {len(df)} rows")
    return df

@timer_decorator
def get_data_marketstack(ticker, start, end, auto_adjust=True):
    """
    Get data from Marketstack API
    
    Parameters:
    - ticker: Stock symbol or list of symbols
    - start: Start date (string or datetime)
    - end: End date (string or datetime)
    - auto_adjust: Whether to use adjusted prices
    
    Returns:
    - DataFrame with stock data
    
    Note:
    - Requires Marketstack API key set as environment variable MARKETSTACK_API_KEY
    - Free tier has a limit of 100 requests per month
    """
    # Get API key from environment variable
    api_key = os.environ.get('MARKETSTACK_API_KEY')
    
    if not api_key:
        logger.warning("Marketstack API key not found in environment variables. "
                      "Set MARKETSTACK_API_KEY environment variable.")
        return None
    
    # Handle list of tickers
    if isinstance(ticker, list):
        # For multiple tickers, we need to fetch them one by one
        all_data = []
        for t in ticker:
            data = get_single_ticker_marketstack(t, start, end, api_key)
            if data is not None:
                all_data.append(data)
            # Add delay to respect rate limits
            time.sleep(1)
        
        if not all_data:
            return None
            
        # Combine data frames for multiple tickers
        if len(all_data) == 1:
            return all_data[0]
        else:
            # Create hierarchical columns for multiple tickers
            combined = pd.concat(all_data, axis=1, keys=ticker)
            return combined
    else:
        return get_single_ticker_marketstack(ticker, start, end, api_key)

def get_single_ticker_marketstack(ticker, start, end, api_key):
    """
    Get data for a single ticker from Marketstack
    
    Parameters:
    - ticker: Stock symbol
    - start: Start date (string or datetime)
    - end: End date (string or datetime)
    - api_key: Marketstack API key
    
    Returns:
    - DataFrame with stock data
    """
    # Format dates
    if isinstance(start, datetime.datetime):
        start = start.strftime('%Y-%m-%d')
    if isinstance(end, datetime.datetime):
        end = end.strftime('%Y-%m-%d')
    
    # Marketstack API URL
    base_url = 'http://api.marketstack.com/v1/eod'
    
    # Build parameters
    params = {
        'access_key': api_key,
        'symbols': ticker,
        'date_from': start,
        'date_to': end,
        'limit': 1000  # Maximum per request
    }
    
    # Make API request
    logger.info(f"Requesting data from Marketstack for {ticker}")
    
    all_data = []
    offset = 0
    
    # Paginate through all results
    while True:
        params['offset'] = offset
        try:
            r = requests.get(base_url, params=params)
            
            # Check if request was successful
            if r.status_code != 200:
                logger.warning(f"Marketstack API request failed with status code: {r.status_code}")
                break
            
            # Parse JSON response
            data = r.json()
            
            # Check for error messages
            if 'error' in data:
                logger.warning(f"Marketstack error for {ticker}: {data['error']['message']}")
                break
            
            # Extract data
            if 'data' not in data or len(data['data']) == 0:
                if offset == 0:
                    logger.warning(f"No data found in Marketstack response for {ticker}")
                break
            
            all_data.extend(data['data'])
            
            # Check if we've received all data
            if len(data['data']) < params['limit']:
                break
                
            # Update offset for next page
            offset += len(data['data'])
            
        except Exception as e:
            logger.warning(f"Failed to retrieve data from Marketstack: {str(e)}")
            break
    
    if not all_data:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Set the date as the index
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.sort_index()
    
    # Rename columns to match Yahoo Finance format
    columns = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'adj_close': 'Adj Close'
    }
    
    # Rename columns
    df = df.rename(columns={k: v for k, v in columns.items() if k in df.columns})
    
    # Add Adj Close if it's not present (Marketstack may not provide adjusted prices in free tier)
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    
    # Select only needed columns
    needed_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    df = df[[col for col in needed_columns if col in df.columns]]
    
    logger.info(f"Successfully retrieved Marketstack data for {ticker}: {len(df)} rows")
    return df

@timer_decorator
def get_data_pandas_datareader(ticker, start, end, auto_adjust=True):
    """Get data using pandas_datareader"""
    try:
        # Try stooq as data source
        data = pdr.data.DataReader(ticker, 'stooq', start=start, end=end)
        
        # Stooq data has columns in lowercase, convert to match Yahoo Finance format
        data.columns = [col.capitalize() for col in data.columns]
        
        # Add Adj Close column if not there and auto_adjust is True
        if 'Adj Close' not in data.columns and auto_adjust:
            data['Adj Close'] = data['Close']
        
        return data
    except Exception as e:
        logger.warning(f"Pandas-datareader error: {str(e)}")
        raise

def calculate_technical_indicators(data):
    """
    Calculate various technical indicators for stock data
    
    Parameters:
    - data: DataFrame with OHLC stock data
    
    Returns:
    - DataFrame with added technical indicators
    """
    df = data.copy()
    
    # Moving averages
    for window in [5, 10, 20, 50, 200]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
    
    # Exponential moving averages
    for window in [12, 26]:
        df[f'EMA{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 0.000001)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def calculate_metrics(y_true, y_pred):
    """
    Calculate performance metrics for regression models
    
    Parameters:
    - y_true: Actual values
    - y_pred: Predicted values
    
    Returns:
    - Dictionary with metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

@timer_decorator
def get_data_fmp(ticker, start, end, auto_adjust=True):
    """
    Get data from Financial Modeling Prep API
    
    Parameters:
    - ticker: Stock symbol or list of symbols
    - start: Start date (string or datetime)
    - end: End date (string or datetime)
    - auto_adjust: Whether to use adjusted prices
    
    Returns:
    - DataFrame with stock data
    
    Note:
    - Requires Financial Modeling Prep API key set as environment variable FMP_API_KEY
    - Free tier has limitations on endpoint access and request frequency
    """
    # Get API key from environment variable
    api_key = os.environ.get('FMP_API_KEY')
    
    if not api_key:
        logger.warning("Financial Modeling Prep API key not found in environment variables. "
                      "Set FMP_API_KEY environment variable.")
        return None
    
    # Handle list of tickers
    if isinstance(ticker, list):
        # For multiple tickers, we need to fetch them one by one
        all_data = []
        for t in ticker:
            data = get_single_ticker_fmp(t, start, end, api_key, auto_adjust)
            if data is not None:
                all_data.append(data)
            # Add delay to respect rate limits
            time.sleep(0.5)
        
        if not all_data:
            return None
            
        # Combine data frames for multiple tickers
        if len(all_data) == 1:
            return all_data[0]
        else:
            # Create hierarchical columns for multiple tickers
            combined = pd.concat(all_data, axis=1, keys=ticker)
            return combined
    else:
        return get_single_ticker_fmp(ticker, start, end, api_key, auto_adjust)

def get_single_ticker_fmp(ticker, start, end, api_key, auto_adjust=True):
    """
    Get data for a single ticker from Financial Modeling Prep
    
    Parameters:
    - ticker: Stock symbol
    - start: Start date (string or datetime)
    - end: End date (string or datetime)
    - api_key: Financial Modeling Prep API key
    - auto_adjust: Whether to use adjusted prices
    
    Returns:
    - DataFrame with stock data
    """
    # Format dates for API request
    if isinstance(start, datetime.datetime):
        start = start.strftime('%Y-%m-%d')
    if isinstance(end, datetime.datetime):
        end = end.strftime('%Y-%m-%d')
    
    # Financial Modeling Prep API URL for historical data
    # Choose appropriate endpoint based on auto_adjust
    endpoint = 'historical-price-full/daily-adjusted' if auto_adjust else 'historical-price-full'
    base_url = f'https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}'
    
    # Build parameters
    params = {
        'apikey': api_key,
        'from': start,
        'to': end,
    }
    
    # Make API request
    logger.info(f"Requesting data from Financial Modeling Prep for {ticker}")
    
    try:
        r = requests.get(base_url, params=params)
        
        # Check if request was successful
        if r.status_code != 200:
            logger.warning(f"Financial Modeling Prep API request failed with status code: {r.status_code}")
            return None
        
        # Parse JSON response
        data = r.json()
        
        # Check for error messages
        if 'Error Message' in data:
            logger.warning(f"Financial Modeling Prep error for {ticker}: {data['Error Message']}")
            return None
            
        # Extract historical data
        if 'historical' not in data or len(data['historical']) == 0:
            logger.warning(f"No historical data found in Financial Modeling Prep response for {ticker}")
            return None
        
        # Convert to DataFrame
        historical_data = data['historical']
        df = pd.DataFrame(historical_data)
        
        # Set the date as the index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.sort_index()
        
        # Rename columns to match Yahoo Finance format
        columns = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adjClose': 'Adj Close'
        }
        
        # Rename columns
        df = df.rename(columns={k: v for k, v in columns.items() if k in df.columns})
        
        # If auto_adjust is True but 'Adj Close' is not available, use 'close'
        if 'Adj Close' not in df.columns and auto_adjust:
            if 'adjClose' in df.columns:
                df['Adj Close'] = df['adjClose']
            else:
                df['Adj Close'] = df['Close']
        
        # Select only needed columns
        needed_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        df = df[[col for col in needed_columns if col in df.columns]]
        
        logger.info(f"Successfully retrieved Financial Modeling Prep data for {ticker}: {len(df)} rows")
        return df
        
    except Exception as e:
        logger.warning(f"Failed to retrieve data from Financial Modeling Prep: {str(e)}")
        return None
