import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import importlib

def check_imports():
    """Check if all required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'streamlit',
        'yfinance', 'scikit-learn', 'scipy', 'requests', 'nltk', 'textblob'
    ]
    
    optional_packages = ['tensorflow', 'newspaper3k']
    
    print("Checking required packages...")
    all_required_ok = True
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed - please install with: pip install {package}")
            all_required_ok = False
    
    print("\nChecking optional packages...")
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"⚠️ {package} is NOT installed - some features may be disabled")
    
    return all_required_ok

def check_data_access():
    """Check if we can access stock market data"""
    print("\nTesting data access...")
    
    # First try with yfinance
    try:
        import yfinance as yf
        print("Testing Yahoo Finance API...")
        data = yf.download('AAPL', period='1d')
        if data.empty:
            print("⚠️ Could not retrieve data from Yahoo Finance - trying alternative sources")
        else:
            print(f"✅ Successfully retrieved data from Yahoo Finance")
            return True
    except Exception as e:
        print(f"⚠️ Error accessing Yahoo Finance: {str(e)} - trying alternative sources")
    
    # Check Financial Modeling Prep API key
    fmp_key = os.environ.get('FMP_API_KEY')
    if fmp_key:
        print(f"✅ Financial Modeling Prep API key found in environment variables")
        
        # Test Financial Modeling Prep API if key is available
        try:
            import requests
            print("Testing Financial Modeling Prep API...")
            url = f'https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={fmp_key}'
            r = requests.get(url)
            data = r.json()
            
            if isinstance(data, list) and len(data) > 0:
                print(f"✅ Successfully retrieved data from Financial Modeling Prep")
                return True
            else:
                print(f"⚠️ Financial Modeling Prep API returned unexpected response")
        except Exception as e:
            print(f"⚠️ Error testing Financial Modeling Prep API: {str(e)}")
    else:
        print("⚠️ No Financial Modeling Prep API key found. Set FMP_API_KEY environment variable for fallback data source.")
    
    # Check Alpha Vantage API key
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if api_key:
        print(f"✅ Alpha Vantage API key found in environment variables")
        
        # Test Alpha Vantage API if key is available
        try:
            import requests
            print("Testing Alpha Vantage API...")
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&outputsize=compact&apikey={api_key}'
            r = requests.get(url)
            data = r.json()
            
            if 'Time Series (Daily)' in data:
                print(f"✅ Successfully retrieved data from Alpha Vantage")
                return True
            elif 'Note' in data and 'API call frequency' in data['Note']:
                print(f"⚠️ Alpha Vantage API rate limit hit: {data['Note']}")
            else:
                print(f"⚠️ Alpha Vantage API returned unexpected response")
        except Exception as e:
            print(f"⚠️ Error testing Alpha Vantage API: {str(e)}")
    else:
        print("⚠️ No Alpha Vantage API key found. Set ALPHA_VANTAGE_API_KEY environment variable for fallback data source.")
    
    # Try with our multi-source function
    try:
        from utils import get_stock_data
        print("Testing multi-source data retrieval...")
        data = get_stock_data('AAPL', start='2023-01-01', end='2023-01-31')
        if data.empty:
            print("❌ Could not retrieve data from any source - check your internet connection")
            return False
        else:
            source = "alternative source" if data.shape[0] > 0 else "unknown source"
            print(f"✅ Successfully retrieved data from {source}")
            return True
    except Exception as e:
        print(f"❌ Error accessing all data sources: {str(e)}")
        return False

def check_directories():
    """Check if required directories exist, create them if they don't"""
    print("\nChecking required directories...")
    required_dirs = ['results', 'results/portfolio', 'results/risk']
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"✅ Created directory: {directory}")
            except Exception as e:
                print(f"❌ Could not create directory {directory}: {str(e)}")
                return False
        else:
            print(f"✅ Directory exists: {directory}")
    
    return True

def check_modules():
    """Check if our modules can be imported"""
    print("\nChecking modules...")
    modules = [
        'utils', 'stock_prediction', 'advanced_model', 
        'ensemble_model', 'sentiment_analysis', 'backtesting'
    ]
    
    all_ok = True
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✅ Module {module} is accessible")
        except Exception as e:
            print(f"❌ Could not import module {module}: {str(e)}")
            all_ok = False
    
    return all_ok

def run_full_check():
    """Run all diagnostic checks"""
    print("Starting diagnostic checks...\n")
    
    packages_ok = check_imports()
    data_ok = check_data_access()
    dirs_ok = check_directories()
    modules_ok = check_modules()
    
    print("\nDiagnostic summary:")
    print("=" * 30)
    print(f"Required packages: {'✅ OK' if packages_ok else '❌ Issues found'}")
    print(f"Data access: {'✅ OK' if data_ok else '❌ Issues found'}")
    print(f"Directories: {'✅ OK' if dirs_ok else '❌ Issues found'}")
    print(f"Module imports: {'✅ OK' if modules_ok else '❌ Issues found'}")
    
    if all([packages_ok, data_ok, dirs_ok, modules_ok]):
        print("\nAll checks passed! The application should work correctly.")
        print("To start the app, run: streamlit run app.py")
    else:
        print("\nIssues were found that may prevent the application from working correctly.")
        print("Please address the issues marked with ❌ above.")

if __name__ == "__main__":
    run_full_check()
