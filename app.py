import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load
import os
import sys
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# Set page config - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title='Stock Price Prediction Dashboard', layout='wide', page_icon="📈")

# Apply custom CSS styling after the set_page_config call
st.markdown("""
    <style>
    .css-18e3th9 {
        padding-top: 0;
        padding-bottom: 0;
    }
    .css-1d391kg {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
    }
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    .stTextInput, .stSelectbox, .stMultiselect {
        background-color: #2C2C2C;
        color: white;
    }
    .stDataFrame {
        background-color: #2C2C2C;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1E1E1E;
        border-radius: 4px 4px 0 0;
        color: white;
        padding: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    .stPlotlyChart {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 10px;
    }
    .css-145kmo2 {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our model scripts
import stock_prediction
import advanced_model
import ensemble_model
from utils import calculate_technical_indicators

try:
    import lstm_model
    has_tensorflow = True
except ImportError:
    has_tensorflow = False
    
try:
    import sentiment_analysis
    has_sentiment = True
except ImportError:
    has_sentiment = False

try:
    from portfolio_optimization import PortfolioOptimizer
    has_portfolio = True
except ImportError:
    has_portfolio = False

try:
    from backtesting import backtest_model, compare_backtest_models
    has_backtesting = True
except ImportError:
    has_backtesting = False

# Apply custom CSS if style.css exists
if os.path.exists('style.css'):
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_data
def load_data(ticker, start_date, end_date):
    """
    Load stock data for the given ticker and date range
    """
    try:
        from utils import get_stock_data
        data = get_stock_data(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for {ticker} from {start_date} to {end_date}")
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def get_technical_indicators(data):
    """
    Calculate technical indicators for visualization
    """
    try:
        if data.empty:
            return pd.DataFrame()
            
        # Check for sufficient data
        if len(data) < 200:
            st.warning(f"Only {len(data)} data points available. For reliable indicators, 200+ points recommended.")
            
        df = data.copy()
        # Calculate technical indicators
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate RSI - corrected implementation
        delta = df['Close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        down = down.abs()
        
        # Calculate first average
        avg_up = up.rolling(window=14).mean()
        avg_down = down.rolling(window=14).mean()
        
        # Calculate RS and RSI
        rs = avg_up / avg_down.replace(0, 0.001)  # Avoid division by zero
        df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
        
        # Calculate MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Line'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD_Line'] - df['MACD_Signal']
        
        # Calculate Bollinger Bands
        middle = df['Close'].rolling(window=20).mean()
        std_dev = df['Close'].rolling(window=20).std()
        
        df['BB_Middle'] = middle
        df['BB_Upper'] = middle + (2 * std_dev)
        df['BB_Lower'] = middle - (2 * std_dev)
        
        # Log indicator data availability (replaced debug with comments)
        non_na_counts = {
            'MA20': df['MA20'].notna().sum(),
            'MA50': df['MA50'].notna().sum(),
            'MA200': df['MA200'].notna().sum(),
            'BB_Middle': df['BB_Middle'].notna().sum(),
            'BB_Upper': df['BB_Upper'].notna().sum(),
            'BB_Lower': df['BB_Lower'].notna().sum()
        }
        
        # Removed st.debug() calls which were causing the error
        # For debugging purposes, you could uncomment these lines and use st.write() instead
        # st.write(f"Data points available: {len(df)}")
        # st.write(f"Non-NaN values in indicators: {non_na_counts}")
        
        return df
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return pd.DataFrame()

def plot_interactive_stock_chart(data, ticker):
    """
    Create an interactive stock chart with selectable indicators
    """
    # Create a DataFrame with technical indicators
    tech_data = get_technical_indicators(data)
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                      vertical_spacing=0.1, row_heights=[0.7, 0.3],
                      subplot_titles=(f'{ticker} Stock Price', 'Volume'))
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA20'], name='20-day MA', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA50'], name='50-day MA', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA200'], name='200-day MA', line=dict(color='red')), row=1, col=1)
    
    # Add volume bar chart
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='rgba(0, 0, 255, 0.3)'),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Analysis',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

@st.cache_data
def get_stock_list():
    """
    Returns a dictionary of popular stock tickers and their company names
    """
    return {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com, Inc.",
        "META": "Meta Platforms, Inc.",
        "TSLA": "Tesla, Inc.",
        "NVDA": "NVIDIA Corporation",
        "JPM": "JPMorgan Chase & Co.",
        "V": "Visa Inc.",
        "WMT": "Walmart Inc.",
        "PG": "Procter & Gamble Co.",
        "JNJ": "Johnson & Johnson",
        "MA": "Mastercard Inc.",
        "UNH": "UnitedHealth Group Inc.",
        "HD": "The Home Depot, Inc.",
        "BAC": "Bank of America Corp.",
        "XOM": "Exxon Mobil Corporation",
        "DIS": "The Walt Disney Company",
        "NFLX": "Netflix, Inc.",
        "INTC": "Intel Corporation"
    }

# Add navigation sidebar
st.sidebar.image('https://www.pngkey.com/png/detail/147-1472304_stock-trading-logo-png-transparent-stock-market-icon.png', width=100)
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Stock Analysis', 'Technical Analysis', 'Portfolio Optimization', 'Multi-Stock Comparison', 'About'])

if page == 'Stock Analysis':
    # Main application
    st.title('Stock Price Analysis & Prediction')
    
    # Stock selection
    stock_list = get_stock_list()
    ticker = st.sidebar.selectbox('Select Stock', list(stock_list.keys()), format_func=lambda x: f"{x} - {stock_list[x]}")
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input('Start Date', datetime.date(2020, 1, 1))
    with col2:
        end_date = st.date_input('End Date', datetime.date.today())
    
    # Load data button with clear error handling
    if st.sidebar.button('Load Data', key='load_data_button'):
        with st.spinner('Loading data...'):
            data = load_data(ticker, start_date, end_date)
            if not data.empty:
                st.session_state['data'] = data
                st.session_state['ticker'] = ticker
                st.success(f"Successfully loaded data for {ticker}")
            else:
                st.error(f"Failed to load data for {ticker}")
    
    # Main content - display data and predictions
    if 'data' in st.session_state:
        data = st.session_state['data']
        ticker = st.session_state['ticker']
        
        # Display interactive stock chart
        st.header(f'{ticker} Stock Price History')
        try:
            fig = plot_interactive_stock_chart(data, ticker)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
        
        # Display basic statistics with improved formatting
        st.subheader('Summary Statistics')
        
        try:
            # Create a better statistics table
            stats_df = data.describe().T
            
            # Only keep certain statistics
            stats_df = stats_df[['count', 'mean', 'std', 'min', 'max']]
            
            # Rename columns
            stats_df.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max']
            
            # Round to 2 decimal places
            stats_df = stats_df.round(2)
            
            st.dataframe(stats_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying statistics: {str(e)}")
        
elif page == 'Technical Analysis':
    st.title('Technical Analysis Dashboard')
    
    # Stock selection
    stock_list = get_stock_list()
    ticker = st.sidebar.selectbox('Select Stock', list(stock_list.keys()), format_func=lambda x: f"{x} - {stock_list[x]}")
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input('Start Date', datetime.date(2020, 1, 1))
    with col2:
        # Ensure end date is not in the future
        today = datetime.date.today()
        end_date = st.date_input('End Date', today, max_value=today)
    
    # Indicators selection
    indicators = st.sidebar.multiselect('Select Indicators', 
                                       ['Moving Averages', 'RSI', 'MACD', 'Bollinger Bands'],
                                       default=['Moving Averages', 'RSI'])
    
    # Make the Analyze button more prominent
    if st.sidebar.button('Analyze', key='tech_analyze_button', use_container_width=True):
        if not indicators:
            st.warning("Please select at least one indicator to display")
        else:
            with st.spinner('Loading and analyzing data...'):
                try:
                    data = load_data(ticker, start_date, end_date)
                    if data.empty:
                        st.error(f"No data found for {ticker}")
                    else:
                        # Calculate technical indicators
                        tech_data = get_technical_indicators(data)
                        
                        if tech_data.empty:
                            st.error("Failed to calculate technical indicators")
                        else:
                            # Create tabs for each selected indicator
                            tabs = st.tabs(indicators)
                            
                            for i, indicator in enumerate(indicators):
                                if indicator == 'Moving Averages':
                                    with tabs[i]:
                                        fig = go.Figure()
                                        
                                        # Add price
                                        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='black')))
                                        
                                        # Check if there's actual data in the MA columns (not just all NaN)
                                        ma_data_available = False
                                        
                                        if 'MA20' in tech_data.columns and tech_data['MA20'].notna().any():
                                            fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA20'], name='20-day MA', line=dict(color='blue')))
                                            ma_data_available = True
                                            
                                        if 'MA50' in tech_data.columns and tech_data['MA50'].notna().any():
                                            fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA50'], name='50-day MA', line=dict(color='orange')))
                                            ma_data_available = True
                                            
                                        if 'MA200' in tech_data.columns and tech_data['MA200'].notna().any():
                                            fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA200'], name='200-day MA', line=dict(color='red')))
                                            ma_data_available = True
                                        
                                        if not ma_data_available:
                                            st.warning("No moving average data available. Try selecting a longer date range.")
                                            
                                        fig.update_layout(
                                            title=f'{ticker} Moving Averages',
                                            xaxis_title='Date',
                                            yaxis_title='Price (USD)',
                                            height=500
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Show interpretation only if we have data
                                        if ma_data_available:
                                            st.info('''
                                                **Moving Averages Interpretation**:
                                                - When the price crosses above the moving average: Potential bullish signal
                                                - When the price crosses below the moving average: Potential bearish signal
                                                - When shorter-term MA crosses above longer-term MA: Golden Cross (bullish)
                                                - When shorter-term MA crosses below longer-term MA: Death Cross (bearish)
                                            ''')
                                
                                elif indicator == 'RSI':
                                    with tabs[i]:
                                        if 'RSI' in tech_data.columns and not tech_data['RSI'].isna().all():
                                            fig = go.Figure()
                                            
                                            # Add RSI line
                                            fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['RSI'], name='RSI', line=dict(color='purple')))
                                            
                                            # Add overbought/oversold lines
                                            fig.add_shape(type="line", x0=tech_data.index[0], y0=70, x1=tech_data.index[-1], y1=70,
                                                        line=dict(color="red", width=2, dash="dash"))
                                            fig.add_shape(type="line", x0=tech_data.index[0], y0=30, x1=tech_data.index[-1], y1=30,
                                                        line=dict(color="green", width=2, dash="dash"))
                                            
                                            fig.update_layout(
                                                title="Relative Strength Index (RSI)",
                                                xaxis_title="Date",
                                                yaxis_title="RSI Value",
                                                yaxis=dict(range=[0, 100]),
                                                height=500
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            st.markdown("""
                                            ### RSI Interpretation
                                            - RSI > 70: Overbought condition, potential sell signal
                                            - RSI < 30: Oversold condition, potential buy signal
                                            - RSI trend: Can confirm or contradict price trends
                                            """)
                                        else:
                                            st.error("RSI data is not available. This may be due to insufficient historical data for calculation.")
                                
                                elif indicator == 'MACD':
                                    with tabs[i]:
                                        if all(col in tech_data.columns for col in ['MACD_Line', 'MACD_Signal']):
                                            # Create figure with secondary y-axis
                                            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                                                vertical_spacing=0.1, row_heights=[0.7, 0.3])
                                            
                                            # Add price to top plot
                                            fig.add_trace(
                                                go.Scatter(x=data.index, y=data['Close'], name='Close Price'),
                                                row=1, col=1
                                            )
                                            
                                            # Add MACD to bottom plot
                                            fig.add_trace(
                                                go.Scatter(x=tech_data.index, y=tech_data['MACD_Line'], name='MACD Line', line=dict(color='blue')),
                                                row=2, col=1
                                            )
                                            fig.add_trace(
                                                go.Scatter(x=tech_data.index, y=tech_data['MACD_Signal'], name='Signal Line', line=dict(color='red')),
                                                row=2, col=1
                                            )
                                            
                                            # Add MACD histogram
                                            if 'MACD_Histogram' in tech_data.columns:
                                                colors = ['green' if val >= 0 else 'red' for val in tech_data['MACD_Histogram']]
                                                fig.add_trace(
                                                    go.Bar(x=tech_data.index, y=tech_data['MACD_Histogram'], name='Histogram', marker_color=colors),
                                                    row=2, col=1
                                                )
                                            
                                            fig.update_layout(
                                                title=f'{ticker} MACD',
                                                xaxis_title='Date',
                                                height=600
                                            )
                                            
                                            fig.update_yaxes(title='Price (USD)', row=1, col=1)
                                            fig.update_yaxes(title='MACD', row=2, col=1)
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            st.info('''
                                                **MACD Interpretation**:
                                                - When MACD crosses above Signal Line: Bullish signal
                                                - When MACD crosses below Signal Line: Bearish signal
                                                - Histogram shows momentum (green=positive, red=negative)
                                                - Divergence between MACD and price can indicate potential reversals
                                            ''')
                                        else:
                                            st.error("MACD data not available")
                                
                                elif indicator == 'Bollinger Bands':
                                    with tabs[i]:
                                        # Check if BB data exists and has non-NaN values
                                        bb_cols = ['BB_Upper', 'BB_Middle', 'BB_Lower']
                                        bb_data_available = all(col in tech_data.columns for col in bb_cols) and all(tech_data[col].notna().any() for col in bb_cols)
                                        
                                        if bb_data_available:
                                            fig = go.Figure()
                                            
                                            # Add price
                                            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='black')))
                                            
                                            # Add Bollinger Bands
                                            fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['BB_Upper'], name='Upper Band', line=dict(color='red', dash='dash')))
                                            fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['BB_Middle'], name='Middle Band (SMA20)', line=dict(color='orange')))
                                            fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['BB_Lower'], name='Lower Band', line=dict(color='green', dash='dash')))
                                            
                                            # Add fill between bands
                                            fig.add_trace(go.Scatter(
                                                x=tech_data.index.tolist() + tech_data.index.tolist()[::-1],
                                                y=tech_data['BB_Upper'].tolist() + tech_data['BB_Lower'].tolist()[::-1],
                                                fill='toself',
                                                fillcolor='rgba(0,176,246,0.2)',
                                                line=dict(color='rgba(255,255,255,0)'),
                                                name='Band Range'
                                            ))
                                            
                                            fig.update_layout(
                                                title=f'{ticker} Bollinger Bands',
                                                xaxis_title='Date',
                                                yaxis_title='Price (USD)',
                                                height=500
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            st.info('''
                                                **Bollinger Bands Interpretation**:
                                                - Price touching Upper Band: Potentially overbought
                                                - Price touching Lower Band: Potentially oversold
                                                - Narrowing bands: Lower volatility (potential breakout coming)
                                                - Widening bands: Increasing volatility
                                                - Price moving outside bands: Strong trend continuation or potential reversal
                                            ''')
                                        else:
                                            st.warning("Bollinger Bands data not available. Try selecting a longer date range (at least 20 days).")
                            
                except Exception as e:
                    st.error(f"Error performing technical analysis: {str(e)}")
                    st.exception(e)
    else:
        # Display instructions when first loading the page
        st.info("👈 Please select a stock, date range, and indicators, then click 'Analyze' to view technical analysis.")
        
        # Show sample image of what to expect
        st.image("https://www.investopedia.com/thmb/z8l3UdCQu_mFE9i6WzGYFaJXYPM=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/technical-analysis-d5ba96f1109c4314821fce4b6900373d.png", 
                caption="Sample Technical Analysis Chart", 
                use_column_width=True)

elif page == 'Portfolio Optimization':
    # Clear any previous portfolio results to prevent them from showing in other tabs
    if 'portfolio_results' in st.session_state:
        del st.session_state['portfolio_results']
        
    st.title('Portfolio Optimization')
    
    # Stock selection - allow multiple stocks
    stock_list = get_stock_list()
    tickers = st.sidebar.multiselect(
        'Select Stocks for Portfolio', 
        list(stock_list.keys()), 
        default=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
        format_func=lambda x: f"{x} - {stock_list[x]}"
    )
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input('Start Date', datetime.date(2018, 1, 1), key='po_start_date')
    with col2:
        end_date = st.date_input('End Date', datetime.date.today(), key='po_end_date')
    
    # Optimization parameters
    optimization_method = st.sidebar.selectbox(
        'Optimization Method',
        ['Maximum Sharpe Ratio', 'Minimum Volatility', 'Efficient Return', 'Efficient Risk'],
        index=0
    )
    
    # If efficient return or risk is selected, show slider for target
    if optimization_method == 'Efficient Return':
        target_return = st.sidebar.slider('Target Annual Return (%)', min_value=1, max_value=50, value=15) / 100
    elif optimization_method == 'Efficient Risk':
        target_volatility = st.sidebar.slider('Target Annual Volatility (%)', min_value=1, max_value=50, value=20) / 100
    
    # Add risk-free rate input
    risk_free_rate = st.sidebar.slider('Risk-Free Rate (%)', min_value=0.0, max_value=5.0, value=0.02, step=0.01) / 100
    
    # Optimize button
    if st.sidebar.button('Optimize Portfolio', key='optimize_button'):
        if len(tickers) < 2:
            st.error("Please select at least 2 stocks for the portfolio")
        else:
            with st.spinner('Optimizing portfolio...'):
                try:
                    # Import here to avoid circular imports
                    from portfolio_optimization import PortfolioOptimizer
                    
                    # Initialize optimizer
                    optimizer = PortfolioOptimizer(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                    
                    # Load data
                    data = optimizer.load_data()
                    
                    if data is None or data.empty:
                        st.error("Failed to load stock data. Please check your internet connection and try again.")
                    else:
                        st.success(f"Successfully loaded data for {len(tickers)} stocks")
                        
                        # Optimize portfolio
                        if optimization_method == 'Maximum Sharpe Ratio':
                            result = optimizer.optimize_portfolio(risk_free_rate=risk_free_rate, method='sharpe')
                        elif optimization_method == 'Minimum Volatility':
                            result = optimizer.optimize_portfolio(risk_free_rate=risk_free_rate, method='min_volatility')
                        elif optimization_method == 'Efficient Return':
                            result = optimizer.optimize_portfolio(risk_free_rate=risk_free_rate, target_return=target_return, method='efficient_return')
                        elif optimization_method == 'Efficient Risk':
                            result = optimizer.optimize_portfolio(risk_free_rate=risk_free_rate, target_volatility=target_volatility, method='efficient_risk')
                            
                        # Display results
                        st.header('Optimized Portfolio Results')
                        
                        # Create metrics display
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Expected Annual Return", f"{result['Return']*100:.2f}%")
                        with col2:
                            st.metric("Annual Volatility", f"{result['Volatility']*100:.2f}%")
                        with col3:
                            st.metric("Sharpe Ratio", f"{result['Sharpe']:.2f}")
                            
                        # Display asset allocation
                        st.subheader('Asset Allocation')
                        
                        # Create a DataFrame for the weights
                        weights_df = pd.DataFrame({
                            'Stock': result['Weights'].index,
                            'Allocation (%)': (result['Weights'].values * 100).round(2)
                        })
                        
                        # Display as a bar chart
                        st.bar_chart(weights_df.set_index('Stock'))
                        
                        # Display as a table
                        st.dataframe(weights_df, use_container_width=True)
                        
                        # Option to run backtest
                        if st.button('Backtest this Portfolio', key='backtest_button'):
                            with st.spinner('Running backtest...'):
                                backtest_results = optimizer.backtest_portfolio(result['Weights'])
                                
                                # Display backtest results
                                st.header('Backtest Results')
                                
                                # Show metrics
                                metric_cols = st.columns(3)
                                with metric_cols[0]:
                                    st.metric("Total Return", f"{backtest_results['total_return']*100:.2f}%")
                                with metric_cols[1]:
                                    st.metric("Annual Return", f"{backtest_results['annual_return']*100:.2f}%")
                                with metric_cols[2]:
                                    st.metric("Sharpe Ratio", f"{backtest_results['sharpe']:.2f}")
                                    
                                metric_cols2 = st.columns(3)
                                with metric_cols2[0]:
                                    st.metric("Max Drawdown", f"{backtest_results['max_drawdown']*100:.2f}%")
                                with metric_cols2[1]:
                                    st.metric("Volatility", f"{backtest_results['volatility']*100:.2f}%")
                                with metric_cols2[2]:
                                    st.metric("Sortino Ratio", f"{backtest_results['sortino']:.2f}")
                                
                                # Show performance chart
                                st.subheader('Portfolio Performance')
                                st.image('results/portfolio/portfolio_backtest.png')
                                
                                # Show allocation pie chart
                                st.subheader('Final Portfolio Allocation')
                                st.image('results/portfolio/portfolio_allocation.png')
                                
                except Exception as e:
                    st.error(f"Error during portfolio optimization: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
elif page == 'Multi-Stock Comparison':
    st.title('Multi-Stock Comparison')
    
    # Stock selection - allow multiple stocks
    stock_list = get_stock_list()
    compare_tickers = st.sidebar.multiselect(
        'Select Stocks to Compare', 
        list(stock_list.keys()), 
        default=['AAPL', 'MSFT', 'GOOGL'],
        format_func=lambda x: f"{x} - {stock_list[x]}"
    )
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input('Start Date', datetime.date(2020, 1, 1), key='compare_start_date')
    with col2:
        end_date = st.date_input('End Date', datetime.date.today(), key='compare_end_date')
    
    # Comparison options
    comparison_type = st.sidebar.selectbox(
        'Comparison Type',
        ['Price', 'Normalized Price', 'Returns', 'Volatility'],
        index=1
    )
    
    # Compare button
    if st.sidebar.button('Compare Stocks', key='compare_button'):
        if len(compare_tickers) < 1:
            st.error("Please select at least one stock to compare")
        else:
            with st.spinner('Loading data for comparison...'):
                try:
                    # Load data for all selected tickers
                    from utils import get_stock_data
                    all_data = get_stock_data(compare_tickers, start=start_date, end=end_date)['Adj Close']
                except KeyError:
                    # If Adj Close is not available, fall back to Close prices
                    all_data = get_stock_data(compare_tickers, start=start_date, end=end_date)['Close']
                    st.info("Using 'Close' prices instead of 'Adj Close' for comparison")
                
                # Handle case of single ticker
                if len(compare_tickers) == 1:
                    all_data = pd.DataFrame(all_data, columns=compare_tickers)

                if all_data.empty:
                    st.error("No data found for the selected stocks and date range")
                else:
                    st.success(f"Successfully loaded data for {len(compare_tickers)} stocks")
                    
                    # Create the comparison chart based on selected type
                    st.subheader(f'{comparison_type} Comparison')
                    
                    if comparison_type == 'Price':
                        # Create price chart
                        fig = go.Figure()
                        
                        for ticker in compare_tickers:
                            fig.add_trace(go.Scatter(
                                x=all_data.index,
                                y=all_data[ticker],
                                name=ticker
                            ))
                        
                        fig.update_layout(
                            title=f'Stock Price Comparison',
                            xaxis_title='Date',
                            yaxis_title='Price (USD)',
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif comparison_type == 'Normalized Price':
                        # Normalize prices to 100 at the start
                        normalized_data = all_data.copy()
                        for ticker in compare_tickers:
                            normalized_data[ticker] = all_data[ticker] / all_data[ticker].iloc[0] * 100
                        
                        # Create normalized price chart
                        fig = go.Figure()
                        
                        for ticker in compare_tickers:
                            fig.add_trace(go.Scatter(
                                x=normalized_data.index,
                                y=normalized_data[ticker],
                                name=ticker
                            ))
                        
                        fig.update_layout(
                            title=f'Normalized Price Comparison (Base = 100)',
                            xaxis_title='Date',
                            yaxis_title='Normalized Price',
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif comparison_type == 'Returns':
                        # Calculate daily returns
                        returns_data = all_data.pct_change().dropna()
                        
                        # Create returns chart
                        fig = go.Figure()
                        
                        for ticker in compare_tickers:
                            fig.add_trace(go.Scatter(
                                x=returns_data.index,
                                y=returns_data[ticker],
                                mode='lines',
                                name=ticker
                            ))
                        
                        fig.update_layout(
                            title=f'Daily Returns Comparison',
                            xaxis_title='Date',
                            yaxis_title='Daily Return',
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Also show cumulative returns
                        cumulative_returns = (1 + returns_data).cumprod() - 1
                        
                        fig2 = go.Figure()
                        
                        for ticker in compare_tickers:
                            fig2.add_trace(go.Scatter(
                                x=cumulative_returns.index,
                                y=cumulative_returns[ticker],
                                name=ticker
                            ))
                        
                        fig2.update_layout(
                            title=f'Cumulative Returns Comparison',
                            xaxis_title='Date',
                            yaxis_title='Cumulative Return',
                            height=600
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                        
                    elif comparison_type == 'Volatility':
                        # Calculate rolling volatility (20-day standard deviation of returns)
                        returns_data = all_data.pct_change().dropna()
                        volatility_data = returns_data.rolling(window=20).std() * np.sqrt(252)  # Annualized
                        
                        # Create volatility chart
                        fig = go.Figure()
                        
                        for ticker in compare_tickers:
                            fig.add_trace(go.Scatter(
                                x=volatility_data.index,
                                y=volatility_data[ticker],
                                name=ticker
                            ))
                        
                        fig.update_layout(
                            title=f'20-Day Rolling Volatility (Annualized)',
                            xaxis_title='Date',
                            yaxis_title='Volatility',
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Add correlation heatmap
                    st.subheader('Correlation Matrix')
                    
                    # Calculate correlation matrix
                    try:
                        corr_matrix = all_data.corr()
                        
                        # Create heatmap
                        fig = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.index,
                            zmax=1,
                            colorscale='RdBu_r'
                        ))
                        
                        fig.update_layout(
                            title='Correlation Between Stocks',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error during stock comparison: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                        
elif page == 'About':
    st.title('About this Dashboard')
    st.markdown('''
    ## Stock Price Prediction and Analysis Dashboard
    
    This application provides comprehensive tools for stock price analysis, prediction, and portfolio optimization.
    
    ### Features:
    #### 1. Stock Price Prediction
    - **Linear Regression**: Basic time series forecasting
    - **Advanced Model**: Uses technical indicators and features 
    - **Ensemble Model**: Combines multiple algorithms for improved accuracy
    - **LSTM Model**: Deep learning for time series predictions
    - **Sentiment Analysis**: Incorporates news sentiment for enhanced predictions
    
    #### 2. Technical Analysis
    - Interactive stock charts with candlestick patterns
    - Multiple technical indicators (Moving Averages, RSI, MACD, Bollinger Bands)
    - Volume analysis
    - Indicator interpretation guides
    
    #### 3. Portfolio Optimization
    - Modern Portfolio Theory implementation
    - Efficient frontier visualization
    - Maximum Sharpe ratio portfolio
    - Minimum volatility portfolio
    - Target return and target volatility options
    - Portfolio backtesting with benchmark comparison
    
    #### 4. Multi-Stock Comparison
    - Compare multiple stocks on the same chart
    - Normalized price comparison
    - Performance and volatility comparisonsTechnologies Used:
    - Daily returns analysis    - Python 3
    This tool is for educational purposes only. Stock predictions shown here should not be used as the sole basis for investment decisions.
    Past performance is not indicative of future results, and all investments involve risk.
    ''')

# Display disclaimer at the bottom of every page
st.sidebar.header('Disclaimer')
st.sidebar.warning('''
This tool is for educational purposes only. Stock predictions shown here should not be used for actual investment decisions.
Past performance is not indicative of future results.
''')

# Add version number at the bottom
st.sidebar.text("Version 2.0")
st.sidebar.markdown("---")
