import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy.optimize import minimize
import seaborn as sns
from sklearn.covariance import ledoit_wolf
import warnings
from utils import get_stock_data, timer_decorator, ensure_dir_exists, logger

class PortfolioOptimizer:
    """
    A class for portfolio optimization using various strategies
    """
    
    def __init__(self, tickers=None, start_date=None, end_date=None):
        """
        Initialize the portfolio optimizer
        
        Parameters:
        - tickers: List of stock tickers
        - start_date: Start date for historical data
        - end_date: End date for historical data
        """
        self.tickers = tickers or []
        self.start_date = start_date or (datetime.datetime.now() - datetime.timedelta(days=365*3)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        
        # Create results directory
        ensure_dir_exists('results/portfolio/')
        
        logger.info(f"PortfolioOptimizer initialized with {len(tickers) if tickers else 0} tickers")

    @timer_decorator
    def load_data(self, tickers=None, start_date=None, end_date=None):
        """
        Load historical price data for analysis
        
        Parameters:
        - tickers: List of stock tickers (optional)
        - start_date: Start date (optional)
        - end_date: End date (optional)
        
        Returns:
        - DataFrame with historical price data
        """
        if tickers:
            self.tickers = tickers
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date
            
        logger.info(f"Loading data for {len(self.tickers)} stocks from {self.start_date} to {self.end_date}")
        
        try:
            # Download data with auto_adjust=False to get both Close and Adj Close
            data = get_stock_data(self.tickers, start=self.start_date, end=self.end_date, auto_adjust=False)
            
            if data.empty:
                logger.error("No data found for the selected tickers")
                return None
                
            # Check if we have multiple tickers or just one
            if len(self.tickers) == 1:
                # For a single ticker, ensure the data is formatted correctly
                adj_close = data['Adj Close']
                adj_close = pd.DataFrame(adj_close, columns=self.tickers)
            else:
                # For multiple tickers, extract the Adj Close data
                adj_close = data['Adj Close']
            
            # Store price data
            self.data = adj_close
            
            # Calculate returns
            self.returns = self.data.pct_change().dropna()
            
            logger.info(f"Loaded {len(self.data)} days of data")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            
            # Fallback to using Close prices if Adj Close is not available
            try:
                logger.info("Attempting to use Close prices instead of Adj Close")
                data = get_stock_data(self.tickers, start=self.start_date, end=self.end_date)
                
                if data.empty:
                    logger.error("No data found for the selected tickers")
                    return None
                
                # Use Close prices
                if len(self.tickers) == 1:
                    close_prices = data['Close']
                    close_prices = pd.DataFrame(close_prices, columns=self.tickers)
                else:
                    close_prices = data['Close']
                
                # Store price data
                self.data = close_prices
                
                # Calculate returns
                self.returns = self.data.pct_change().dropna()
                
                logger.info(f"Loaded {len(self.data)} days of data (using Close prices)")
                return self.data
                
            except Exception as nested_e:
                logger.error(f"Error in fallback loading: {str(nested_e)}")
                return None

    def calculate_expected_shortfall(self, weights, alpha=0.05):
        """
        Calculate Expected Shortfall (Conditional Value at Risk) for the portfolio
        
        Parameters:
        - weights: Portfolio weights
        - alpha: Significance level (default 0.05 for 95% confidence)
        
        Returns:
        - Expected shortfall value
        """
        # Convert weights to array
        weights = np.array(weights)
        
        # Calculate portfolio returns - direct matrix multiplication gives us daily portfolio returns
        portfolio_returns = self.returns.values @ weights
        
        # Sort returns
        sorted_returns = np.sort(portfolio_returns)
        
        # Find cutoff index
        cutoff_index = int(np.ceil(alpha * len(sorted_returns)))
        
        # Calculate expected shortfall (mean of worst alpha% returns)
        expected_shortfall = -np.mean(sorted_returns[:cutoff_index])
        
        return expected_shortfall
        
    def minimize_expected_shortfall(self, weights):
        """
        Objective function to minimize expected shortfall
        """
        return self.calculate_expected_shortfall(weights)

    def calculate_drawdowns(self, returns):
        """
        Calculate drawdowns from returns series
        
        Parameters:
        - returns: Series of returns
        
        Returns:
        - DataFrame with wealth index and drawdowns
        """
        # Calculate wealth index
        wealth_index = (1 + returns).cumprod()
        
        # Calculate previous peaks
        previous_peaks = wealth_index.cummax()
        
        # Calculate drawdowns
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        
        return pd.DataFrame({
            'Wealth': wealth_index,
            'Previous Peak': previous_peaks,
            'Drawdown': drawdowns
        })

    def negative_sharpe(self, weights, risk_free_rate=0.0):
        """
        Calculate the negative Sharpe ratio for minimization
        
        Parameters:
        - weights: Portfolio weights
        - risk_free_rate: Risk-free rate
        
        Returns:
        - Negative Sharpe ratio
        """
        # Check if mean_returns or cov_matrix is None
        if self.mean_returns is None or self.cov_matrix is None:
            # Try to recalculate returns if data is available
            if hasattr(self, 'data') and self.data is not None:
                self._calculate_returns_and_covariance()
            else:
                print("Error: Cannot calculate Sharpe ratio - mean returns or covariance matrix is not initialized")
                return 0  # Return neutral value to avoid optimization crash
        
        weights = np.array(weights)
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        
        # Calculate Sharpe ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Return negative Sharpe ratio for minimization
        return -sharpe_ratio
    
    def _calculate_returns_and_covariance(self):
        """
        Calculate mean returns and covariance matrix from price data
        """
        if not hasattr(self, 'data') or self.data is None:
            print("Error: No price data available to calculate returns")
            return
            
        try:
            # Calculate daily returns
            returns = self.data.pct_change().dropna()
            
            # Calculate mean returns and covariance
            self.mean_returns = returns.mean()
            self.cov_matrix = returns.cov()
            
            print(f"Successfully calculated returns for {len(self.mean_returns)} assets")
        except Exception as e:
            print(f"Error calculating returns and covariance: {str(e)}")

    def calculate_portfolio_performance(self, weights):
        """
        Calculate performance metrics for a given set of portfolio weights
        
        Parameters:
        - weights: Array of portfolio weights
        
        Returns:
        - Dictionary with performance metrics
        """
        # Convert weights to numpy array if not already
        weights = np.array(weights)
        
        # Calculate annualized return
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        
        # Calculate annualized volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }

    def optimize_portfolio(self, risk_free_rate=0.0, target_return=None, target_volatility=None, method='sharpe'):
        """
        Optimize portfolio weights using various methods
        
        Parameters:
        - risk_free_rate: Risk-free rate for Sharpe ratio calculation
        - target_return: Target return for minimum volatility optimization
        - target_volatility: Target volatility for maximum return optimization
        - method: Optimization method ('sharpe', 'min_vol', 'max_return', 'min_cvar')
        
        Returns:
        - Dictionary with optimization results
        """
        if self.returns is None:
            logger.error("No data loaded. Please call load_data() first.")
            return None
            
        num_assets = len(self.tickers)
        logger.info(f"Optimizing portfolio using {method} method for {num_assets} assets")
        
        # Initial weights (equal weight)
        initial_weights = np.array([1.0/num_assets] * num_assets)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Constraint: weights sum to 1
        constraint = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        
        if method == 'sharpe':
            # Maximize Sharpe ratio
            result = minimize(
                self.negative_sharpe,
                initial_weights, 
                method='SLSQP',
                args=(risk_free_rate,),
                bounds=bounds, 
                constraints=(constraint,)
            )
            
        elif method == 'min_vol':
            # Minimize volatility
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
                
            result = minimize(
                portfolio_volatility,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=(constraint,)
            )
            
        elif method == 'max_return':
            # Maximize return
            def negative_portfolio_return(weights):
                return -np.sum(self.mean_returns * weights) * 252
                
            result = minimize(
                negative_portfolio_return,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=(constraint,)
            )
            
        elif method == 'target_return':
            # Minimize volatility subject to target return
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
            
            # Target return constraint
            target_return_constraint = {
                'type': 'eq',
                'fun': lambda weights: np.sum(self.mean_returns * weights) * 252 - target_return
            }
            
            result = minimize(
                portfolio_volatility,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=(constraint, target_return_constraint)
            )
            
        elif method == 'target_volatility':
            # Maximize return subject to target volatility
            def negative_portfolio_return(weights):
                return -np.sum(self.mean_returns * weights) * 252
                
            # Target volatility constraint
            target_volatility_constraint = {
                'type': 'eq',
                'fun': lambda weights: np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))) - target_volatility
            }
            
            result = minimize(
                negative_portfolio_return,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=(constraint, target_volatility_constraint)
            )
        
        elif method == 'min_cvar':
            result = minimize(
                self.minimize_expected_shortfall,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=(constraint,)
            )
        
        else:
            logger.error(f"Invalid optimization method: {method}")
            return None
        
        # Extract optimal weights
        optimal_weights = result['x']
        
        # Calculate performance metrics
        performance = self.calculate_portfolio_performance(optimal_weights)
        
        # Calculate expected shortfall
        expected_shortfall = self.calculate_expected_shortfall(optimal_weights)
        
        # Create a DataFrame of weights for better readability
        weights_df = pd.Series(optimal_weights, index=self.tickers)
        
        # Create results dictionary with capitalized keys for consistency with app.py
        optimization_results = {
            'Weights': weights_df,
            'Return': performance['return'],
            'Volatility': performance['volatility'],
            'Sharpe': performance['sharpe_ratio'],
            'ExpectedShortfall': expected_shortfall,
            'Method': method,
            'Success': result['success'],
            'Message': result['message']
        }
        
        logger.info(f"Optimization {result['success']} - {result['message']}")
        return optimization_results

    def backtest_portfolio(self, weights, start_date=None, end_date=None):
        """
        Backtest a portfolio with specified weights
        
        Parameters:
        - weights: Array of weights for each asset
        - start_date: Start date for backtesting (within loaded data range)
        - end_date: End date for backtesting (within loaded data range)
        
        Returns:
        - DataFrame with portfolio performance
        - Dictionary with portfolio metrics
        """
        if self.data is None:
            logger.error("No data loaded. Please call load_data() first.")
            return None
        
        # Convert weights to dictionary
        if isinstance(weights, np.ndarray):
            weights_dict = {ticker: weight for ticker, weight in zip(self.tickers, weights)}
        elif isinstance(weights, dict):
            weights_dict = weights
        else:
            logger.error("Invalid weights format")
            return None
        
        # Filter data for backtest period
        if start_date:
            backtest_data = self.data[self.data.index >= start_date]
        else:
            backtest_data = self.data.copy()
            
        if end_date:
            backtest_data = backtest_data[backtest_data.index <= end_date]
        
        # Calculate portfolio value
        portfolio = pd.DataFrame(index=backtest_data.index)
        
        # Add each stock's contribution based on weight
        for ticker, weight in weights_dict.items():
            if ticker in backtest_data.columns:
                # Normalize price to start with weight * 100
                normalized_price = backtest_data[ticker] / backtest_data[ticker].iloc[0] * weight * 100
                portfolio[ticker] = normalized_price
            else:
                logger.warning(f"Warning: {ticker} not in data")
        
        # Total portfolio value
        portfolio['Total'] = portfolio.sum(axis=1)
        
        # Calculate daily returns
        portfolio['Daily_Return'] = portfolio['Total'].pct_change()
        
        # Calculate cumulative returns
        portfolio['Cumulative_Return'] = (portfolio['Daily_Return'] + 1).cumprod() - 1
        
        # Calculate metrics
        total_days = len(portfolio)
        total_return = (portfolio['Total'].iloc[-1] / portfolio['Total'].iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / total_days) - 1
        
        daily_returns = portfolio['Daily_Return'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = annual_return / volatility if volatility != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino = annual_return / downside_deviation if downside_deviation != 0 else 0
        
        # Maximum drawdown
        portfolio['Peak'] = portfolio['Total'].cummax()
        portfolio['Drawdown'] = (portfolio['Total'] - portfolio['Peak']) / portfolio['Peak']
        max_drawdown = portfolio['Drawdown'].min()
        
        # Calculate Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk (VaR)
        var_95 = np.percentile(daily_returns, 5)
        var_99 = np.percentile(daily_returns, 1)
        
        # Expected Shortfall (Conditional VaR)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()
        cvar_99 = daily_returns[daily_returns <= var_99].mean()
        
        # Print metrics
        logger.info(f"Backtest Period: {backtest_data.index[0]} to {backtest_data.index[-1]}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Annual Return: {annual_return:.2%}")
        logger.info(f"Annual Volatility: {volatility:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Sortino Ratio: {sortino:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"Calmar Ratio: {calmar_ratio:.2f}")
        logger.info(f"VaR (95%): {var_95:.2%}")
        logger.info(f"Expected Shortfall (95%): {cvar_95:.2%}")
        
        # Create metrics dictionary
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'var_99': var_99,
            'cvar_99': cvar_99
        }
        
        return portfolio, metrics

    def plot_backtest_results(self, portfolio_df, ticker_benchmark=None):
        """
        Plot backtest results
        
        Parameters:
        - portfolio_df: Portfolio DataFrame from backtest_portfolio
        - ticker_benchmark: Optional ticker symbol to use as benchmark
        """
        plt.figure(figsize=(14, 12))
        
        # Plot total portfolio value
        plt.subplot(3, 1, 1)
        plt.plot(portfolio_df['Total'], label='Portfolio')
        
        # Add benchmark if specified
        if ticker_benchmark and ticker_benchmark in self.data.columns:
            # Normalize benchmark to match portfolio starting value
            normalized_benchmark = self.data[ticker_benchmark] / self.data[ticker_benchmark].loc[portfolio_df.index[0]] * 100
            benchmark_slice = normalized_benchmark.loc[portfolio_df.index]
            plt.plot(benchmark_slice, label=f'{ticker_benchmark} (Benchmark)')
        
        plt.title('Portfolio Performance')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        # Plot drawdown
        plt.subplot(3, 1, 2)
        drawdown = (portfolio_df['Total'] / portfolio_df['Total'].cummax() - 1) * 100
        plt.fill_between(portfolio_df.index, drawdown, 0, color='red', alpha=0.3)
        plt.title('Portfolio Drawdown')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        # Plot daily returns
        plt.subplot(3, 1, 3)
        plt.bar(portfolio_df.index, portfolio_df['Daily_Return'] * 100, color='blue', alpha=0.7)
        plt.title('Daily Returns')
        plt.ylabel('Return (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/portfolio/portfolio_backtest.png')
        
        # Create additional chart for asset allocation
        plt.figure(figsize=(10, 6))
        asset_values = portfolio_df.iloc[-1].drop(['Total', 'Daily_Return', 'Cumulative_Return', 'Peak', 'Drawdown'])
        asset_allocation = asset_values / asset_values.sum() * 100
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(asset_allocation)))
        asset_allocation.plot(kind='pie', autopct='%1.1f%%', colors=colors)
        plt.title('Final Portfolio Allocation')
        plt.ylabel('')
        plt.savefig('results/portfolio/portfolio_allocation.png')

def main():
    # Example usage
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'V', 'JPM', 'WMT', 'PG']
    start_date = "2018-01-01"
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Initialize portfolio optimizer
    optimizer = PortfolioOptimizer(tickers, start_date, end_date)
    
    # Load data
    optimizer.load_data()
    
    # Plot correlation matrix
    optimizer.plot_correlation_matrix()
    
    # Generate efficient frontier
    print("Generating efficient frontier...")
    frontier_results = optimizer.generate_efficient_frontier(num_portfolios=1000)
    
    # Plot efficient frontier
    optimizer.plot_efficient_frontier(frontier_results)
    
    # Print optimal portfolios
    print("\nMaximum Sharpe Ratio Portfolio:")
    max_sharpe = frontier_results['max_sharpe']
    print(f"Expected Return: {max_sharpe['Return']:.4f}")
    print(f"Volatility: {max_sharpe['Volatility']:.4f}")
    print(f"Sharpe Ratio: {max_sharpe['Sharpe']:.4f}")
    
    print("\nMinimum Volatility Portfolio:")
    min_vol = frontier_results['min_vol']
    print(f"Expected Return: {min_vol['Return']:.4f}")
    print(f"Volatility: {min_vol['Volatility']:.4f}")
    print(f"Sharpe Ratio: {min_vol['Sharpe']:.4f}")
    
    # Out-of-sample testing
    print("\nRunning out-of-sample optimization and backtesting...")
    results = optimizer.optimize_and_backtest(method='sharpe', backtest_split=0.3)
    
    print("Done!")

if __name__ == "__main__":
    main()
