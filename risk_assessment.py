import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import pandas as pd
from utils import get_stock_data, timer_decorator, ensure_dir_exists, logger
from datetime import datetime, timedelta

class RiskAnalyzer:
    """
    A class for analyzing and visualizing investment risk metrics
    """
    
    def __init__(self, tickers=None, benchmark_ticker='SPY', start_date=None, end_date=None):
        """
        Initialize the risk analyzer
        
        Parameters:
        - tickers: List of stock tickers to analyze
        - benchmark_ticker: Ticker to use as market benchmark
        - start_date: Start date for analysis
        - end_date: End date for analysis
        """
        self.tickers = tickers or []
        self.benchmark_ticker = benchmark_ticker
        self.start_date = start_date or (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.returns = None
        self.benchmark_data = None
        self.benchmark_returns = None
        
        # Create directories for results
        ensure_dir_exists('results/risk/')
        
        logger.info(f"RiskAnalyzer initialized with {len(tickers) if tickers else 0} tickers")
        
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
            
        # Include benchmark in the download
        all_tickers = self.tickers + [self.benchmark_ticker]
        
        logger.info(f"Loading data for {len(all_tickers)} assets from {self.start_date} to {self.end_date}")
        
        try:
            # Download data
            data = get_stock_data(all_tickers, start=self.start_date, end=self.end_date)['Adj Close']
            
            # Handle single ticker case
            if len(all_tickers) == 1:
                data = pd.DataFrame(data, columns=all_tickers)
                
            # Extract benchmark data
            if self.benchmark_ticker in data.columns:
                self.benchmark_data = data[[self.benchmark_ticker]]
                # Remove benchmark from main data if it's not in tickers
                if self.benchmark_ticker not in self.tickers:
                    data = data.drop(columns=[self.benchmark_ticker])
                    
            # Store price data
            self.data = data
            
            # Calculate returns
            self.returns = self.data.pct_change().dropna()
            
            if self.benchmark_data is not None:
                self.benchmark_returns = self.benchmark_data.pct_change().dropna()
                
            logger.info(f"Loaded {len(self.data)} days of data")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
            
    def calculate_risk_metrics(self):
        """
        Calculate comprehensive risk metrics for each asset
        
        Returns:
        - DataFrame with risk metrics
        """
        if self.returns is None or self.benchmark_returns is None:
            logger.error("Data not loaded. Please call load_data() first.")
            return None
            
        logger.info("Calculating risk metrics")
        
        # Align benchmark returns with asset returns
        aligned_returns = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
        
        # Calculate risk metrics
        metrics = {}
        
        for ticker in self.returns.columns:
            returns = aligned_returns[ticker]
            benchmark_returns = aligned_returns[self.benchmark_ticker]
            
            # Annualization factor
            annual_factor = np.sqrt(252)
            
            # Calculate basic statistics
            mean_return = returns.mean()
            annual_return = (1 + mean_return) ** 252 - 1
            volatility = returns.std() * annual_factor
            
            # Calculate downside risk metrics
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * annual_factor if len(downside_returns) > 0 else np.nan
            
            # Calculate maximum drawdown
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max) - 1
            max_drawdown = drawdown.min()
            
            # Calculate Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Calculate Expected Shortfall / Conditional VaR
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else np.nan
            cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else np.nan
            
            # Calculate CAPM metrics
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else np.nan
            
            # Calculate alpha (Jensen's Alpha)
            risk_free_rate = 0.0  # Assuming 0% risk-free rate for simplicity
            benchmark_mean_return = benchmark_returns.mean()
            expected_return = risk_free_rate + beta * (benchmark_mean_return - risk_free_rate)
            alpha = mean_return - expected_return
            annual_alpha = (1 + alpha) ** 252 - 1
            
            # Calculate Sharpe, Sortino, and Treynor ratios
            sharpe_ratio = (mean_return - risk_free_rate) / returns.std() if returns.std() > 0 else np.nan
            annual_sharpe = sharpe_ratio * np.sqrt(252)
            sortino_ratio = (mean_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else np.nan
            treynor_ratio = (mean_return - risk_free_rate) / beta if beta > 0 else np.nan
            
            # Calculate information ratio
            tracking_error = (returns - benchmark_returns).std() * annual_factor
            information_ratio = (annual_return - (1 + benchmark_mean_return) ** 252 + 1) / tracking_error if tracking_error > 0 else np.nan
            
            # Calculate Calmar ratio
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
            
            # Store metrics
            metrics[ticker] = {
                'Annual Return': annual_return,
                'Annual Volatility': volatility,
                'Sharpe Ratio': annual_sharpe,
                'Sortino Ratio': sortino_ratio,
                'Max Drawdown': max_drawdown,
                'Calmar Ratio': calmar_ratio,
                'Beta': beta,
                'Alpha (Annual)': annual_alpha,
                'Treynor Ratio': treynor_ratio,
                'Information Ratio': information_ratio,
                'VaR (95%)': var_95,
                'CVaR (95%)': cvar_95,
                'VaR (99%)': var_99,
                'CVaR (99%)': cvar_99
            }
            
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics).T
        
        return metrics_df
        
    def plot_risk_return_profile(self, metrics_df=None):
        """
        Plot risk-return profile for all assets
        
        Parameters:
        - metrics_df: DataFrame with risk metrics (optional)
        
        Returns:
        - Figure object
        """
        if metrics_df is None:
            metrics_df = self.calculate_risk_metrics()
            if metrics_df is None:
                return None
                
        # Create risk-return scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each ticker
        scatter = ax.scatter(
            metrics_df['Annual Volatility'], 
            metrics_df['Annual Return'], 
            s=metrics_df['Sharpe Ratio'] * 50,  # Size based on Sharpe ratio
            c=metrics_df['Beta'],  # Color based on beta
            cmap='viridis',
            alpha=0.7
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Beta')
        
        # Add labels for each point
        for ticker in metrics_df.index:
            ax.annotate(
                ticker, 
                (metrics_df.loc[ticker, 'Annual Volatility'], metrics_df.loc[ticker, 'Annual Return']),
                xytext=(5, 5),
                textcoords='offset points'
            )
            
        # Set chart labels
        ax.set_xlabel('Annual Volatility')
        ax.set_ylabel('Annual Return')
        ax.set_title('Risk-Return Profile')
        ax.grid(True)
        
        # Add a reference line for the benchmark
        if self.benchmark_ticker in metrics_df.index:
            benchmark_return = metrics_df.loc[self.benchmark_ticker, 'Annual Return']
            benchmark_vol = metrics_df.loc[self.benchmark_ticker, 'Annual Volatility']
            ax.axhline(y=benchmark_return, color='r', linestyle='--', alpha=0.3, label=f'{self.benchmark_ticker} Return')
            ax.axvline(x=benchmark_vol, color='r', linestyle='--', alpha=0.3, label=f'{self.benchmark_ticker} Volatility')
            
        ax.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig('results/risk/risk_return_profile.png')
        
        return fig
        
    def plot_efficient_frontier_with_assets(self, metrics_df=None, num_portfolios=1000):
        """
        Plot efficient frontier with individual assets
        
        Parameters:
        - metrics_df: DataFrame with risk metrics (optional)
        - num_portfolios: Number of random portfolios to generate
        
        Returns:
        - Figure object
        """
        if self.returns is None:
            logger.error("Data not loaded. Please call load_data() first.")
            return None
            
        if metrics_df is None:
            metrics_df = self.calculate_risk_metrics()
            if metrics_df is None:
                return None
                
        # Generate random portfolios
        logger.info(f"Generating {num_portfolios} random portfolios for efficient frontier")
        
        num_assets = len(self.tickers)
        results = np.zeros((num_portfolios, num_assets + 3))
        
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            
            # Calculate portfolio return
            portfolio_return = np.sum(self.returns.mean() * weights) * 252
            
            # Calculate portfolio volatility
            portfolio_volatility = np.sqrt(
                np.dot(weights.T, np.dot(self.returns.cov() * 252, weights))
            )
            
            # Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)
            sharpe = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
            
            # Store results
            results[i, 0] = portfolio_return
            results[i, 1] = portfolio_volatility
            results[i, 2] = sharpe
            results[i, 3:] = weights
                
        # Convert to DataFrame
        columns = ['Return', 'Volatility', 'Sharpe']
        columns.extend(self.tickers)
        portfolios_df = pd.DataFrame(results, columns=columns)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot random portfolios
        scatter = ax.scatter(
            portfolios_df['Volatility'], 
            portfolios_df['Return'],
            c=portfolios_df['Sharpe'],
            cmap='viridis',
            alpha=0.5
        )
        
        # Plot individual assets
        for i, ticker in enumerate(self.tickers):
            if ticker in metrics_df.index:
                annual_return = metrics_df.loc[ticker, 'Annual Return']
                annual_volatility = metrics_df.loc[ticker, 'Annual Volatility']
                ax.scatter(annual_volatility, annual_return, 
                          marker='o', s=100, color='red', label=ticker)
                ax.annotate(ticker, (annual_volatility, annual_return), 
                           xytext=(5, 5), textcoords='offset points')
        
        # Find and plot minimum volatility portfolio
        min_vol_idx = portfolios_df['Volatility'].idxmin()
        ax.scatter(
            portfolios_df.loc[min_vol_idx, 'Volatility'],
            portfolios_df.loc[min_vol_idx, 'Return'],
            marker='*', color='green', s=200, label='Minimum Volatility'
        )
        
        # Find and plot maximum Sharpe ratio portfolio
        max_sharpe_idx = portfolios_df['Sharpe'].idxmax()
        ax.scatter(
            portfolios_df.loc[max_sharpe_idx, 'Volatility'],
            portfolios_df.loc[max_sharpe_idx, 'Return'],
            marker='*', color='blue', s=200, label='Maximum Sharpe Ratio'
        )
        
        # Set chart labels
        ax.set_xlabel('Annual Volatility')
        ax.set_ylabel('Annual Return')
        ax.set_title('Efficient Frontier with Individual Assets')
        ax.grid(True)
        
        # Add a colorbar to show Sharpe ratio
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio')
        
        # Add legend
        ax.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig('results/risk/efficient_frontier.png')
        
        return fig
        
    def plot_correlation_heatmap(self):
        """
        Plot correlation heatmap for all assets
        
        Returns:
        - Figure object
        """
        if self.returns is None:
            logger.error("Data not loaded. Please call load_data() first.")
            return None
            
        # Calculate correlation matrix
        corr_matrix = self.returns.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            mask=mask,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": .8},
            ax=ax
        )
        
        ax.set_title('Correlation Matrix')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig('results/risk/correlation_heatmap.png')
        
        return fig
        
    def plot_rolling_beta(self, window=60):
        """
        Plot rolling beta for all assets against the benchmark
        
        Parameters:
        - window: Window size for rolling calculation
        
        Returns:
        - Figure object
        """
        if self.returns is None or self.benchmark_returns is None:
            logger.error("Data not loaded. Please call load_data() first.")
            return None
            
        # Create empty DataFrame to store rolling betas
        rolling_betas = pd.DataFrame(index=self.returns.index[window-1:])
        
        benchmark_returns = self.benchmark_returns[self.benchmark_ticker]
        
        # Calculate rolling beta for each asset
        for ticker in self.returns.columns:
            asset_returns = self.returns[ticker]
            
            # Calculate rolling covariance
            rolling_cov = asset_returns.rolling(window=window).cov(benchmark_returns)
            
            # Calculate rolling variance of benchmark
            rolling_var = benchmark_returns.rolling(window=window).var()
            
            # Calculate rolling beta
            rolling_beta = rolling_cov / rolling_var
            
            rolling_betas[ticker] = rolling_beta
            
        # Plot rolling betas
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for ticker in rolling_betas.columns:
            ax.plot(rolling_betas.index, rolling_betas[ticker], label=ticker)
            
        # Add reference line at beta = 1
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.3)
        
        ax.set_title(f'Rolling {window}-Day Beta')
        ax.set_xlabel('Date')
        ax.set_ylabel('Beta')
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Save the figure
        plt.tight_layout()
        plt.savefig('results/risk/rolling_beta.png')
        
        return fig
        
    def plot_drawdowns(self):
        """
        Plot drawdowns for all assets
        
        Returns:
        - Figure object
        """
        if self.returns is None:
            logger.error("Data not loaded. Please call load_data() first.")
            return None
            
        # Calculate drawdowns for each asset
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for ticker in self.returns.columns:
            # Calculate cumulative returns
            cum_returns = (1 + self.returns[ticker]).cumprod()
            
            # Calculate running maximum
            running_max = cum_returns.cummax()
            
            # Calculate drawdown
            drawdown = (cum_returns / running_max) - 1
            
            # Plot drawdown
            ax.plot(drawdown.index, drawdown * 100, label=ticker)
            
        # Add benchmark drawdown
        if self.benchmark_returns is not None:
            # Calculate benchmark drawdown
            bench_cum_returns = (1 + self.benchmark_returns[self.benchmark_ticker]).cumprod()
            bench_running_max = bench_cum_returns.cummax()
            bench_drawdown = (bench_cum_returns / bench_running_max) - 1
            
            # Plot benchmark drawdown
            ax.plot(bench_drawdown.index, bench_drawdown * 100, 'k--', 
                   label=f'{self.benchmark_ticker} (Benchmark)', linewidth=2)
            
        ax.set_title('Historical Drawdowns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Save the figure
        plt.tight_layout()
        plt.savefig('results/risk/drawdowns.png')
        
        return fig

    def generate_risk_report(self):
        """
        Generate comprehensive risk report with all metrics and visualizations
        
        Returns:
        - Dictionary with all risk analysis results
        """
        if self.returns is None:
            self.load_data()
            if self.returns is None:
                logger.error("Failed to load data. Cannot generate report.")
                return None
                
        logger.info("Generating comprehensive risk report")
        
        # Calculate risk metrics
        metrics_df = self.calculate_risk_metrics()
        
        # Generate all plots
        risk_return_fig = self.plot_risk_return_profile(metrics_df)
        efficient_frontier_fig = self.plot_efficient_frontier_with_assets(metrics_df)
        correlation_fig = self.plot_correlation_heatmap()
        rolling_beta_fig = self.plot_rolling_beta()
        drawdowns_fig = self.plot_drawdowns()
        
        # Sort assets by Sharpe ratio
        top_assets_by_sharpe = metrics_df.sort_values('Sharpe Ratio', ascending=False)
        
        # Sort assets by risk-adjusted return (Sharpe)
        top_assets_by_return = metrics_df.sort_values('Annual Return', ascending=False)
        
        # Sort assets by lowest volatility
        top_assets_by_volatility = metrics_df.sort_values('Annual Volatility')
        
        # Format metrics for display (to 4 decimal places)
        formatted_metrics = metrics_df.round(4)
        
        # Save metrics to CSV
        formatted_metrics.to_csv('results/risk/risk_metrics.csv')
        
        # Return all results
        return {
            'metrics': metrics_df,
            'formatted_metrics': formatted_metrics,
            'top_by_sharpe': top_assets_by_sharpe,
            'top_by_return': top_assets_by_return,
            'top_by_volatility': top_assets_by_volatility,
            'figures': {
                'risk_return': risk_return_fig,
                'efficient_frontier': efficient_frontier_fig,
                'correlation': correlation_fig,
                'rolling_beta': rolling_beta_fig,
                'drawdowns': drawdowns_fig
            }
        }


def main():
    # Example usage
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'V', 'JPM', 'WMT', 'PG']
    benchmark = 'SPY'
    start_date = "2018-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Initialize risk analyzer
    analyzer = RiskAnalyzer(tickers, benchmark, start_date, end_date)
    
    # Load data
    analyzer.load_data()
    
    # Generate comprehensive report
    report = analyzer.generate_risk_report()
    
    print("\nRisk Analysis Report")
    print("===================")
    print(f"\nTop assets by Sharpe ratio:")
    print(report['top_by_sharpe'][['Annual Return', 'Annual Volatility', 'Sharpe Ratio']].head())
    
    print(f"\nTop assets by Return:")
    print(report['top_by_return'][['Annual Return', 'Annual Volatility', 'Sharpe Ratio']].head())
    
    print(f"\nTop assets by lowest Volatility:")
    print(report['top_by_volatility'][['Annual Return', 'Annual Volatility', 'Sharpe Ratio']].head())
    
    print("\nDone! Check the 'results/risk' folder for generated visualizations.")


if __name__ == "__main__":
    main()
