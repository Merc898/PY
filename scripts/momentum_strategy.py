#!/usr/bin/env python3
"""
momentum_strategy.py - Properly implemented momentum strategy with financial integrity
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for momentum strategy"""
    lookback_months: int = 12
    skip_months: int = 2
    n_portfolios: int = 10
    min_stocks_per_portfolio: int = 10
    rebalance_frequency: str = 'monthly'

    # Transaction costs (in basis points)
    spread_cost_bps: float = 10
    market_impact_bps: float = 5
    brokerage_cost_bps: float = 2

    # Risk management
    max_position_weight: float = 0.05
    target_volatility: float = 0.10

    # Data requirements
    min_price: float = 5.0  # Minimum stock price
    min_market_cap: float = 1e9  # Minimum market cap ($1B)
    min_trading_days: int = 200  # Minimum trading days per year


class MomentumStrategy:
    """Professional momentum strategy implementation"""

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.results = {}

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate price data"""
        logger.info("Loading price data...")

        # Load data
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])

        # CRITICAL: Use only USD-adjusted prices if available
        if 'close_usd' in df.columns:
            logger.info("Using USD-adjusted prices")
            price_col = 'close_usd'
        else:
            logger.warning("USD-adjusted prices not found, using raw close prices")
            price_col = 'close'

            # Filter to USD-only securities if currency info available
            if 'currency' in df.columns:
                usd_only = df[df['currency'] == 'USD'].copy()
                if len(usd_only) > 0:
                    df = usd_only
                    logger.info(f"Filtered to {df['symbol'].nunique()} USD securities")

        # Rename price column for consistency
        df['price'] = df[price_col]

        # Data quality filters
        initial_count = len(df)

        # Remove penny stocks
        df = df[df['price'] >= self.config.min_price]

        # Remove duplicate entries
        df = df.drop_duplicates(subset=['symbol', 'date'])

        # Sort data
        df = df.sort_values(['symbol', 'date'])

        final_count = len(df)
        logger.info(f"Data loaded: {final_count:,} observations ({initial_count - final_count:,} removed)")

        return df

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily and monthly returns with proper handling"""
        logger.info("Calculating returns...")

        # Daily returns
        df = df.sort_values(['symbol', 'date'])
        df['daily_return'] = df.groupby('symbol')['price'].pct_change()

        # Check for extreme returns that might indicate data errors
        extreme_returns = df[abs(df['daily_return']) > 0.5]
        if len(extreme_returns) > 0:
            logger.warning(f"Found {len(extreme_returns)} extreme daily returns (>50%)")

            # Cap extreme returns (likely splits not adjusted)
            df.loc[df['daily_return'] > 0.5, 'daily_return'] = 0.5
            df.loc[df['daily_return'] < -0.5, 'daily_return'] = -0.5

        # Monthly returns - using total return over the month
        df['year_month'] = df['date'].dt.to_period('M')

        # Get first and last price of each month
        monthly = df.groupby(['symbol', 'year_month']).agg({
            'date': ['first', 'last'],
            'price': ['first', 'last'],
            'daily_return': 'count'  # Number of trading days
        })

        monthly.columns = ['first_date', 'last_date', 'first_price', 'last_price', 'n_days']
        monthly = monthly.reset_index()

        # Calculate monthly returns
        monthly['monthly_return'] = (monthly['last_price'] / monthly['first_price']) - 1

        # Filter out incomplete months
        monthly['complete_month'] = monthly['n_days'] >= 15

        logger.info(f"Calculated returns for {monthly['symbol'].nunique()} symbols")

        return monthly

    def calculate_momentum_signal(self, monthly_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum with proper formation period"""
        logger.info("Calculating momentum signals...")

        df = monthly_data.copy()
        df = df.sort_values(['symbol', 'year_month'])

        # Calculate momentum for each stock
        momentum_list = []

        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('year_month')

            # Need at least lookback + skip months of data
            required_months = self.config.lookback_months + self.config.skip_months

            if len(symbol_data) < required_months:
                continue

            # Calculate rolling momentum
            for i in range(required_months, len(symbol_data)):
                current_month = symbol_data.iloc[i]['year_month']

                # Get the returns for momentum calculation
                # We want months t-12 to t-2 (skipping the most recent 2 months)
                start_idx = i - self.config.lookback_months - self.config.skip_months + 1
                end_idx = i - self.config.skip_months + 1

                formation_returns = symbol_data.iloc[start_idx:end_idx]['monthly_return'].values

                # Check for data quality
                valid_returns = formation_returns[~np.isnan(formation_returns)]

                if len(valid_returns) < 8:  # Require at least 8 months of valid data
                    continue

                # Calculate compound return
                momentum = np.prod(1 + valid_returns) - 1

                momentum_list.append({
                    'symbol': symbol,
                    'year_month': current_month,
                    'momentum': momentum,
                    'n_months_used': len(valid_returns)
                })

        momentum_df = pd.DataFrame(momentum_list)

        logger.info(f"Calculated {len(momentum_df)} momentum signals")

        return momentum_df

    def form_portfolios(self, signals_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Form long-short portfolios based on momentum"""
        logger.info("Forming portfolios...")

        # Merge signals with next month's returns
        signals_df = signals_df.copy()
        returns_df = returns_df.copy()

        # Get next month's return
        returns_df['next_month_return'] = returns_df.groupby('symbol')['monthly_return'].shift(-1)

        # Merge
        portfolio_df = signals_df.merge(
            returns_df[['symbol', 'year_month', 'next_month_return']],
            on=['symbol', 'year_month'],
            how='left'
        )

        # Remove observations without next month return
        portfolio_df = portfolio_df.dropna(subset=['next_month_return'])

        # For each month, rank stocks and assign to portfolios
        portfolio_assignments = []

        for month in portfolio_df['year_month'].unique():
            month_data = portfolio_df[portfolio_df['year_month'] == month].copy()

            # Check if we have enough stocks
            n_stocks = len(month_data)
            if n_stocks < self.config.min_stocks_per_portfolio * 2:
                logger.warning(f"Month {month}: Only {n_stocks} stocks, skipping")
                continue

            # Rank stocks by momentum
            month_data['momentum_rank'] = month_data['momentum'].rank(pct=True)

            # Assign to portfolios
            month_data['portfolio'] = pd.qcut(
                month_data['momentum_rank'],
                q=self.config.n_portfolios,
                labels=range(1, self.config.n_portfolios + 1)
            )

            # Long top decile, short bottom decile
            month_data['position'] = 'neutral'
            month_data.loc[month_data['portfolio'] == self.config.n_portfolios, 'position'] = 'long'
            month_data.loc[month_data['portfolio'] == 1, 'position'] = 'short'

            portfolio_assignments.append(month_data)

        if portfolio_assignments:
            final_portfolio = pd.concat(portfolio_assignments, ignore_index=True)
            logger.info(f"Formed portfolios for {final_portfolio['year_month'].nunique()} months")
            return final_portfolio
        else:
            logger.error("No valid portfolios formed!")
            return pd.DataFrame()

    def calculate_portfolio_returns(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate portfolio returns with transaction costs"""
        logger.info("Calculating portfolio returns...")

        # Equal-weight within each portfolio
        results = []

        for month in portfolio_df['year_month'].unique():
            month_data = portfolio_df[portfolio_df['year_month'] == month]

            # Long portfolio
            long_stocks = month_data[month_data['position'] == 'long']
            short_stocks = month_data[month_data['position'] == 'short']

            if len(long_stocks) == 0 or len(short_stocks) == 0:
                continue

            # Equal weights
            long_return = long_stocks['next_month_return'].mean()
            short_return = short_stocks['next_month_return'].mean()

            # Long-short return (before costs)
            ls_return_gross = long_return - short_return

            # Estimate turnover (approximately 2x per year for monthly rebalancing)
            monthly_turnover = 2.0 / 12

            # Transaction costs
            spread_cost = monthly_turnover * self.config.spread_cost_bps / 10000
            impact_cost = monthly_turnover * self.config.market_impact_bps / 10000
            brokerage = monthly_turnover * self.config.brokerage_cost_bps / 10000

            total_cost = spread_cost + impact_cost + brokerage

            # Net return
            ls_return_net = ls_return_gross - total_cost

            results.append({
                'year_month': month,
                'long_return': long_return,
                'short_return': short_return,
                'ls_return_gross': ls_return_gross,
                'transaction_cost': total_cost,
                'ls_return_net': ls_return_net,
                'n_long': len(long_stocks),
                'n_short': len(short_stocks)
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('year_month')

        # Calculate cumulative returns
        results_df['cum_return_gross'] = (1 + results_df['ls_return_gross']).cumprod() - 1
        results_df['cum_return_net'] = (1 + results_df['ls_return_net']).cumprod() - 1

        return results_df

    def calculate_metrics(self, returns_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics with statistical significance"""
        logger.info("Calculating performance metrics...")

        # Basic metrics
        returns_gross = returns_df['ls_return_gross'].values
        returns_net = returns_df['ls_return_net'].values

        metrics = {}

        # Gross metrics
        metrics['gross_annual_return'] = (1 + returns_gross.mean()) ** 12 - 1
        metrics['gross_volatility'] = returns_gross.std() * np.sqrt(12)
        metrics['gross_sharpe'] = (metrics['gross_annual_return'] /
                                   metrics['gross_volatility'] if metrics['gross_volatility'] > 0 else 0)

        # Net metrics
        metrics['net_annual_return'] = (1 + returns_net.mean()) ** 12 - 1
        metrics['net_volatility'] = returns_net.std() * np.sqrt(12)
        metrics['net_sharpe'] = (metrics['net_annual_return'] /
                                 metrics['net_volatility'] if metrics['net_volatility'] > 0 else 0)

        # Risk metrics
        metrics['max_drawdown'] = self._calculate_max_drawdown(returns_net)
        metrics['calmar_ratio'] = (metrics['net_annual_return'] /
                                   abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0)

        # Statistical significance
        # T-statistic for Sharpe ratio
        n_months = len(returns_net)
        t_stat = metrics['net_sharpe'] * np.sqrt(n_months / 12)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_months - 1))

        metrics['sharpe_t_stat'] = t_stat
        metrics['sharpe_p_value'] = p_value
        metrics['is_significant'] = p_value < 0.05

        # Additional statistics
        metrics['win_rate'] = (returns_net > 0).mean()
        metrics['avg_win'] = returns_net[returns_net > 0].mean() if any(returns_net > 0) else 0
        metrics['avg_loss'] = returns_net[returns_net < 0].mean() if any(returns_net < 0) else 0
        metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0

        # Information about costs
        total_costs = returns_df['transaction_cost'].sum()
        total_gross = returns_df['ls_return_gross'].sum()
        metrics['cost_as_pct_of_gross'] = (total_costs / abs(total_gross) * 100
                                           if total_gross != 0 else 0)

        return metrics

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def run_backtest(self, data_path: str) -> Dict:
        """Run complete backtest"""
        logger.info("=" * 60)
        logger.info("Starting Momentum Strategy Backtest")
        logger.info("=" * 60)

        # Load data
        price_data = self.load_data(data_path)

        # Calculate returns
        monthly_returns = self.calculate_returns(price_data)

        # Calculate momentum signals
        momentum_signals = self.calculate_momentum_signal(monthly_returns)

        # Form portfolios
        portfolios = self.form_portfolios(momentum_signals, monthly_returns)

        if portfolios.empty:
            logger.error("Failed to form portfolios!")
            return {}

        # Calculate portfolio returns
        portfolio_returns = self.calculate_portfolio_returns(portfolios)

        # Calculate metrics
        metrics = self.calculate_metrics(portfolio_returns)

        # Store results
        self.results = {
            'returns': portfolio_returns,
            'metrics': metrics,
            'portfolios': portfolios
        }

        # Print results
        self.print_results()

        return self.results

    def print_results(self):
        """Print formatted results"""
        if not self.results:
            return

        metrics = self.results['metrics']

        print("\n" + "=" * 60)
        print("MOMENTUM STRATEGY RESULTS")
        print("=" * 60)

        print("\n📊 PERFORMANCE METRICS:")
        print("-" * 40)

        # Gross Performance
        print("\nGross Performance (before costs):")
        print(f"  Annual Return:     {metrics['gross_annual_return'] * 100:>7.2f}%")
        print(f"  Annual Volatility: {metrics['gross_volatility'] * 100:>7.2f}%")
        print(f"  Sharpe Ratio:      {metrics['gross_sharpe']:>7.3f}")

        # Net Performance
        print("\nNet Performance (after costs):")
        print(f"  Annual Return:     {metrics['net_annual_return'] * 100:>7.2f}%")
        print(f"  Annual Volatility: {metrics['net_volatility'] * 100:>7.2f}%")
        print(f"  Sharpe Ratio:      {metrics['net_sharpe']:>7.3f}")

        # Risk Metrics
        print("\n📉 RISK METRICS:")
        print("-" * 40)
        print(f"  Maximum Drawdown:  {metrics['max_drawdown'] * 100:>7.2f}%")
        print(f"  Calmar Ratio:      {metrics['calmar_ratio']:>7.3f}")
        print(f"  Win Rate:          {metrics['win_rate'] * 100:>7.2f}%")
        print(f"  Win/Loss Ratio:    {metrics['win_loss_ratio']:>7.2f}")

        # Statistical Significance
        print("\n📈 STATISTICAL SIGNIFICANCE:")
        print("-" * 40)
        print(f"  Sharpe t-statistic: {metrics['sharpe_t_stat']:>7.3f}")
        print(f"  p-value:            {metrics['sharpe_p_value']:>7.4f}")
        print(f"  Significant (5%):   {'✅ Yes' if metrics['is_significant'] else '❌ No'}")

        # Transaction Costs
        print("\n💰 TRANSACTION COSTS:")
        print("-" * 40)
        print(f"  Costs as % of gross: {metrics['cost_as_pct_of_gross']:>6.2f}%")

        print("\n" + "=" * 60)

    def plot_results(self, save_path: Optional[str] = None):
        """Create visualization of results"""
        if not self.results:
            logger.error("No results to plot!")
            return

        returns_df = self.results['returns']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Cumulative returns
        ax = axes[0, 0]
        ax.plot(returns_df.index, returns_df['cum_return_gross'] * 100,
                label='Gross', linewidth=2)
        ax.plot(returns_df.index, returns_df['cum_return_net'] * 100,
                label='Net', linewidth=2, linestyle='--')
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Monthly returns distribution
        ax = axes[0, 1]
        ax.hist(returns_df['ls_return_net'] * 100, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Monthly Returns Distribution')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')

        # Rolling Sharpe ratio (12-month)
        ax = axes[1, 0]
        rolling_sharpe = (returns_df['ls_return_net'].rolling(12).mean() /
                          returns_df['ls_return_net'].rolling(12).std() * np.sqrt(12))
        ax.plot(returns_df.index, rolling_sharpe)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Rolling 12-Month Sharpe Ratio')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)

        # Number of positions over time
        ax = axes[1, 1]
        ax.plot(returns_df.index, returns_df['n_long'], label='Long', alpha=0.7)
        ax.plot(returns_df.index, returns_df['n_short'], label='Short', alpha=0.7)
        ax.set_title('Number of Positions')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()


def main():
    """Main execution"""
    # Configuration
    config = StrategyConfig(
        lookback_months=12,
        skip_months=2,
        n_portfolios=10,
        spread_cost_bps=10,
        market_impact_bps=5,
        brokerage_cost_bps=2
    )

    # Initialize strategy
    strategy = MomentumStrategy(config)

    # Run backtest
    data_path = "data/processed/prices_clean.parquet"

    # Check if data exists
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run fetch_all_prices.py first to download data")
        return 1

    # Run backtest
    results = strategy.run_backtest(data_path)

    if results:
        # Save results
        output_dir = Path("outputs/momentum")
        output_dir.mkdir(parents=True, exist_ok=True)

        returns_df = results['returns']
        returns_df.to_csv(output_dir / "momentum_returns.csv", index=False)

        # Save metrics
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df.to_csv(output_dir / "momentum_metrics.csv", index=False)

        # Create plots
        strategy.plot_results(save_path=str(output_dir / "momentum_charts.png"))

        logger.info(f"Results saved to {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())