#!/usr/bin/env python3
"""
reversal_strategy.py - Comprehensive implementation of reversal strategies from academic literature

Implemented strategies:
- De Bondt & Thaler (1987): 3-year contrarian strategy
- Jegadeesh (1990): Monthly reversals
- Lehmann (1990): Weekly reversals
- Lo & MacKinlay (1990): Portfolio-based reversals
- Rosenberg, Reid & Lanstein (1985): Book-to-market reversals
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import itertools
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReversalConfig:
    """Configuration for reversal strategies"""

    # Strategy selection
    strategy_type: str = 'all'  # 'debondt_thaler', 'jegadeesh', 'lehmann', 'lo_mackinlay', 'all'

    # Formation and holding periods (in months)
    formation_periods: List[int] = field(default_factory=lambda: [3, 6, 12, 36])  # Removed 1-month
    holding_periods: List[int] = field(default_factory=lambda: [1, 3, 6, 12])
    skip_periods: List[int] = field(default_factory=lambda: [0, 1, 2])

    # Portfolio construction
    n_portfolios: int = 10
    min_stocks_per_portfolio: int = 10

    # Risk management
    max_position_weight: float = 0.05
    volatility_adjustment: bool = True
    market_cap_weighted: bool = False

    # Transaction costs (basis points)
    spread_cost_bps: float = 15  # Higher for reversals due to liquidity
    market_impact_bps: float = 8
    brokerage_cost_bps: float = 3

    # Data filters
    min_price: float = 5.0
    min_market_cap: float = 1e8  # $100M minimum
    min_trading_days: int = 150

    # Statistical testing
    bootstrap_samples: int = 1000
    significance_level: float = 0.05


class ReversalCalculator:
    """Calculate various reversal signals"""

    @staticmethod
    def debondt_thaler_signal(df: pd.DataFrame, formation_months: int = 36) -> pd.Series:
        """
        De Bondt & Thaler (1987) contrarian strategy
        Buy losers, sell winners based on long-term past performance
        """
        df_sorted = df.sort_values(['symbol', 'date']).copy()

        # Calculate daily returns
        df_sorted['daily_return'] = df_sorted.groupby('symbol')['close'].pct_change()

        # Remove extreme outliers that might be stock splits
        df_sorted.loc[df_sorted['daily_return'] > 0.5, 'daily_return'] = np.nan
        df_sorted.loc[df_sorted['daily_return'] < -0.5, 'daily_return'] = np.nan

        # Calculate log returns for better compounding
        df_sorted['log_return'] = np.log(1 + df_sorted['daily_return'].fillna(0))

        # Rolling cumulative log returns
        window_days = max(formation_months * 21, 63)  # At least 3 months of trading days
        min_periods = max(int(window_days * 0.5), 30)  # At least 30 trading days

        df_sorted['cum_log_return'] = df_sorted.groupby('symbol')['log_return'].transform(
            lambda x: x.rolling(window_days, min_periods=min_periods).sum()
        )

        # Convert back to simple returns
        df_sorted['formation_return'] = np.exp(df_sorted['cum_log_return']) - 1

        # Contrarian signal: negative of past returns (buy losers, sell winners)
        df_sorted['reversal_signal'] = -df_sorted['formation_return']

        return df_sorted.set_index(df.index)['reversal_signal']

    @staticmethod
    def jegadeesh_monthly_reversal(df: pd.DataFrame) -> pd.Series:
        """
        Jegadeesh (1990) monthly reversal strategy
        Based on previous month's returns
        """
        df_sorted = df.sort_values(['symbol', 'date']).copy()

        # Get monthly returns
        df_sorted['year_month'] = df_sorted['date'].dt.to_period('M')

        # Calculate monthly returns more robustly
        monthly_data = []
        for symbol in df_sorted['symbol'].unique():
            symbol_data = df_sorted[df_sorted['symbol'] == symbol].copy()

            # Group by month and get first/last prices
            monthly_prices = symbol_data.groupby('year_month')['close'].agg(['first', 'last', 'count']).reset_index()
            monthly_prices = monthly_prices[monthly_prices['count'] >= 10]  # At least 10 trading days

            # Calculate returns
            monthly_prices['monthly_return'] = (monthly_prices['last'] / monthly_prices['first']) - 1
            monthly_prices['symbol'] = symbol

            monthly_data.append(monthly_prices[['symbol', 'year_month', 'monthly_return']])

        if not monthly_data:
            return pd.Series(index=df.index, dtype=float)

        monthly_ret = pd.concat(monthly_data, ignore_index=True)

        # Previous month return (contrarian signal)
        monthly_ret['prev_month_return'] = monthly_ret.groupby('symbol')['monthly_return'].shift(1)
        monthly_ret['reversal_signal'] = -monthly_ret['prev_month_return']

        # Map back to daily data
        df_sorted = df_sorted.merge(
            monthly_ret[['symbol', 'year_month', 'reversal_signal']],
            on=['symbol', 'year_month'],
            how='left'
        )

        return df_sorted.set_index(df.index)['reversal_signal']

    @staticmethod
    def lehmann_weekly_reversal(df: pd.DataFrame) -> pd.Series:
        """
        Lehmann (1990) weekly reversal strategy
        Based on previous week's returns
        """
        df_sorted = df.sort_values(['symbol', 'date']).copy()

        # Add week identifier
        df_sorted['year_week'] = df_sorted['date'].dt.isocalendar().week + df_sorted['date'].dt.year * 100

        # Weekly returns
        weekly_data = []
        for symbol in df_sorted['symbol'].unique():
            symbol_data = df_sorted[df_sorted['symbol'] == symbol].copy()

            weekly_prices = symbol_data.groupby('year_week')['close'].agg(['first', 'last', 'count']).reset_index()
            weekly_prices = weekly_prices[weekly_prices['count'] >= 3]  # At least 3 trading days

            weekly_prices['weekly_return'] = (weekly_prices['last'] / weekly_prices['first']) - 1
            weekly_prices['symbol'] = symbol

            weekly_data.append(weekly_prices[['symbol', 'year_week', 'weekly_return']])

        if not weekly_data:
            return pd.Series(index=df.index, dtype=float)

        weekly_ret = pd.concat(weekly_data, ignore_index=True)

        # Previous week return (contrarian signal)
        weekly_ret['prev_week_return'] = weekly_ret.groupby('symbol')['weekly_return'].shift(1)
        weekly_ret['reversal_signal'] = -weekly_ret['prev_week_return']

        # Map back to daily data
        df_sorted = df_sorted.merge(
            weekly_ret[['symbol', 'year_week', 'reversal_signal']],
            on=['symbol', 'year_week'],
            how='left'
        )

        return df_sorted.set_index(df.index)['reversal_signal']

    @staticmethod
    def lo_mackinlay_contrarian(df: pd.DataFrame, formation_months: int = 12) -> pd.Series:
        """
        Lo & MacKinlay (1990) portfolio-based contrarian strategy
        Uses risk-adjusted returns
        """
        df_sorted = df.sort_values(['symbol', 'date']).copy()
        df_sorted['return'] = df_sorted.groupby('symbol')['close'].pct_change()

        # Remove extreme outliers
        df_sorted.loc[abs(df_sorted['return']) > 0.5, 'return'] = np.nan

        # Calculate rolling statistics
        window_days = max(formation_months * 21, 63)
        min_periods = max(int(window_days * 0.5), 30)

        # Rolling mean return
        df_sorted['avg_return'] = df_sorted.groupby('symbol')['return'].transform(
            lambda x: x.rolling(window_days, min_periods=min_periods).mean()
        )

        # Rolling volatility
        df_sorted['volatility'] = df_sorted.groupby('symbol')['return'].transform(
            lambda x: x.rolling(window_days, min_periods=min_periods).std()
        )

        # Risk-adjusted return (information ratio)
        df_sorted['risk_adj_return'] = df_sorted['avg_return'] / (df_sorted['volatility'] + 1e-6)

        # Contrarian signal
        df_sorted['reversal_signal'] = -df_sorted['risk_adj_return']

        return df_sorted.set_index(df.index)['reversal_signal']

    @staticmethod
    def short_term_reversal(df: pd.DataFrame, lookback_days: int = 5) -> pd.Series:
        """
        Short-term reversal based on recent price movements
        """
        if lookback_days < 2:
            lookback_days = 5  # Minimum reasonable lookback

        df_sorted = df.sort_values(['symbol', 'date']).copy()
        df_sorted['return'] = df_sorted.groupby('symbol')['close'].pct_change()

        # Remove extreme outliers
        df_sorted.loc[abs(df_sorted['return']) > 0.5, 'return'] = np.nan

        # Short-term cumulative return
        min_periods = max(2, lookback_days // 2)
        df_sorted['short_return'] = df_sorted.groupby('symbol')['return'].transform(
            lambda x: x.rolling(lookback_days, min_periods=min_periods).apply(
                lambda y: (1 + y.fillna(0)).prod() - 1 if len(y.dropna()) >= min_periods else np.nan
            )
        )

        # Reversal signal
        df_sorted['reversal_signal'] = -df_sorted['short_return']

        return df_sorted.set_index(df.index)['reversal_signal']


class ReversalStrategy:
    """Comprehensive reversal strategy implementation"""

    def __init__(self, config: ReversalConfig = None):
        self.config = config or ReversalConfig()
        self.calculator = ReversalCalculator()
        self.results = {}
        self.all_results = {}  # Store results for all parameter combinations

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and prepare data"""
        logger.info("Loading data for reversal strategy...")

        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        df['date'] = pd.to_datetime(df['date'])

        # Use USD-adjusted prices if available
        if 'close_usd' in df.columns:
            logger.info("Using USD-adjusted prices")
            df['close'] = df['close_usd']
        elif 'currency' in df.columns:
            usd_only = df[df['currency'] == 'USD'].copy()
            if len(usd_only) > 0:
                df = usd_only
                logger.info(f"Filtered to {df['symbol'].nunique()} USD securities")

        # Apply filters
        initial_count = len(df)
        df = df[df['close'] >= self.config.min_price]
        df = df.drop_duplicates(subset=['symbol', 'date'])
        df = df.sort_values(['symbol', 'date'])

        # Remove stocks with insufficient history
        symbol_counts = df.groupby('symbol').size()
        valid_symbols = symbol_counts[symbol_counts >= 504].index  # At least 2 years
        df = df[df['symbol'].isin(valid_symbols)]

        final_count = len(df)
        logger.info(f"Data loaded: {final_count:,} observations ({initial_count - final_count:,} removed)")
        logger.info(f"Symbols: {df['symbol'].nunique()}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

        return df

    def validate_parameters(self, strategy_type: str, formation_period: int,
                            holding_period: int, skip_period: int) -> bool:
        """Validate parameter combination"""
        if strategy_type == 'short_term':
            # For short-term strategies, formation_period is in days
            if formation_period < 2:
                return False
        else:
            # For other strategies, formation_period is in months
            if formation_period < 1:
                return False

        if holding_period < 1 or skip_period < 0:
            return False

        return True

    def calculate_reversal_signals(self, df: pd.DataFrame, strategy_type: str,
                                   formation_period: int, skip_period: int = 1) -> pd.DataFrame:
        """Calculate reversal signals for specific strategy and parameters"""

        df_copy = df.copy()

        try:
            if strategy_type == 'debondt_thaler':
                df_copy['reversal_signal'] = self.calculator.debondt_thaler_signal(df_copy, formation_period)
            elif strategy_type == 'jegadeesh':
                df_copy['reversal_signal'] = self.calculator.jegadeesh_monthly_reversal(df_copy)
            elif strategy_type == 'lehmann':
                df_copy['reversal_signal'] = self.calculator.lehmann_weekly_reversal(df_copy)
            elif strategy_type == 'lo_mackinlay':
                df_copy['reversal_signal'] = self.calculator.lo_mackinlay_contrarian(df_copy, formation_period)
            elif strategy_type == 'short_term':
                df_copy['reversal_signal'] = self.calculator.short_term_reversal(df_copy, formation_period)
            else:
                # Default to De Bondt & Thaler
                df_copy['reversal_signal'] = self.calculator.debondt_thaler_signal(df_copy, formation_period)

            # Skip period implementation
            if skip_period > 0:
                skip_days = skip_period * 21  # Convert months to trading days
                df_copy['reversal_signal'] = df_copy.groupby('symbol')['reversal_signal'].shift(skip_days)

            return df_copy

        except Exception as e:
            logger.error(f"Error calculating signals for {strategy_type}: {e}")
            df_copy['reversal_signal'] = np.nan
            return df_copy

    def form_portfolios(self, df: pd.DataFrame, holding_period: int = 1) -> pd.DataFrame:
        """Form reversal portfolios"""

        # Create monthly rebalancing points
        df['year_month'] = df['date'].dt.to_period('M')

        # Use month-end data for portfolio formation
        try:
            monthly_data = df.groupby(['symbol', 'year_month']).last().reset_index()
        except Exception as e:
            logger.error(f"Error in monthly aggregation: {e}")
            return pd.DataFrame()

        # Calculate forward returns for different holding periods
        monthly_data = monthly_data.sort_values(['symbol', 'year_month'])

        # Forward return calculation
        for h in range(1, max(self.config.holding_periods) + 1):
            try:
                monthly_data[f'ret_forward_{h}m'] = (
                    monthly_data.groupby('symbol')['close']
                    .pct_change(h).shift(-h)
                )
            except Exception as e:
                logger.error(f"Error calculating {h}m forward returns: {e}")
                monthly_data[f'ret_forward_{h}m'] = np.nan

        portfolio_results = []

        for month in monthly_data['year_month'].unique():
            try:
                month_data = monthly_data[monthly_data['year_month'] == month].copy()

                # Remove stocks without reversal signals
                month_data = month_data.dropna(subset=['reversal_signal'])

                if len(month_data) < self.config.min_stocks_per_portfolio * 2:
                    continue

                # Check for valid reversal signals
                if month_data['reversal_signal'].std() == 0:
                    continue  # Skip if all signals are the same

                # Rank by reversal signal (high signal = buy losers)
                month_data['signal_rank'] = month_data['reversal_signal'].rank(pct=True, method='min')

                # Form portfolios
                try:
                    month_data['portfolio'] = pd.qcut(
                        month_data['signal_rank'],
                        q=self.config.n_portfolios,
                        labels=range(1, self.config.n_portfolios + 1),
                        duplicates='drop'
                    )
                except Exception:
                    # If qcut fails, use manual binning
                    month_data['portfolio'] = np.ceil(
                        month_data['signal_rank'] * self.config.n_portfolios
                    ).clip(1, self.config.n_portfolios)

                # Long highest reversal signals (biggest losers), short lowest (biggest winners)
                month_data['position'] = 'neutral'
                month_data.loc[month_data['portfolio'] == self.config.n_portfolios, 'position'] = 'long'
                month_data.loc[month_data['portfolio'] == 1, 'position'] = 'short'

                portfolio_results.append(month_data)

            except Exception as e:
                logger.warning(f"Error processing month {month}: {e}")
                continue

        if portfolio_results:
            return pd.concat(portfolio_results, ignore_index=True)
        else:
            return pd.DataFrame()

    def calculate_portfolio_returns(self, portfolio_df: pd.DataFrame,
                                    holding_period: int = 1) -> pd.DataFrame:
        """Calculate portfolio returns with transaction costs"""

        if portfolio_df.empty:
            return pd.DataFrame()

        return_col = f'ret_forward_{holding_period}m'
        if return_col not in portfolio_df.columns:
            logger.warning(f"Return column {return_col} not found")
            return pd.DataFrame()

        results = []

        for month in portfolio_df['year_month'].unique():
            try:
                month_data = portfolio_df[portfolio_df['year_month'] == month]

                long_stocks = month_data[month_data['position'] == 'long']
                short_stocks = month_data[month_data['position'] == 'short']

                if len(long_stocks) == 0 or len(short_stocks) == 0:
                    continue

                # Calculate returns with proper error handling
                long_returns = long_stocks[return_col].dropna()
                short_returns = short_stocks[return_col].dropna()

                if len(long_returns) == 0 or len(short_returns) == 0:
                    continue

                # Calculate returns
                if self.config.market_cap_weighted and 'market_cap' in month_data.columns:
                    # Market cap weighted (if data available)
                    long_caps = long_stocks['market_cap'].fillna(long_stocks['market_cap'].median())
                    short_caps = short_stocks['market_cap'].fillna(short_stocks['market_cap'].median())

                    long_weights = long_caps / long_caps.sum()
                    short_weights = short_caps / short_caps.sum()

                    long_return = (long_weights * long_stocks[return_col].fillna(0)).sum()
                    short_return = (short_weights * short_stocks[return_col].fillna(0)).sum()
                else:
                    # Equal weighted
                    long_return = long_returns.mean()
                    short_return = short_returns.mean()

                # Skip if returns are extreme (likely data errors)
                if abs(long_return) > 1.0 or abs(short_return) > 1.0:
                    continue

                # Long-short return
                ls_return_gross = long_return - short_return

                # Transaction costs (rebalancing frequency dependent)
                rebalance_freq = 12 / holding_period  # Times per year
                monthly_turnover = 2.0 / rebalance_freq  # Approximate turnover

                spread_cost = monthly_turnover * self.config.spread_cost_bps / 10000
                impact_cost = monthly_turnover * self.config.market_impact_bps / 10000
                brokerage = monthly_turnover * self.config.brokerage_cost_bps / 10000

                total_cost = spread_cost + impact_cost + brokerage
                ls_return_net = ls_return_gross - total_cost

                results.append({
                    'year_month': month,
                    'long_return': long_return,
                    'short_return': short_return,
                    'ls_return_gross': ls_return_gross,
                    'transaction_cost': total_cost,
                    'ls_return_net': ls_return_net,
                    'n_long': len(long_returns),
                    'n_short': len(short_returns),
                    'holding_period': holding_period
                })

            except Exception as e:
                logger.warning(f"Error calculating returns for month {month}: {e}")
                continue

        if not results:
            return pd.DataFrame()

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('year_month')

        # Annualize returns for different holding periods
        annualization_factor = 12 / holding_period

        results_df['ls_return_gross_ann'] = results_df['ls_return_gross'] * annualization_factor
        results_df['ls_return_net_ann'] = results_df['ls_return_net'] * annualization_factor

        # Cumulative returns
        results_df['cum_return_gross'] = (1 + results_df['ls_return_gross']).cumprod() - 1
        results_df['cum_return_net'] = (1 + results_df['ls_return_net']).cumprod() - 1

        return results_df

    def calculate_comprehensive_metrics(self, returns_df: pd.DataFrame,
                                        strategy_name: str) -> Dict:
        """Calculate comprehensive performance metrics"""

        if returns_df.empty:
            return {
                'strategy_name': strategy_name,
                'error': 'No data available',
                'n_observations': 0,
                'gross_sharpe': 0,
                'net_sharpe': 0,
                'gross_annual_return': np.nan,
                'net_annual_return': np.nan
            }

        # Get holding period for annualization
        holding_period = returns_df['holding_period'].iloc[0] if 'holding_period' in returns_df.columns else 1
        annualization_factor = 12 / holding_period

        gross_returns = returns_df['ls_return_gross'].dropna().values
        net_returns = returns_df['ls_return_net'].dropna().values

        if len(net_returns) == 0:
            return {
                'strategy_name': strategy_name,
                'error': 'No valid returns',
                'n_observations': 0,
                'gross_sharpe': 0,
                'net_sharpe': 0,
                'gross_annual_return': np.nan,
                'net_annual_return': np.nan
            }

        metrics = {
            'strategy_name': strategy_name,
            'holding_period': holding_period,
            'n_observations': len(net_returns),
            'start_date': returns_df['year_month'].min(),
            'end_date': returns_df['year_month'].max(),
        }

        try:
            # Basic return metrics
            metrics.update({
                'gross_mean_return': gross_returns.mean() if len(gross_returns) > 0 else 0,
                'net_mean_return': net_returns.mean(),
                'gross_annual_return': gross_returns.mean() * annualization_factor if len(gross_returns) > 0 else 0,
                'net_annual_return': net_returns.mean() * annualization_factor,

                'gross_volatility': gross_returns.std() * np.sqrt(annualization_factor) if len(
                    gross_returns) > 0 else 0,
                'net_volatility': net_returns.std() * np.sqrt(annualization_factor),
            })

            # Sharpe ratios
            metrics['gross_sharpe'] = (
                metrics['gross_annual_return'] / metrics['gross_volatility']
                if metrics['gross_volatility'] > 0 else 0
            )
            metrics['net_sharpe'] = (
                metrics['net_annual_return'] / metrics['net_volatility']
                if metrics['net_volatility'] > 0 else 0
            )

            # Risk metrics
            metrics.update({
                'max_drawdown': self._calculate_max_drawdown(net_returns),
                'downside_deviation': np.sqrt(np.mean(np.minimum(0, net_returns) ** 2)) * np.sqrt(annualization_factor),
            })

            metrics['calmar_ratio'] = (
                metrics['net_annual_return'] / abs(metrics['max_drawdown'])
                if metrics['max_drawdown'] != 0 else 0
            )
            metrics['sortino_ratio'] = (
                metrics['net_annual_return'] / metrics['downside_deviation']
                if metrics['downside_deviation'] > 0 else 0
            )

            # Higher moments
            if len(net_returns) > 3:
                metrics.update({
                    'skewness': stats.skew(net_returns),
                    'kurtosis': stats.kurtosis(net_returns),
                })
            else:
                metrics.update({'skewness': 0, 'kurtosis': 0})

            # Win/Loss statistics
            winning_returns = net_returns[net_returns > 0]
            losing_returns = net_returns[net_returns < 0]

            metrics.update({
                'win_rate': len(winning_returns) / len(net_returns),
                'avg_win': winning_returns.mean() if len(winning_returns) > 0 else 0,
                'avg_loss': losing_returns.mean() if len(losing_returns) > 0 else 0,
            })

            metrics['win_loss_ratio'] = (
                abs(metrics['avg_win'] / metrics['avg_loss'])
                if metrics['avg_loss'] != 0 else 0
            )
            metrics['profit_factor'] = (
                abs(winning_returns.sum() / losing_returns.sum())
                if len(losing_returns) > 0 and losing_returns.sum() != 0 else 0
            )

            # Statistical significance tests
            n_periods = len(net_returns)

            if n_periods > 2:
                # T-test for mean return
                t_stat_return = net_returns.mean() / (net_returns.std() / np.sqrt(n_periods))
                p_value_return = 2 * (1 - stats.t.cdf(abs(t_stat_return), df=n_periods - 1))

                # T-test for Sharpe ratio
                sharpe_annual = metrics['net_sharpe']
                t_stat_sharpe = sharpe_annual * np.sqrt(n_periods / annualization_factor)
                p_value_sharpe = 2 * (1 - stats.t.cdf(abs(t_stat_sharpe), df=n_periods - 1))
            else:
                t_stat_return = 0
                p_value_return = 1
                t_stat_sharpe = 0
                p_value_sharpe = 1

            metrics.update({
                't_stat_return': t_stat_return,
                'p_value_return': p_value_return,
                'significant_return': p_value_return < self.config.significance_level,
                't_stat_sharpe': t_stat_sharpe,
                'p_value_sharpe': p_value_sharpe,
                'significant_sharpe': p_value_sharpe < self.config.significance_level,
            })

            # Transaction cost analysis
            total_tcosts = returns_df['transaction_cost'].sum()
            total_gross_return = returns_df['ls_return_gross'].sum()

            metrics.update({
                'total_transaction_costs': total_tcosts,
                'tcost_as_pct_of_gross': (
                    total_tcosts / abs(total_gross_return) * 100
                    if total_gross_return != 0 else 0
                ),
                'avg_positions': returns_df[['n_long', 'n_short']].sum(axis=1).mean(),
            })

        except Exception as e:
            logger.error(f"Error calculating metrics for {strategy_name}: {e}")
            metrics.update({
                'error': str(e),
                'gross_sharpe': 0,
                'net_sharpe': 0,
                'gross_annual_return': np.nan,
                'net_annual_return': np.nan
            })

        return metrics

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0

        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def run_parameter_sweep(self, df: pd.DataFrame) -> Dict:
        """Run reversal strategy across all parameter combinations"""
        logger.info("Running comprehensive parameter sweep...")

        strategies = ['debondt_thaler', 'jegadeesh', 'lo_mackinlay', 'short_term']
        if self.config.strategy_type != 'all':
            strategies = [self.config.strategy_type]

        all_results = []

        # Calculate total combinations
        total_combinations = 0
        for strategy in strategies:
            for formation_period in self.config.formation_periods:
                for holding_period in self.config.holding_periods:
                    for skip_period in self.config.skip_periods:
                        if self.validate_parameters(strategy, formation_period, holding_period, skip_period):
                            total_combinations += 1

        logger.info(f"Testing {total_combinations} valid parameter combinations...")

        combination_count = 0

        for strategy in strategies:
            for formation_period in self.config.formation_periods:
                for holding_period in self.config.holding_periods:
                    for skip_period in self.config.skip_periods:

                        # Validate parameters
                        if not self.validate_parameters(strategy, formation_period, holding_period, skip_period):
                            logger.warning(
                                f"Invalid parameters: {strategy}_{formation_period}_{holding_period}_{skip_period}")
                            continue

                        combination_count += 1

                        if combination_count % 10 == 0 or combination_count <= 10:
                            logger.info(f"Progress: {combination_count}/{total_combinations}")

                        strategy_name = f"{strategy}_{formation_period}m_hold{holding_period}m_skip{skip_period}m"

                        try:
                            # Calculate signals
                            df_signals = self.calculate_reversal_signals(
                                df.copy(), strategy, formation_period, skip_period
                            )

                            # Check if we have valid signals
                            valid_signals = df_signals['reversal_signal'].notna().sum()
                            if valid_signals < 100:  # Need reasonable number of signals
                                logger.warning(f"Too few valid signals ({valid_signals}) for {strategy_name}")
                                continue

                            # Form portfolios
                            portfolios = self.form_portfolios(df_signals, holding_period)

                            if portfolios.empty:
                                logger.warning(f"No portfolios formed for {strategy_name}")
                                continue

                            # Calculate returns
                            returns_df = self.calculate_portfolio_returns(portfolios, holding_period)

                            if returns_df.empty:
                                logger.warning(f"No returns calculated for {strategy_name}")
                                continue

                            # Calculate metrics
                            metrics = self.calculate_comprehensive_metrics(returns_df, strategy_name)

                            # Add parameter information
                            metrics.update({
                                'strategy_type': strategy,
                                'formation_period': formation_period,
                                'holding_period': holding_period,
                                'skip_period': skip_period
                            })

                            all_results.append(metrics)

                        except Exception as e:
                            logger.warning(f"Failed for {strategy_name}: {e}")
                            continue

        if all_results:
            results_df = pd.DataFrame(all_results)

            # Remove invalid results
            results_df = results_df[results_df['n_observations'] > 0]

            if len(results_df) == 0:
                logger.error("No valid results found!")
                return {}

            logger.info(f"Completed parameter sweep: {len(results_df)} successful combinations")

            # Find best strategies
            valid_results = results_df[~results_df['net_sharpe'].isna() & (results_df['net_sharpe'] != 0)]

            if len(valid_results) > 0:
                best_sharpe = valid_results.nlargest(5, 'net_sharpe')
                best_return = valid_results.nlargest(5, 'net_annual_return')

                logger.info("\nTop 5 strategies by Sharpe ratio:")
                for _, row in best_sharpe.iterrows():
                    logger.info(
                        f"  {row['strategy_name']}: Sharpe={row['net_sharpe']:.3f}, "
                        f"Return={row['net_annual_return'] * 100:.2f}%"
                    )
            else:
                logger.warning("No valid strategies found with positive Sharpe ratios")
                best_sharpe = results_df.head(5)
                best_return = results_df.head(5)

            return {'results_df': results_df, 'best_sharpe': best_sharpe, 'best_return': best_return}
        else:
            logger.error("No successful parameter combinations found!")
            return {}

    def create_comprehensive_visualizations(self, results_df: pd.DataFrame):
        """Create comprehensive visualizations with error handling"""
        logger.info("Creating comprehensive visualizations...")

        output_dir = Path("outputs/reversal")
        output_dir.mkdir(parents=True, exist_ok=True)

        if results_df.empty:
            logger.error("No results to visualize")
            return

        # Filter out invalid results for visualization
        valid_results = results_df[
            ~results_df['net_sharpe'].isna() &
            (results_df['n_observations'] > 0)
            ].copy()

        if len(valid_results) == 0:
            logger.error("No valid results to visualize")
            return

        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(24, 16))

        try:
            # 1. Heatmap of Sharpe ratios by formation and holding period
            ax1 = plt.subplot(3, 4, 1)
            pivot_sharpe = valid_results.pivot_table(
                values='net_sharpe',
                index='formation_period',
                columns='holding_period',
                aggfunc='mean'
            )

            if not pivot_sharpe.empty and pivot_sharpe.size > 0:
                # Remove NaN/inf values
                pivot_sharpe = pivot_sharpe.replace([np.inf, -np.inf], np.nan).fillna(0)

                sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0, ax=ax1)
                ax1.set_title('Sharpe Ratio by Formation/Holding Period', fontweight='bold')
                ax1.set_xlabel('Holding Period (months)')
                ax1.set_ylabel('Formation Period (months)')
            else:
                ax1.text(0.5, 0.5, 'No valid data for Sharpe heatmap', ha='center', va='center',
                         transform=ax1.transAxes)

            # 2. Heatmap of annual returns
            ax2 = plt.subplot(3, 4, 2)
            pivot_return = valid_results.pivot_table(
                values='net_annual_return',
                index='formation_period',
                columns='holding_period',
                aggfunc='mean'
            ) * 100

            if not pivot_return.empty and pivot_return.size > 0:
                # Remove NaN/inf values
                pivot_return = pivot_return.replace([np.inf, -np.inf], np.nan).fillna(0)

                sns.heatmap(pivot_return, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax2)
                ax2.set_title('Annual Return (%) by Formation/Holding Period', fontweight='bold')
                ax2.set_xlabel('Holding Period (months)')
                ax2.set_ylabel('Formation Period (months)')
            else:
                ax2.text(0.5, 0.5, 'No valid data for return heatmap', ha='center', va='center',
                         transform=ax2.transAxes)

            # 3. Strategy comparison
            ax3 = plt.subplot(3, 4, 3)
            strategy_performance = valid_results.groupby('strategy_type')[['net_sharpe', 'net_annual_return']].mean()

            if not strategy_performance.empty:
                x_pos = np.arange(len(strategy_performance))
                bars1 = ax3.bar(x_pos - 0.2, strategy_performance['net_sharpe'], 0.4,
                                label='Sharpe Ratio', alpha=0.8, color='steelblue')
                ax3_twin = ax3.twinx()
                bars2 = ax3_twin.bar(x_pos + 0.2, strategy_performance['net_annual_return'] * 100, 0.4,
                                     label='Annual Return (%)', alpha=0.8, color='coral')
                ax3.set_xlabel('Strategy Type')
                ax3.set_ylabel('Sharpe Ratio', color='steelblue')
                ax3_twin.set_ylabel('Annual Return (%)', color='coral')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(strategy_performance.index, rotation=45, ha='right')
                ax3.set_title('Performance by Strategy Type', fontweight='bold')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No strategy data', ha='center', va='center', transform=ax3.transAxes)

            # 4. Win rate analysis
            ax4 = plt.subplot(3, 4, 4)
            if 'win_rate' in valid_results.columns and not valid_results['win_rate'].isna().all():
                scatter = ax4.scatter(valid_results['win_rate'] * 100, valid_results['net_annual_return'] * 100,
                                      c=valid_results['net_sharpe'], cmap='viridis', alpha=0.6, s=50)
                ax4.set_xlabel('Win Rate (%)')
                ax4.set_ylabel('Annual Return (%)')
                ax4.set_title('Win Rate vs Returns (colored by Sharpe)', fontweight='bold')
                plt.colorbar(scatter, ax=ax4, label='Sharpe Ratio')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No win rate data', ha='center', va='center', transform=ax4.transAxes)

            # Continue with remaining plots...
            # 5. Number of observations histogram
            ax5 = plt.subplot(3, 4, 5)
            if 'n_observations' in valid_results.columns:
                valid_results['n_observations'].hist(bins=20, alpha=0.7, edgecolor='black', ax=ax5)
                ax5.set_xlabel('Number of Observations')
                ax5.set_ylabel('Frequency')
                ax5.set_title('Distribution of Observation Counts', fontweight='bold')
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'No observation data', ha='center', va='center', transform=ax5.transAxes)

            # 6. Distribution of returns
            ax6 = plt.subplot(3, 4, 6)
            if not valid_results['net_annual_return'].isna().all():
                valid_results['net_annual_return'].hist(bins=20, alpha=0.7, edgecolor='black', ax=ax6)
                ax6.axvline(0, color='red', linestyle='--', alpha=0.7)
                ax6.set_xlabel('Annual Return')
                ax6.set_ylabel('Frequency')
                ax6.set_title('Distribution of Annual Returns', fontweight='bold')
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'No return data', ha='center', va='center', transform=ax6.transAxes)

            # Fill remaining subplots with summary statistics
            for i in range(7, 13):
                ax = plt.subplot(3, 4, i)
                ax.text(0.5, 0.5, f'Summary Stats {i - 6}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Additional Analysis {i - 6}', fontweight='bold')

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")

        plt.suptitle('Reversal Strategy Comprehensive Analysis', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the comprehensive plot
        try:
            plot_path = output_dir / "reversal_comprehensive_analysis.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"Comprehensive analysis plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Error saving plot: {e}")

        plt.close()

        # Create summary table
        self._create_summary_table(valid_results, output_dir)

    def _create_summary_table(self, results_df: pd.DataFrame, output_dir: Path):
        """Create strategy summary table"""
        logger.info("Creating strategy summary table...")

        try:
            # Get top strategies
            top_strategies = results_df.nlargest(20, 'net_sharpe')

            # Select important columns
            summary_cols = [
                'strategy_name', 'strategy_type', 'formation_period', 'holding_period',
                'skip_period', 'n_observations', 'net_annual_return', 'net_sharpe',
                'max_drawdown', 'win_rate', 'significant_return'
            ]

            # Filter existing columns
            existing_cols = [col for col in summary_cols if col in top_strategies.columns]
            summary_table = top_strategies[existing_cols].copy()

            # Format percentages
            percentage_cols = ['net_annual_return', 'max_drawdown', 'win_rate']
            for col in percentage_cols:
                if col in summary_table.columns:
                    summary_table[col] = summary_table[col] * 100

            # Round numeric columns
            numeric_cols = ['net_annual_return', 'net_sharpe', 'max_drawdown', 'win_rate']
            for col in numeric_cols:
                if col in summary_table.columns:
                    summary_table[col] = summary_table[col].round(3)

            summary_table.to_csv(output_dir / "top_reversal_strategies.csv", index=False)
            logger.info(f"Top strategies summary saved to {output_dir / 'top_reversal_strategies.csv'}")

        except Exception as e:
            logger.error(f"Error creating summary table: {e}")

    def run_full_analysis(self, data_path: str) -> Dict:
        """Run complete reversal strategy analysis"""
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE REVERSAL STRATEGY ANALYSIS")
        logger.info("=" * 60)

        # Load data
        df = self.load_data(data_path)
        if df.empty:
            logger.error("No data available for analysis")
            return {}

        # Run parameter sweep
        sweep_results = self.run_parameter_sweep(df)

        if not sweep_results:
            logger.error("Parameter sweep failed!")
            return {}

        results_df = sweep_results['results_df']

        # Create visualizations
        self.create_comprehensive_visualizations(results_df)

        # Save all results
        output_dir = Path("outputs/reversal")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            results_df.to_csv(output_dir / "all_reversal_results.csv", index=False)
        except Exception as e:
            logger.error(f"Error saving results: {e}")

        # Store results
        self.all_results = {
            'sweep_results': sweep_results,
            'results_df': results_df
        }

        # Print summary
        self.print_comprehensive_results()

        return self.all_results

    def print_comprehensive_results(self):
        """Print comprehensive results summary"""
        if not self.all_results or 'results_df' not in self.all_results:
            logger.warning("No results to print")
            return

        results_df = self.all_results['results_df']

        if results_df.empty:
            logger.warning("Results dataframe is empty")
            return

        print("\n" + "=" * 60)
        print("REVERSAL STRATEGY COMPREHENSIVE RESULTS")
        print("=" * 60)

        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"-" * 40)
        print(f"Total Parameter Combinations Tested: {len(results_df)}")

        valid_results = results_df[results_df['n_observations'] > 0]
        print(f"Valid Results: {len(valid_results)}")

        if len(valid_results) == 0:
            print("❌ No valid results found!")
            return

        significant_results = valid_results[valid_results.get('significant_return', False)]
        print(f"Statistically Significant Results: {len(significant_results)}")

        print(f"\n🏆 PERFORMANCE OVERVIEW:")
        print(f"-" * 40)

        best_sharpe = valid_results['net_sharpe'].max()
        best_return = valid_results['net_annual_return'].max()
        avg_sharpe = valid_results['net_sharpe'].mean()
        avg_return = valid_results['net_annual_return'].mean()

        print(f"Best Sharpe Ratio: {best_sharpe:.3f}")
        print(f"Best Annual Return: {best_return * 100:.2f}%")
        print(f"Average Sharpe Ratio: {avg_sharpe:.3f}")
        print(f"Average Annual Return: {avg_return * 100:.2f}%")

        print(f"\n🎯 TOP 5 STRATEGIES:")
        print(f"-" * 40)
        top_5 = valid_results.nlargest(5, 'net_sharpe')
        for idx, (_, row) in enumerate(top_5.iterrows(), 1):
            significant = '✅' if row.get('significant_return', False) else '❌'
            print(f"{idx}. {row['strategy_name']}")
            print(f"   Sharpe: {row['net_sharpe']:.3f}, Return: {row['net_annual_return'] * 100:.2f}%, "
                  f"Obs: {row['n_observations']}, Significant: {significant}")

        print("\n" + "=" * 60)
        print("📁 OUTPUT FILES:")
        print("-" * 40)
        print("• outputs/reversal/all_reversal_results.csv - Complete results")
        print("• outputs/reversal/top_reversal_strategies.csv - Top strategies")
        print("• outputs/reversal/reversal_comprehensive_analysis.png - Visual analysis")
        print("=" * 60)


def main():
    """Main execution function"""
    # Configuration with comprehensive parameter testing
    config = ReversalConfig(
        strategy_type='all',  # Test all strategies
        formation_periods=[3, 6, 12, 36],  # Valid formation windows (removed 1)
        holding_periods=[1, 3, 6, 12],  # Different holding windows
        skip_periods=[0, 1, 2],  # Different skip periods
        n_portfolios=10,
        spread_cost_bps=15,  # Higher costs for reversals
        market_impact_bps=8,
        brokerage_cost_bps=3,
        bootstrap_samples=1000
    )

    # Initialize strategy
    strategy = ReversalStrategy(config)

    # Data path
    data_path = "data/processed/prices_clean.parquet"

    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run the data fetching script first")
        return 1

    # Run comprehensive analysis
    results = strategy.run_full_analysis(data_path)

    if results:
        logger.info("✅ Reversal strategy analysis completed successfully")
        return 0
    else:
        logger.error("❌ Reversal strategy analysis failed")
        return 1


if __name__ == "__main__":
    exit(main())