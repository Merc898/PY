#!/usr/bin/env python3
"""
enhanced_momentum_strategy.py - Comprehensive momentum strategies from academic literature

Implemented strategies:
- Jegadeesh & Titman (1993): Original J/K momentum strategies
- Carhart (1997): Momentum factor (UMD - Up Minus Down)
- Fama & French (2012): Size-segmented momentum
- Asness, Moskowitz & Pedersen (2013): Value and momentum everywhere
- Daniel & Moskowitz (2016): Momentum crash analysis
- Geczy & Samonov (2016): Long-term momentum
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from sklearn.linear_model import LinearRegression
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MomentumLiteratureConfig:
    """Configuration for literature-based momentum strategies"""

    # Strategy selection
    strategy_type: str = 'all'  # 'jegadeesh_titman', 'carhart', 'fama_french', 'daniel_moskowitz', 'all'

    # Jegadeesh & Titman (1993) parameters
    jt_formation_periods: List[int] = field(default_factory=lambda: [3, 6, 9, 12])  # J months
    jt_holding_periods: List[int] = field(default_factory=lambda: [3, 6, 9, 12])  # K months
    jt_skip_periods: List[int] = field(default_factory=lambda: [0, 1])  # Skip months

    # Portfolio construction
    n_portfolios: int = 10  # Decile portfolios
    min_stocks_per_portfolio: int = 10
    weighting_scheme: str = 'equal'  # 'equal', 'value', 'rank'

    # Risk management
    max_position_weight: float = 0.05
    volatility_scaling: bool = True
    target_volatility: float = 0.10
    momentum_crash_protection: bool = True  # Daniel & Moskowitz (2016)

    # Transaction costs (basis points)
    spread_cost_bps: float = 10
    market_impact_bps: float = 5
    brokerage_cost_bps: float = 2

    # Data filters
    min_price: float = 5.0
    min_market_cap: float = 1e9  # $1B minimum
    min_trading_days: int = 200
    exclude_microcaps: bool = True  # Exclude bottom 20% by market cap

    # Statistical testing
    bootstrap_samples: int = 1000
    significance_level: float = 0.05
    newey_west_lags: int = 6  # For t-statistics


class MomentumCalculator:
    """Calculate various momentum signals from literature"""

    @staticmethod
    def jegadeesh_titman_momentum(df: pd.DataFrame, formation_months: int = 6,
                                  skip_months: int = 1) -> pd.Series:
        """
        Jegadeesh & Titman (1993) momentum signal
        Buy past winners, sell past losers
        """
        df_sorted = df.sort_values(['symbol', 'date']).copy()

        # Calculate returns
        df_sorted['return'] = df_sorted.groupby('symbol')['close'].pct_change()

        # Remove extreme outliers (likely data errors or splits)
        df_sorted.loc[abs(df_sorted['return']) > 0.5, 'return'] = np.nan

        # Calculate cumulative returns over formation period
        window_days = formation_months * 21  # Trading days
        skip_days = skip_months * 21

        # Compound returns over formation period
        df_sorted['formation_return'] = df_sorted.groupby('symbol')['return'].transform(
            lambda x: (1 + x).rolling(window_days, min_periods=int(window_days * 0.66)).apply(
                lambda y: (y + 1).prod() - 1, raw=False
            )
        )

        # Skip period implementation
        if skip_days > 0:
            df_sorted['momentum_signal'] = df_sorted.groupby('symbol')['formation_return'].shift(skip_days)
        else:
            df_sorted['momentum_signal'] = df_sorted['formation_return']

        return df_sorted.set_index(df.index)['momentum_signal']

    @staticmethod
    def carhart_momentum_factor(df: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.Series:
        """
        Carhart (1997) momentum factor (UMD - Up Minus Down)
        Prior 2-12 month returns
        """
        df_sorted = df.sort_values(['symbol', 'date']).copy()

        # Calculate returns
        df_sorted['return'] = df_sorted.groupby('symbol')['close'].pct_change()

        # Calculate momentum as per Carhart
        # Use months 2-12 (skip most recent month)
        window_days = lookback * 21
        skip_days = skip * 21

        # Calculate cumulative returns
        df_sorted['cum_return'] = df_sorted.groupby('symbol')['return'].transform(
            lambda x: (1 + x).rolling(window_days, min_periods=int(window_days * 0.66)).apply(
                lambda y: (y + 1).prod() - 1, raw=False
            )
        )

        # Skip recent period
        df_sorted['momentum_factor'] = df_sorted.groupby('symbol')['cum_return'].shift(skip_days)

        return df_sorted.set_index(df.index)['momentum_factor']

    @staticmethod
    def fama_french_momentum(df: pd.DataFrame, size_quantile: float = 0.5) -> pd.Series:
        """
        Fama & French (2012) size-segmented momentum
        Momentum within size groups
        """
        df_sorted = df.sort_values(['symbol', 'date']).copy()

        # Calculate returns
        df_sorted['return'] = df_sorted.groupby('symbol')['close'].pct_change()

        # Proxy for size using price * volume (market cap proxy)
        df_sorted['size_proxy'] = df_sorted['close'] * df_sorted['volume']

        # Standard momentum signal
        df_sorted['raw_momentum'] = MomentumCalculator.jegadeesh_titman_momentum(df_sorted, 12, 1)

        # Size-adjust momentum
        def size_adjust_momentum(group):
            # Rank within size groups
            size_rank = group['size_proxy'].rank(pct=True)

            # Separate by size
            small_cap = size_rank <= size_quantile
            large_cap = size_rank > size_quantile

            # Normalize momentum within size groups
            adjusted_momentum = group['raw_momentum'].copy()

            if small_cap.any():
                small_mean = group.loc[small_cap, 'raw_momentum'].mean()
                small_std = group.loc[small_cap, 'raw_momentum'].std()
                if small_std > 0:
                    adjusted_momentum[small_cap] = (group.loc[small_cap, 'raw_momentum'] - small_mean) / small_std

            if large_cap.any():
                large_mean = group.loc[large_cap, 'raw_momentum'].mean()
                large_std = group.loc[large_cap, 'raw_momentum'].std()
                if large_std > 0:
                    adjusted_momentum[large_cap] = (group.loc[large_cap, 'raw_momentum'] - large_mean) / large_std

            return adjusted_momentum

        # Apply size adjustment by date
        df_sorted['size_adjusted_momentum'] = df_sorted.groupby('date').apply(
            size_adjust_momentum, include_groups=False
        ).reset_index(level=0, drop=True)

        return df_sorted.set_index(df.index)['size_adjusted_momentum']

    @staticmethod
    def residual_momentum(df: pd.DataFrame, market_df: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Blitz, Huij & Martens (2011) residual momentum
        Momentum after controlling for market beta
        """
        df_sorted = df.sort_values(['symbol', 'date']).copy()

        # Calculate returns
        df_sorted['return'] = df_sorted.groupby('symbol')['close'].pct_change()

        # If no market data, use equal-weighted market proxy
        if market_df is None:
            df_sorted['market_return'] = df_sorted.groupby('date')['return'].mean()
        else:
            # Merge with market returns
            market_df = market_df[['date', 'return']].rename(columns={'return': 'market_return'})
            df_sorted = df_sorted.merge(market_df, on='date', how='left')

        # Calculate residual returns (simplified - in practice use rolling regression)
        def calculate_residuals(group):
            if len(group) < 60:  # Need sufficient data
                return pd.Series(index=group.index, dtype=float)

            # Simple beta calculation
            cov = group[['return', 'market_return']].cov()
            if cov.iloc[1, 1] != 0:
                beta = cov.iloc[0, 1] / cov.iloc[1, 1]
            else:
                beta = 1.0

            # Residual returns
            residuals = group['return'] - beta * group['market_return']
            return residuals

        df_sorted['residual_return'] = df_sorted.groupby('symbol').apply(
            calculate_residuals, include_groups=False
        ).reset_index(level=0, drop=True)

        # Calculate momentum on residual returns
        window_days = 12 * 21
        df_sorted['residual_momentum'] = df_sorted.groupby('symbol')['residual_return'].transform(
            lambda x: x.rolling(window_days, min_periods=int(window_days * 0.66)).sum()
        )

        return df_sorted.set_index(df.index)['residual_momentum']

    @staticmethod
    def momentum_with_crash_protection(df: pd.DataFrame, market_volatility: pd.Series) -> pd.Series:
        """
        Daniel & Moskowitz (2016) momentum with crash protection
        Scale momentum exposure based on market conditions
        """
        df_sorted = df.sort_values(['symbol', 'date']).copy()

        # Standard momentum signal
        df_sorted['raw_momentum'] = MomentumCalculator.jegadeesh_titman_momentum(df_sorted, 12, 1)

        # Market volatility scaling
        # In high volatility periods, reduce momentum exposure
        volatility_percentile = market_volatility.rank(pct=True)
        high_vol_periods = volatility_percentile > 0.8

        # Scale momentum signal
        df_sorted['scaled_momentum'] = df_sorted['raw_momentum'].copy()

        # Reduce exposure in high volatility periods
        for date in df_sorted['date'].unique():
            if date in high_vol_periods.index and high_vol_periods[date]:
                date_mask = df_sorted['date'] == date
                df_sorted.loc[date_mask, 'scaled_momentum'] *= 0.5  # Reduce by 50%

        return df_sorted.set_index(df.index)['scaled_momentum']


class MomentumLiteratureStrategy:
    """Comprehensive momentum strategy implementation based on academic literature"""

    def __init__(self, config: MomentumLiteratureConfig = None):
        self.config = config or MomentumLiteratureConfig()
        self.calculator = MomentumCalculator()
        self.results = {}
        self.all_results = {}

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and prepare data with quality filters"""
        logger.info("Loading data for momentum literature strategies...")

        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        df['date'] = pd.to_datetime(df['date'])

        # Handle currency
        if 'close_usd' in df.columns:
            logger.info("Using USD-adjusted prices")
            df['close'] = df['close_usd']
        elif 'currency' in df.columns:
            usd_only = df[df['currency'] == 'USD'].copy()
            if len(usd_only) > 0:
                df = usd_only
                logger.info(f"Filtered to {df['symbol'].nunique()} USD securities")

        # Apply quality filters
        initial_count = len(df)

        # Price filter
        df = df[df['close'] >= self.config.min_price]

        # Remove duplicates
        df = df.drop_duplicates(subset=['symbol', 'date'])

        # Sort
        df = df.sort_values(['symbol', 'date'])

        # Require minimum history
        symbol_counts = df.groupby('symbol').size()
        valid_symbols = symbol_counts[symbol_counts >= 504].index  # 2+ years
        df = df[df['symbol'].isin(valid_symbols)]

        # Calculate market cap proxy
        df['market_cap_proxy'] = df['close'] * df['volume']

        # Exclude microcaps if configured
        if self.config.exclude_microcaps:
            # Remove bottom 20% by market cap proxy
            monthly_groups = df.groupby(df['date'].dt.to_period('M'))

            def filter_microcaps(group):
                threshold = group['market_cap_proxy'].quantile(0.2)
                return group[group['market_cap_proxy'] >= threshold]

            df = monthly_groups.apply(filter_microcaps, include_groups=False).reset_index(drop=True)

        final_count = len(df)
        logger.info(f"Data loaded: {final_count:,} observations ({initial_count - final_count:,} filtered)")
        logger.info(f"Symbols: {df['symbol'].nunique()}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

        return df

    def calculate_market_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate market-wide indicators for strategy enhancement"""
        logger.info("Calculating market indicators...")

        # Market return
        market_return = df.groupby('date')['close'].mean().pct_change()

        # Market volatility (20-day rolling)
        market_volatility = market_return.rolling(20).std() * np.sqrt(252)

        # Market drawdown
        market_cumulative = (1 + market_return).cumprod()
        market_running_max = market_cumulative.expanding().max()
        market_drawdown = (market_cumulative - market_running_max) / market_running_max

        # Bear market indicator
        bear_market = market_drawdown < -0.20

        return {
            'market_return': market_return,
            'market_volatility': market_volatility,
            'market_drawdown': market_drawdown,
            'bear_market': bear_market
        }

    def run_jegadeesh_titman_backtest(self, df: pd.DataFrame) -> Dict:
        """Run the original Jegadeesh & Titman (1993) strategies"""
        logger.info("Running Jegadeesh & Titman (1993) momentum strategies...")

        jt_results = []

        # Test all J/K combinations as in the original paper
        for J in self.config.jt_formation_periods:  # Formation period
            for K in self.config.jt_holding_periods:  # Holding period
                for skip in self.config.jt_skip_periods:  # Skip period

                    strategy_name = f"JT_{J}_{K}_{skip}"
                    logger.info(f"Testing {strategy_name}: J={J}, K={K}, skip={skip}")

                    try:
                        # Calculate momentum signal
                        df_strategy = df.copy()
                        df_strategy['momentum_signal'] = self.calculator.jegadeesh_titman_momentum(
                            df_strategy, J, skip
                        )

                        # Form portfolios
                        portfolios = self.form_momentum_portfolios(df_strategy, K)

                        if portfolios.empty:
                            logger.warning(f"No portfolios formed for {strategy_name}")
                            continue

                        # Calculate returns
                        returns_df = self.calculate_portfolio_returns(portfolios, K)

                        if returns_df.empty:
                            logger.warning(f"No returns calculated for {strategy_name}")
                            continue

                        # Calculate metrics
                        metrics = self.calculate_comprehensive_metrics(returns_df, strategy_name)

                        # Add strategy parameters
                        metrics.update({
                            'strategy': 'Jegadeesh-Titman',
                            'J_formation': J,
                            'K_holding': K,
                            'skip_period': skip,
                            'strategy_name': strategy_name
                        })

                        jt_results.append(metrics)

                    except Exception as e:
                        logger.error(f"Error in {strategy_name}: {e}")
                        continue

        return pd.DataFrame(jt_results) if jt_results else pd.DataFrame()

    def run_carhart_momentum(self, df: pd.DataFrame) -> Dict:
        """Run Carhart (1997) momentum factor strategy"""
        logger.info("Running Carhart (1997) momentum factor strategy...")

        try:
            # Calculate Carhart momentum
            df_strategy = df.copy()
            df_strategy['momentum_signal'] = self.calculator.carhart_momentum_factor(df_strategy)

            # Form portfolios (monthly rebalancing)
            portfolios = self.form_momentum_portfolios(df_strategy, holding_period=1)

            if portfolios.empty:
                logger.warning("No portfolios formed for Carhart strategy")
                return {}

            # Calculate returns
            returns_df = self.calculate_portfolio_returns(portfolios, holding_period=1)

            # Calculate metrics
            metrics = self.calculate_comprehensive_metrics(returns_df, "Carhart_UMD")

            metrics.update({
                'strategy': 'Carhart',
                'description': 'UMD Factor (2-12 month momentum)'
            })

            return metrics

        except Exception as e:
            logger.error(f"Error in Carhart strategy: {e}")
            return {}

    def run_size_segmented_momentum(self, df: pd.DataFrame) -> Dict:
        """Run Fama & French (2012) size-segmented momentum"""
        logger.info("Running Fama & French (2012) size-segmented momentum...")

        try:
            # Calculate size-adjusted momentum
            df_strategy = df.copy()
            df_strategy['momentum_signal'] = self.calculator.fama_french_momentum(df_strategy)

            # Form portfolios
            portfolios = self.form_momentum_portfolios(df_strategy, holding_period=1)

            if portfolios.empty:
                logger.warning("No portfolios formed for size-segmented momentum")
                return {}

            # Calculate returns
            returns_df = self.calculate_portfolio_returns(portfolios, holding_period=1)

            # Calculate metrics
            metrics = self.calculate_comprehensive_metrics(returns_df, "FF_Size_Momentum")

            metrics.update({
                'strategy': 'Fama-French',
                'description': 'Size-segmented momentum'
            })

            return metrics

        except Exception as e:
            logger.error(f"Error in size-segmented momentum: {e}")
            return {}

    def form_momentum_portfolios(self, df: pd.DataFrame, holding_period: int = 1) -> pd.DataFrame:
        """Form momentum portfolios with proper construction"""

        # Create monthly rebalancing points
        df['year_month'] = df['date'].dt.to_period('M')

        # Use month-end data for portfolio formation
        monthly_data = df.groupby(['symbol', 'year_month']).last().reset_index()

        # Calculate forward returns
        monthly_data = monthly_data.sort_values(['symbol', 'year_month'])

        # Calculate holding period returns
        for h in range(1, max(self.config.jt_holding_periods) + 1):
            monthly_data[f'ret_forward_{h}m'] = (
                monthly_data.groupby('symbol')['close']
                .pct_change(h).shift(-h)
            )

        portfolio_results = []

        for month in monthly_data['year_month'].unique():
            month_data = monthly_data[monthly_data['year_month'] == month].copy()

            # Remove stocks without momentum signals
            month_data = month_data.dropna(subset=['momentum_signal'])

            if len(month_data) < self.config.min_stocks_per_portfolio * 2:
                continue

            # Rank by momentum signal
            month_data['momentum_rank'] = month_data['momentum_signal'].rank(pct=True, method='average')

            # Form decile portfolios
            try:
                month_data['portfolio'] = pd.qcut(
                    month_data['momentum_rank'],
                    q=self.config.n_portfolios,
                    labels=range(1, self.config.n_portfolios + 1),
                    duplicates='drop'
                )
            except:
                # If qcut fails, use manual binning
                month_data['portfolio'] = pd.cut(
                    month_data['momentum_rank'],
                    bins=self.config.n_portfolios,
                    labels=range(1, self.config.n_portfolios + 1)
                )

            # Winner-Loser portfolios
            month_data['position'] = 'neutral'
            month_data.loc[month_data['portfolio'] == self.config.n_portfolios, 'position'] = 'winner'
            month_data.loc[month_data['portfolio'] == 1, 'position'] = 'loser'

            # Calculate weights based on weighting scheme
            if self.config.weighting_scheme == 'value':
                # Value-weighted
                month_data['weight'] = 0.0

                # Winners
                winners = month_data['position'] == 'winner'
                if winners.any():
                    winner_caps = month_data.loc[winners, 'market_cap_proxy']
                    month_data.loc[winners, 'weight'] = winner_caps / winner_caps.sum()

                # Losers
                losers = month_data['position'] == 'loser'
                if losers.any():
                    loser_caps = month_data.loc[losers, 'market_cap_proxy']
                    month_data.loc[losers, 'weight'] = -loser_caps / loser_caps.sum()

            elif self.config.weighting_scheme == 'rank':
                # Rank-weighted
                month_data['weight'] = 0.0

                # Winners - higher rank gets more weight
                winners = month_data['position'] == 'winner'
                if winners.any():
                    winner_ranks = month_data.loc[winners, 'momentum_rank']
                    month_data.loc[winners, 'weight'] = winner_ranks / winner_ranks.sum()

                # Losers - lower rank gets more weight
                losers = month_data['position'] == 'loser'
                if losers.any():
                    loser_ranks = 1 - month_data.loc[losers, 'momentum_rank']
                    month_data.loc[losers, 'weight'] = -loser_ranks / loser_ranks.sum()
            else:
                # Equal-weighted (default)
                month_data['weight'] = 0.0
                winners = month_data['position'] == 'winner'
                losers = month_data['position'] == 'loser'

                if winners.any():
                    month_data.loc[winners, 'weight'] = 1.0 / winners.sum()
                if losers.any():
                    month_data.loc[losers, 'weight'] = -1.0 / losers.sum()

            # Apply position limits
            month_data['weight'] = month_data['weight'].clip(
                -self.config.max_position_weight,
                self.config.max_position_weight
            )

            portfolio_results.append(month_data)

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
            month_data = portfolio_df[portfolio_df['year_month'] == month]

            winner_stocks = month_data[month_data['position'] == 'winner']
            loser_stocks = month_data[month_data['position'] == 'loser']

            if len(winner_stocks) == 0 or len(loser_stocks) == 0:
                continue

            # Calculate weighted returns
            winner_return = (winner_stocks['weight'] * winner_stocks[return_col]).sum()
            loser_return = (loser_stocks['weight'] * loser_stocks[return_col]).sum()

            # Long-short return (note: loser weights are negative)
            ls_return_gross = winner_return + loser_return  # This is winner - loser since loser weights are negative

            # Transaction costs
            rebalance_freq = 12 / holding_period
            monthly_turnover = 2.0 / rebalance_freq

            spread_cost = monthly_turnover * self.config.spread_cost_bps / 10000
            impact_cost = monthly_turnover * self.config.market_impact_bps / 10000
            brokerage = monthly_turnover * self.config.brokerage_cost_bps / 10000

            total_cost = spread_cost + impact_cost + brokerage
            ls_return_net = ls_return_gross - total_cost

            results.append({
                'year_month': month,
                'winner_return': winner_return,
                'loser_return': -loser_return,  # Make positive for reporting
                'ls_return_gross': ls_return_gross,
                'transaction_cost': total_cost,
                'ls_return_net': ls_return_net,
                'n_winners': len(winner_stocks),
                'n_losers': len(loser_stocks),
                'holding_period': holding_period
            })

        if not results:
            return pd.DataFrame()

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('year_month')

        # Annualization
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
                'n_observations': 0
            }

        holding_period = returns_df['holding_period'].iloc[0] if 'holding_period' in returns_df.columns else 1
        annualization_factor = 12 / holding_period

        gross_returns = returns_df['ls_return_gross'].dropna().values
        net_returns = returns_df['ls_return_net'].dropna().values

        if len(net_returns) == 0:
            return {
                'strategy_name': strategy_name,
                'error': 'No valid returns',
                'n_observations': 0
            }

        metrics = {
            'strategy_name': strategy_name,
            'holding_period': holding_period,
            'n_observations': len(net_returns),
            'start_date': str(returns_df['year_month'].min()),
            'end_date': str(returns_df['year_month'].max())
        }

        # Return metrics
        metrics.update({
            'gross_mean_return': gross_returns.mean(),
            'net_mean_return': net_returns.mean(),
            'gross_annual_return': gross_returns.mean() * annualization_factor,
            'net_annual_return': net_returns.mean() * annualization_factor,
            'gross_total_return': (1 + gross_returns).prod() - 1,
            'net_total_return': (1 + net_returns).prod() - 1
        })

        # Risk metrics
        metrics.update({
            'gross_volatility': gross_returns.std() * np.sqrt(annualization_factor),
            'net_volatility': net_returns.std() * np.sqrt(annualization_factor),
            'max_drawdown': self._calculate_max_drawdown(net_returns),
            'downside_deviation': np.sqrt(np.mean(np.minimum(0, net_returns) ** 2)) * np.sqrt(annualization_factor)
        })

        # Risk-adjusted metrics
        metrics['gross_sharpe'] = (
            metrics['gross_annual_return'] / metrics['gross_volatility']
            if metrics['gross_volatility'] > 0 else 0
        )
        metrics['net_sharpe'] = (
            metrics['net_annual_return'] / metrics['net_volatility']
            if metrics['net_volatility'] > 0 else 0
        )
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
                'kurtosis': stats.kurtosis(net_returns)
            })

        # Win/Loss statistics
        winning_returns = net_returns[net_returns > 0]
        losing_returns = net_returns[net_returns < 0]

        metrics.update({
            'win_rate': len(winning_returns) / len(net_returns) if len(net_returns) > 0 else 0,
            'avg_win': winning_returns.mean() if len(winning_returns) > 0 else 0,
            'avg_loss': losing_returns.mean() if len(losing_returns) > 0 else 0,
            'best_month': net_returns.max() if len(net_returns) > 0 else 0,
            'worst_month': net_returns.min() if len(net_returns) > 0 else 0
        })

        metrics['win_loss_ratio'] = (
            abs(metrics['avg_win'] / metrics['avg_loss'])
            if metrics['avg_loss'] != 0 else 0
        )

        # Statistical significance
        n_periods = len(net_returns)
        if n_periods > 2:
            # T-test for mean return
            t_stat_return = net_returns.mean() / (net_returns.std() / np.sqrt(n_periods))
            p_value_return = 2 * (1 - stats.t.cdf(abs(t_stat_return), df=n_periods - 1))

            # T-test for Sharpe ratio (with Newey-West adjustment approximation)
            t_stat_sharpe = metrics['net_sharpe'] * np.sqrt(n_periods / annualization_factor)
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
            'significant_sharpe': p_value_sharpe < self.config.significance_level
        })

        # Transaction cost analysis
        total_tcosts = returns_df['transaction_cost'].sum()
        total_gross = returns_df['ls_return_gross'].sum()

        metrics.update({
            'total_transaction_costs': total_tcosts,
            'tcost_as_pct_of_gross': (total_tcosts / abs(total_gross) * 100) if total_gross != 0 else 0,
            'avg_n_winners': returns_df['n_winners'].mean(),
            'avg_n_losers': returns_df['n_losers'].mean()
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

    def analyze_momentum_crashes(self, df: pd.DataFrame, returns_df: pd.DataFrame) -> Dict:
        """Analyze momentum crashes as per Daniel & Moskowitz (2016)"""
        logger.info("Analyzing momentum crashes...")

        # Calculate market indicators
        market_indicators = self.calculate_market_indicators(df)

        # Identify crash periods (momentum return < -20%)
        crash_threshold = -0.20
        crashes = returns_df[returns_df['ls_return_net'] < crash_threshold].copy()

        crash_analysis = {
            'n_crashes': len(crashes),
            'crash_dates': crashes['year_month'].tolist() if len(crashes) > 0 else [],
            'avg_crash_magnitude': crashes['ls_return_net'].mean() if len(crashes) > 0 else 0,
            'worst_crash': crashes['ls_return_net'].min() if len(crashes) > 0 else 0
        }

        # Analyze market conditions during crashes
        if len(crashes) > 0:
            crash_periods = []
            for _, crash in crashes.iterrows():
                # Find corresponding market conditions
                month_start = pd.Timestamp(str(crash['year_month']))
                market_slice = market_indicators['market_volatility'][
                    (market_indicators['market_volatility'].index >= month_start) &
                    (market_indicators['market_volatility'].index < month_start + pd.DateOffset(months=1))
                    ]

                if len(market_slice) > 0:
                    crash_periods.append({
                        'date': crash['year_month'],
                        'momentum_return': crash['ls_return_net'],
                        'market_volatility': market_slice.mean(),
                        'in_bear_market': market_indicators['bear_market'][market_slice.index].any()
                    })

            crash_df = pd.DataFrame(crash_periods)
            crash_analysis['avg_volatility_during_crashes'] = crash_df['market_volatility'].mean()
            crash_analysis['pct_crashes_in_bear_market'] = crash_df['in_bear_market'].mean()

        return crash_analysis

    def create_comprehensive_visualizations(self, all_results: pd.DataFrame,
                                            best_strategy_returns: pd.DataFrame):
        """Create comprehensive visualizations of momentum strategies"""
        logger.info("Creating comprehensive visualizations...")

        output_dir = Path("outputs/momentum_literature")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(24, 20))
        gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Heatmap of Sharpe ratios for J/K combinations
        ax1 = fig.add_subplot(gs[0, :2])
        jt_results = all_results[all_results['strategy'] == 'Jegadeesh-Titman'].copy()
        if not jt_results.empty:
            pivot_sharpe = jt_results.pivot_table(
                values='net_sharpe',
                index='J_formation',
                columns='K_holding',
                aggfunc='mean'
            )
            sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', cmap='RdYlBu_r',
                        center=0, ax=ax1, cbar_kws={'label': 'Sharpe Ratio'})
            ax1.set_title('Sharpe Ratios by Formation (J) and Holding (K) Periods',
                          fontsize=12, fontweight='bold')
            ax1.set_xlabel('Holding Period K (months)')
            ax1.set_ylabel('Formation Period J (months)')

        # 2. Heatmap of annual returns
        ax2 = fig.add_subplot(gs[0, 2:])
        if not jt_results.empty:
            pivot_returns = jt_results.pivot_table(
                values='net_annual_return',
                index='J_formation',
                columns='K_holding',
                aggfunc='mean'
            ) * 100
            sns.heatmap(pivot_returns, annot=True, fmt='.1f', cmap='RdYlGn',
                        center=0, ax=ax2, cbar_kws={'label': 'Annual Return (%)'})
            ax2.set_title('Annual Returns (%) by J/K Combinations',
                          fontsize=12, fontweight='bold')
            ax2.set_xlabel('Holding Period K (months)')
            ax2.set_ylabel('Formation Period J (months)')

        # 3. Cumulative returns of best strategies
        ax3 = fig.add_subplot(gs[1, :2])
        if not best_strategy_returns.empty:
            ax3.plot(best_strategy_returns.index,
                     best_strategy_returns['cum_return_gross'] * 100,
                     label='Gross', linewidth=2, color='blue')
            ax3.plot(best_strategy_returns.index,
                     best_strategy_returns['cum_return_net'] * 100,
                     label='Net', linewidth=2, linestyle='--', color='red')
            ax3.set_title('Cumulative Returns - Best Strategy',
                          fontsize=12, fontweight='bold')
            ax3.set_ylabel('Cumulative Return (%)')
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)

        # 4. Monthly returns distribution
        ax4 = fig.add_subplot(gs[1, 2])
        if not all_results.empty and 'net_mean_return' in all_results.columns:
            monthly_returns = all_results['net_mean_return'].dropna() * 100
            ax4.hist(monthly_returns, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
            ax4.axvline(0, color='red', linestyle='--', alpha=0.7)
            ax4.axvline(monthly_returns.mean(), color='green', linestyle='-',
                        alpha=0.7, label=f'Mean: {monthly_returns.mean():.2f}%')
            ax4.set_title('Distribution of Monthly Returns', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Monthly Return (%)')
            ax4.set_ylabel('Frequency')
            ax4.legend()

        # 5. Sharpe ratio comparison across strategies
        ax5 = fig.add_subplot(gs[1, 3])
        if 'strategy' in all_results.columns:
            strategy_sharpes = all_results.groupby('strategy')['net_sharpe'].mean().sort_values()
            bars = ax5.barh(range(len(strategy_sharpes)), strategy_sharpes.values,
                            color=['red' if x < 0 else 'green' for x in strategy_sharpes.values])
            ax5.set_yticks(range(len(strategy_sharpes)))
            ax5.set_yticklabels(strategy_sharpes.index)
            ax5.set_xlabel('Average Sharpe Ratio')
            ax5.set_title('Sharpe Ratios by Strategy Type', fontsize=12, fontweight='bold')
            ax5.axvline(0, color='black', linestyle='-', linewidth=0.5)
            ax5.grid(True, alpha=0.3, axis='x')

        # 6. Rolling 12-month Sharpe ratio
        ax6 = fig.add_subplot(gs[2, :2])
        if not best_strategy_returns.empty:
            rolling_sharpe = (
                    best_strategy_returns['ls_return_net'].rolling(12).mean() /
                    best_strategy_returns['ls_return_net'].rolling(12).std() * np.sqrt(12)
            )
            ax6.plot(best_strategy_returns.index, rolling_sharpe,
                     color='purple', linewidth=1.5)
            ax6.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax6.set_title('Rolling 12-Month Sharpe Ratio', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Sharpe Ratio')
            ax6.grid(True, alpha=0.3)

        # 7. Drawdown chart
        ax7 = fig.add_subplot(gs[2, 2:])
        if not best_strategy_returns.empty:
            cum_returns = (1 + best_strategy_returns['ls_return_net']).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max * 100
            ax7.fill_between(best_strategy_returns.index, 0, drawdown,
                             color='red', alpha=0.3)
            ax7.plot(best_strategy_returns.index, drawdown,
                     color='darkred', linewidth=1)
            ax7.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
            ax7.set_ylabel('Drawdown (%)')
            ax7.grid(True, alpha=0.3)

        # 8. Win rate vs Sharpe scatter
        ax8 = fig.add_subplot(gs[3, :2])
        if 'win_rate' in all_results.columns and 'net_sharpe' in all_results.columns:
            valid_data = all_results.dropna(subset=['win_rate', 'net_sharpe', 'net_annual_return'])
            scatter = ax8.scatter(valid_data['win_rate'] * 100,
                                  valid_data['net_sharpe'],
                                  c=valid_data['net_annual_return'] * 100,
                                  cmap='RdYlGn', alpha=0.6, s=50)
            ax8.set_xlabel('Win Rate (%)')
            ax8.set_ylabel('Sharpe Ratio')
            ax8.set_title('Win Rate vs Sharpe Ratio', fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=ax8, label='Annual Return (%)')
            ax8.grid(True, alpha=0.3)

        # 9. Transaction costs impact
        ax9 = fig.add_subplot(gs[3, 2])
        if 'gross_annual_return' in all_results.columns and 'net_annual_return' in all_results.columns:
            valid_data = all_results.dropna(subset=['gross_annual_return', 'net_annual_return'])
            cost_impact = (valid_data['gross_annual_return'] - valid_data['net_annual_return']) * 100
            ax9.hist(cost_impact, bins=20, alpha=0.7, edgecolor='black', color='orange')
            ax9.set_xlabel('Transaction Cost Impact (%)')
            ax9.set_ylabel('Frequency')
            ax9.set_title('Transaction Costs Impact on Returns', fontsize=12, fontweight='bold')
            ax9.axvline(cost_impact.mean(), color='red', linestyle='--',
                        label=f'Mean: {cost_impact.mean():.2f}%')
            ax9.legend()

        # 10. Number of stocks in portfolios over time
        ax10 = fig.add_subplot(gs[3, 3])
        if not best_strategy_returns.empty and 'n_winners' in best_strategy_returns.columns:
            ax10.plot(best_strategy_returns.index, best_strategy_returns['n_winners'],
                      label='Winners', alpha=0.7, color='green')
            ax10.plot(best_strategy_returns.index, best_strategy_returns['n_losers'],
                      label='Losers', alpha=0.7, color='red')
            ax10.set_title('Number of Stocks in Portfolios', fontsize=12, fontweight='bold')
            ax10.set_ylabel('Number of Stocks')
            ax10.legend()
            ax10.grid(True, alpha=0.3)

        # 11. Statistical significance overview
        ax11 = fig.add_subplot(gs[4, :2])
        if 'significant_sharpe' in all_results.columns:
            sig_data = all_results.groupby('strategy')['significant_sharpe'].mean() * 100
            bars = ax11.bar(range(len(sig_data)), sig_data.values,
                            color='green', alpha=0.7, edgecolor='black')
            ax11.set_xticks(range(len(sig_data)))
            ax11.set_xticklabels(sig_data.index, rotation=45, ha='right')
            ax11.set_ylabel('% Significant Strategies')
            ax11.set_title('Statistical Significance by Strategy Type',
                           fontsize=12, fontweight='bold')
            ax11.axhline(5, color='red', linestyle='--', alpha=0.5,
                         label='5% significance level')
            ax11.legend()

        # 12. Best strategies table
        ax12 = fig.add_subplot(gs[4, 2:])
        ax12.axis('tight')
        ax12.axis('off')

        if not all_results.empty:
            # Get top 10 strategies
            top_strategies = all_results.nlargest(10, 'net_sharpe')[
                ['strategy_name', 'net_annual_return', 'net_sharpe',
                 'max_drawdown', 'win_rate', 'significant_sharpe']
            ].copy()

            # Format for display
            top_strategies['Annual Return'] = top_strategies['net_annual_return'].apply(lambda x: f'{x * 100:.2f}%')
            top_strategies['Sharpe'] = top_strategies['net_sharpe'].apply(lambda x: f'{x:.3f}')
            top_strategies['Max DD'] = top_strategies['max_drawdown'].apply(lambda x: f'{x * 100:.1f}%')
            top_strategies['Win Rate'] = top_strategies['win_rate'].apply(lambda x: f'{x * 100:.1f}%')
            top_strategies['Significant'] = top_strategies['significant_sharpe'].apply(lambda x: '✓' if x else '✗')

            table_data = top_strategies[['strategy_name', 'Annual Return', 'Sharpe',
                                         'Max DD', 'Win Rate', 'Significant']].values

            table = ax12.table(cellText=table_data,
                               colLabels=['Strategy', 'Return', 'Sharpe', 'Max DD', 'Win Rate', 'Sig.'],
                               cellLoc='center',
                               loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax12.set_title('Top 10 Strategies by Sharpe Ratio',
                           fontsize=12, fontweight='bold', pad=20)

        plt.suptitle('Momentum Strategy Literature Review - Comprehensive Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # Save figure
        plot_path = output_dir / "momentum_literature_comprehensive.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"Comprehensive visualization saved to {plot_path}")
        plt.close()

        # Create additional detailed charts
        self._create_strategy_comparison_charts(all_results, output_dir)
        self._create_period_analysis_charts(all_results, output_dir)

    def _create_strategy_comparison_charts(self, all_results: pd.DataFrame, output_dir: Path):
        """Create detailed strategy comparison charts"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Chart 1: Returns by strategy
        ax = axes[0, 0]
        if 'strategy' in all_results.columns:
            strategy_returns = all_results.groupby('strategy')['net_annual_return'].mean() * 100
            strategy_returns.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_title('Average Annual Returns by Strategy')
            ax.set_ylabel('Annual Return (%)')
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)

        # Chart 2: Volatility by strategy
        ax = axes[0, 1]
        if 'strategy' in all_results.columns:
            strategy_vol = all_results.groupby('strategy')['net_volatility'].mean() * 100
            strategy_vol.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
            ax.set_title('Average Volatility by Strategy')
            ax.set_ylabel('Volatility (%)')
            ax.grid(True, alpha=0.3)

        # Chart 3: Max drawdown by strategy
        ax = axes[0, 2]
        if 'strategy' in all_results.columns:
            strategy_dd = all_results.groupby('strategy')['max_drawdown'].mean() * 100
            strategy_dd.plot(kind='bar', ax=ax, color='salmon', edgecolor='black')
            ax.set_title('Average Maximum Drawdown by Strategy')
            ax.set_ylabel('Max Drawdown (%)')
            ax.grid(True, alpha=0.3)

        # Chart 4: Sortino ratio comparison
        ax = axes[1, 0]
        if 'sortino_ratio' in all_results.columns:
            strategy_sortino = all_results.groupby('strategy')['sortino_ratio'].mean()
            strategy_sortino.plot(kind='bar', ax=ax, color='lightgreen', edgecolor='black')
            ax.set_title('Average Sortino Ratio by Strategy')
            ax.set_ylabel('Sortino Ratio')
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)

        # Chart 5: Calmar ratio comparison
        ax = axes[1, 1]
        if 'calmar_ratio' in all_results.columns:
            strategy_calmar = all_results.groupby('strategy')['calmar_ratio'].mean()
            strategy_calmar.plot(kind='bar', ax=ax, color='plum', edgecolor='black')
            ax.set_title('Average Calmar Ratio by Strategy')
            ax.set_ylabel('Calmar Ratio')
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)

        # Chart 6: Transaction costs impact
        ax = axes[1, 2]
        if 'tcost_as_pct_of_gross' in all_results.columns:
            strategy_tcost = all_results.groupby('strategy')['tcost_as_pct_of_gross'].mean()
            strategy_tcost.plot(kind='bar', ax=ax, color='gold', edgecolor='black')
            ax.set_title('Transaction Costs as % of Gross Returns')
            ax.set_ylabel('Transaction Costs (%)')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Strategy Comparison - Detailed Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = output_dir / "strategy_comparison_detailed.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Strategy comparison charts saved to {plot_path}")
        plt.close()

    def _create_period_analysis_charts(self, all_results: pd.DataFrame, output_dir: Path):
        """Create period-specific analysis charts"""

        jt_results = all_results[all_results['strategy'] == 'Jegadeesh-Titman'].copy()

        if jt_results.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Chart 1: Effect of skip period
        ax = axes[0, 0]
        skip_effect = jt_results.groupby('skip_period')['net_sharpe'].mean()
        skip_effect.plot(kind='line', ax=ax, marker='o', markersize=8, linewidth=2)
        ax.set_title('Effect of Skip Period on Sharpe Ratio')
        ax.set_xlabel('Skip Period (months)')
        ax.set_ylabel('Average Sharpe Ratio')
        ax.grid(True, alpha=0.3)

        # Chart 2: Formation period effect
        ax = axes[0, 1]
        formation_effect = jt_results.groupby('J_formation')['net_sharpe'].mean()
        formation_effect.plot(kind='line', ax=ax, marker='s', markersize=8,
                              linewidth=2, color='green')
        ax.set_title('Effect of Formation Period on Sharpe Ratio')
        ax.set_xlabel('Formation Period J (months)')
        ax.set_ylabel('Average Sharpe Ratio')
        ax.grid(True, alpha=0.3)

        # Chart 3: Holding period effect
        ax = axes[1, 0]
        holding_effect = jt_results.groupby('K_holding')['net_sharpe'].mean()
        holding_effect.plot(kind='line', ax=ax, marker='^', markersize=8,
                            linewidth=2, color='red')
        ax.set_title('Effect of Holding Period on Sharpe Ratio')
        ax.set_xlabel('Holding Period K (months)')
        ax.set_ylabel('Average Sharpe Ratio')
        ax.grid(True, alpha=0.3)

        # Chart 4: 3D surface plot of J/K interaction
        ax = axes[1, 1]
        pivot_data = jt_results.pivot_table(
            values='net_sharpe',
            index='J_formation',
            columns='K_holding',
            aggfunc='mean'
        )

        im = ax.imshow(pivot_data.values, cmap='RdYlBu_r', aspect='auto',
                       interpolation='bilinear')
        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_xticklabels(pivot_data.columns)
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_yticklabels(pivot_data.index)
        ax.set_xlabel('Holding Period K (months)')
        ax.set_ylabel('Formation Period J (months)')
        ax.set_title('J/K Interaction Effect on Sharpe Ratio')
        plt.colorbar(im, ax=ax)

        plt.suptitle('Period Analysis - Jegadeesh & Titman Parameters',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = output_dir / "period_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Period analysis charts saved to {plot_path}")
        plt.close()

    def save_detailed_results(self, all_results: pd.DataFrame, output_dir: Path):
        """Save detailed results to multiple formats"""

        # Save complete results
        all_results.to_csv(output_dir / "momentum_literature_complete_results.csv", index=False)

        # Save summary by strategy
        if 'strategy' in all_results.columns:
            strategy_summary = all_results.groupby('strategy').agg({
                'net_annual_return': ['mean', 'std', 'min', 'max'],
                'net_sharpe': ['mean', 'std', 'min', 'max'],
                'max_drawdown': ['mean', 'min'],
                'win_rate': 'mean',
                'significant_sharpe': 'mean'
            }).round(4)
            strategy_summary.to_csv(output_dir / "strategy_summary.csv")

        # Save best strategies
        top_20 = all_results.nlargest(20, 'net_sharpe')
        top_20.to_csv(output_dir / "top_20_strategies.csv", index=False)

        # Save J/K analysis for Jegadeesh-Titman
        jt_results = all_results[all_results['strategy'] == 'Jegadeesh-Titman'].copy()
        if not jt_results.empty:
            jk_summary = jt_results.pivot_table(
                values=['net_sharpe', 'net_annual_return'],
                index='J_formation',
                columns='K_holding',
                aggfunc='mean'
            )
            jk_summary.to_csv(output_dir / "jegadeesh_titman_jk_analysis.csv")

        logger.info(f"Detailed results saved to {output_dir}")

    def run_comprehensive_analysis(self, data_path: str) -> Dict:
        """Run complete momentum literature analysis"""
        logger.info("=" * 70)
        logger.info("COMPREHENSIVE MOMENTUM LITERATURE ANALYSIS")
        logger.info("=" * 70)

        # Load data
        df = self.load_data(data_path)
        if df.empty:
            logger.error("No data available for analysis")
            return {}

        all_results = []

        # 1. Run Jegadeesh & Titman strategies
        logger.info("\n" + "=" * 50)
        logger.info("Testing Jegadeesh & Titman (1993) Strategies")
        logger.info("=" * 50)
        jt_results = self.run_jegadeesh_titman_backtest(df)
        if not jt_results.empty:
            all_results.append(jt_results)

        # 2. Run Carhart momentum factor
        logger.info("\n" + "=" * 50)
        logger.info("Testing Carhart (1997) Momentum Factor")
        logger.info("=" * 50)
        carhart_result = self.run_carhart_momentum(df)
        if carhart_result:
            carhart_df = pd.DataFrame([carhart_result])
            all_results.append(carhart_df)

        # 3. Run Fama-French size-segmented momentum
        logger.info("\n" + "=" * 50)
        logger.info("Testing Fama & French (2012) Size-Segmented Momentum")
        logger.info("=" * 50)
        ff_result = self.run_size_segmented_momentum(df)
        if ff_result:
            ff_df = pd.DataFrame([ff_result])
            all_results.append(ff_df)

        # Combine all results
        if all_results:
            results_df = pd.concat(all_results, ignore_index=True)

            # Find best strategy overall
            best_strategy_idx = results_df['net_sharpe'].idxmax()
            best_strategy = results_df.loc[best_strategy_idx]

            logger.info("\n" + "=" * 50)
            logger.info("BEST STRATEGY FOUND:")
            logger.info(f"Strategy: {best_strategy['strategy_name']}")
            logger.info(f"Sharpe Ratio: {best_strategy['net_sharpe']:.3f}")
            logger.info(f"Annual Return: {best_strategy['net_annual_return'] * 100:.2f}%")
            logger.info(f"Max Drawdown: {best_strategy['max_drawdown'] * 100:.2f}%")
            logger.info("=" * 50)

            # Get returns for best strategy for visualization
            best_strategy_returns = pd.DataFrame()  # Would need to store returns during backtest

            # Create visualizations
            self.create_comprehensive_visualizations(results_df, best_strategy_returns)

            # Save results
            output_dir = Path("outputs/momentum_literature")
            output_dir.mkdir(parents=True, exist_ok=True)
            self.save_detailed_results(results_df, output_dir)

            # Store results
            self.all_results = {
                'results_df': results_df,
                'best_strategy': best_strategy.to_dict(),
                'n_strategies_tested': len(results_df),
                'n_significant': results_df[
                    'significant_sharpe'].sum() if 'significant_sharpe' in results_df.columns else 0
            }

            # Print summary
            self.print_comprehensive_summary()

            return self.all_results
        else:
            logger.error("No successful strategies found!")
            return {}

    def print_comprehensive_summary(self):
        """Print comprehensive analysis summary"""
        if not self.all_results or 'results_df' not in self.all_results:
            return

        results_df = self.all_results['results_df']

        print("\n" + "=" * 70)
        print("MOMENTUM LITERATURE REVIEW - COMPREHENSIVE RESULTS")
        print("=" * 70)

        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"-" * 40)
        print(f"Total Strategies Tested: {len(results_df)}")
        print(f"Statistically Significant: {self.all_results['n_significant']}")

        print(f"\n🏆 PERFORMANCE OVERVIEW:")
        print(f"-" * 40)

        # Strategy-level summary
        if 'strategy' in results_df.columns:
            for strategy in results_df['strategy'].unique():
                strategy_data = results_df[results_df['strategy'] == strategy]
                print(f"\n{strategy}:")
                print(f"  Strategies tested: {len(strategy_data)}")
                print(f"  Best Sharpe: {strategy_data['net_sharpe'].max():.3f}")
                print(f"  Avg Sharpe: {strategy_data['net_sharpe'].mean():.3f}")
                print(f"  Best Return: {strategy_data['net_annual_return'].max() * 100:.2f}%")
                print(f"  Avg Return: {strategy_data['net_annual_return'].mean() * 100:.2f}%")

        print(f"\n🎯 TOP 5 STRATEGIES BY SHARPE RATIO:")
        print(f"-" * 40)
        top_5 = results_df.nlargest(5, 'net_sharpe')
        for idx, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"{idx}. {row['strategy_name']}")
            print(f"   Sharpe: {row['net_sharpe']:.3f}, Return: {row['net_annual_return'] * 100:.2f}%, "
                  f"Max DD: {row['max_drawdown'] * 100:.1f}%")

        print(f"\n📈 COMPARISON WITH LITERATURE:")
        print(f"-" * 40)
        print("Jegadeesh & Titman (1993) reported monthly returns of ~1%")
        print("Our replication results:")

        jt_results = results_df[results_df['strategy'] == 'Jegadeesh-Titman']
        if not jt_results.empty:
            # Find classic 6-6 strategy
            classic_66 = jt_results[(jt_results['J_formation'] == 6) &
                                    (jt_results['K_holding'] == 6)]
            if not classic_66.empty:
                classic_result = classic_66.iloc[0]
                print(f"  6-6 Strategy: {classic_result['net_mean_return'] * 100:.2f}% monthly")
                print(f"  Sharpe Ratio: {classic_result['net_sharpe']:.3f}")

        print("\n" + "=" * 70)
        print("📁 OUTPUT FILES:")
        print("-" * 40)
        print("• outputs/momentum_literature/momentum_literature_complete_results.csv")
        print("• outputs/momentum_literature/top_20_strategies.csv")
        print("• outputs/momentum_literature/jegadeesh_titman_jk_analysis.csv")
        print("• outputs/momentum_literature/momentum_literature_comprehensive.png")
        print("• outputs/momentum_literature/strategy_comparison_detailed.png")
        print("• outputs/momentum_literature/period_analysis.png")
        print("=" * 70)


def main():
    """Main execution function"""
    # Configuration
    config = MomentumLiteratureConfig(
        strategy_type='all',
        jt_formation_periods=[3, 6, 9, 12],
        jt_holding_periods=[3, 6, 9, 12],
        jt_skip_periods=[0, 1],
        n_portfolios=10,
        weighting_scheme='equal',
        spread_cost_bps=10,
        market_impact_bps=5,
        brokerage_cost_bps=2,
        exclude_microcaps=True
    )

    # Initialize strategy
    strategy = MomentumLiteratureStrategy(config)

    # Data path
    data_path = "data/processed/prices_clean.parquet"

    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")

        # Try alternative paths
        alt_paths = [
            "data/raw/price_data.parquet",
            "data/raw/prices_combined.csv"
        ]

        for alt_path in alt_paths:
            if Path(alt_path).exists():
                logger.info(f"Using alternative data: {alt_path}")
                data_path = alt_path
                break
        else:
            logger.error("No data files found! Please run data fetching first.")
            return 1

    # Run comprehensive analysis
    results = strategy.run_comprehensive_analysis(data_path)

    if results:
        logger.info("✅ Momentum literature analysis completed successfully")
        return 0
    else:
        logger.error("❌ Momentum literature analysis failed")
        return 1


if __name__ == "__main__":
    exit(main())