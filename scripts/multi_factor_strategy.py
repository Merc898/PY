#!/usr/bin/env python3
"""
multi_factor_strategy_fixed.py - Fixed version with momentum calculation bug resolved
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultiFactorConfig:
    """Configuration for multi-factor strategy"""
    # Factor weights (will be optimized)
    momentum_weight: float = 0.25
    value_weight: float = 0.25
    quality_weight: float = 0.25
    low_vol_weight: float = 0.25

    # Factor parameters
    momentum_lookback: int = 12
    momentum_skip: int = 2
    value_metric: str = 'earnings_yield'  # or 'book_to_market'
    quality_metric: str = 'roe'  # or 'gross_profitability'
    volatility_window: int = 252

    # Portfolio construction
    n_portfolios: int = 10
    rebalance_frequency: str = 'monthly'

    # Risk management
    max_position_weight: float = 0.05
    target_volatility: float = 0.10
    use_risk_parity: bool = True

    # Transaction costs
    tcost_bps: float = 20  # Total transaction costs


class FactorCalculator:
    """Calculate various factors for stocks"""

    @staticmethod
    def calculate_momentum(df: pd.DataFrame, lookback: int = 12, skip: int = 2) -> pd.Series:
        """Calculate momentum factor - FIXED VERSION"""
        df = df.sort_values(['symbol', 'date'])

        # Calculate cumulative returns
        df['ret'] = df.groupby('symbol')['close'].pct_change()

        # Calculate momentum for each stock
        def calc_momentum(group):
            if len(group) < lookback + skip:
                return pd.Series([np.nan] * len(group), index=group.index)

            # Calculate rolling compound return
            momentum = []

            # FIXED: Changed len(group) + 1 to len(group) to prevent length mismatch
            for i in range(lookback + skip, len(group)):
                window = group.iloc[i - lookback - skip:i - skip]
                if len(window) >= lookback * 0.7:  # Require 70% of data
                    mom = (1 + window['ret']).prod() - 1
                else:
                    mom = np.nan
                momentum.append(mom)

            # Pad the beginning with NaN
            momentum = [np.nan] * (lookback + skip) + momentum

            # FIXED: Ensure exact length match
            if len(momentum) != len(group):
                # Truncate or pad to exact length
                if len(momentum) > len(group):
                    momentum = momentum[:len(group)]
                else:
                    momentum = momentum + [np.nan] * (len(group) - len(momentum))

            return pd.Series(momentum, index=group.index)

        momentum_series = df.groupby('symbol').apply(calc_momentum, include_groups=False)
        return momentum_series.droplevel(0)  # Remove the symbol level from index

    @staticmethod
    def calculate_value(df: pd.DataFrame, fundamental_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """Calculate value factor (simplified - using price patterns as proxy)"""
        # In production, you'd use actual fundamental data
        # Here we use a simple price-based value proxy

        df = df.sort_values(['symbol', 'date'])

        # Price to 52-week high ratio (inverse as value proxy)
        df['high_52w'] = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(252, min_periods=200).max()
        )
        df['value_score'] = 1 - (df['close'] / df['high_52w'])

        # Alternative: Use mean reversion
        df['sma_200'] = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(200, min_periods=150).mean()
        )
        df['mean_reversion'] = (df['sma_200'] - df['close']) / df['sma_200']

        # Combine both
        df['value'] = 0.5 * df['value_score'] + 0.5 * df['mean_reversion']

        return df['value']

    @staticmethod
    def calculate_quality(df: pd.DataFrame) -> pd.Series:
        """Calculate quality factor (using price stability as proxy)"""
        # In production, use ROE, ROA, gross profitability, etc.
        # Here we use price stability and trend strength as quality proxies

        df = df.sort_values(['symbol', 'date'])

        # Calculate returns
        df['ret'] = df.groupby('symbol')['close'].pct_change()

        # Quality metric 1: Stability (inverse of volatility)
        df['stability'] = df.groupby('symbol')['ret'].transform(
            lambda x: -x.rolling(252, min_periods=200).std()
        )

        # Quality metric 2: Trend consistency (% of days with positive returns)
        df['consistency'] = df.groupby('symbol')['ret'].transform(
            lambda x: x.rolling(252, min_periods=200).apply(lambda y: (y > 0).mean())
        )

        # Quality metric 3: Drawdown resilience
        df['cummax'] = df.groupby('symbol')['close'].cummax()
        df['drawdown'] = (df['close'] - df['cummax']) / df['cummax']
        df['max_dd'] = df.groupby('symbol')['drawdown'].transform(
            lambda x: x.rolling(252, min_periods=200).min()
        )
        df['dd_resilience'] = -df['max_dd']  # Less drawdown = higher quality

        # Combine quality metrics
        df['quality'] = (
                0.33 * df['stability'] +
                0.33 * df['consistency'] +
                0.34 * df['dd_resilience']
        )

        return df['quality']

    @staticmethod
    def calculate_low_volatility(df: pd.DataFrame, window: int = 252) -> pd.Series:
        """Calculate low volatility factor"""
        df = df.sort_values(['symbol', 'date'])

        # Calculate returns
        df['ret'] = df.groupby('symbol')['close'].pct_change()

        # Calculate rolling volatility
        df['volatility'] = df.groupby('symbol')['ret'].transform(
            lambda x: x.rolling(window, min_periods=int(window * 0.8)).std() * np.sqrt(252)
        )

        # Invert for scoring (low vol = high score)
        df['low_vol'] = -df['volatility']

        return df['low_vol']


class MultiFactorStrategy:
    """Advanced multi-factor strategy implementation with fixes"""

    def __init__(self, config: MultiFactorConfig = None):
        self.config = config or MultiFactorConfig()
        self.factor_calculator = FactorCalculator()
        self.results = {}

    def load_and_prepare_data(self, data_path: str) -> pd.DataFrame:
        """Load and prepare data for analysis - Updated with currency fixes"""
        logger.info("Loading and preparing data...")

        # Load data
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        df['date'] = pd.to_datetime(df['date'])

        # FIXED CURRENCY HANDLING - Same as momentum strategy:
        if 'close_usd' in df.columns:
            # Use USD-adjusted prices
            logger.info("Using USD-adjusted prices")
            df['close'] = df['close_usd']
        elif 'currency' in df.columns:
            # Check currencies present
            currencies = df['currency'].unique()
            logger.info(f"Found currencies: {currencies}")

            if len(currencies) > 1:
                logger.warning("Multiple currencies detected!")
                logger.info("Filtering to USD-only securities to avoid FX contamination")

                # Show currency breakdown
                for curr in currencies:
                    count = (df['currency'] == curr).sum()
                    symbols = df[df['currency'] == curr]['symbol'].nunique()
                    logger.info(f"   {curr}: {count:,} obs, {symbols} symbols")

                # Filter to USD only
                df = df[df['currency'] == 'USD'].copy()
                logger.info(f"Filtered to USD-only: {len(df):,} observations")
        else:
            # No currency info - assume USD but warn
            logger.warning("No currency information - assuming all prices are USD")

        # Quality filters
        initial_count = len(df)

        # Remove penny stocks
        df = df[df['close'] >= 5.0]

        # Remove negative prices
        df = df[df['close'] > 0]

        # Remove extreme prices (likely errors)
        df = df[df['close'] <= 50000]

        # Ensure enough history - require minimum observations per symbol
        symbol_counts = df.groupby('symbol').size()
        min_obs = 500  # About 2 years of trading days
        valid_symbols = symbol_counts[symbol_counts >= min_obs].index
        df = df[df['symbol'].isin(valid_symbols)].copy()

        final_count = len(df)
        removed_count = initial_count - final_count

        logger.info(f"Data prepared:")
        logger.info(f"  Final observations: {final_count:,}")
        logger.info(f"  Removed: {removed_count:,} problematic observations")
        logger.info(f"  Valid symbols: {df['symbol'].nunique()}")
        logger.info(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

        return df

    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all factor scores"""
        logger.info("Calculating factor scores...")

        # Calculate each factor
        logger.info("  Calculating momentum...")
        df['momentum'] = self.factor_calculator.calculate_momentum(
            df, self.config.momentum_lookback, self.config.momentum_skip
        )

        logger.info("  Calculating value...")
        df['value'] = self.factor_calculator.calculate_value(df)

        logger.info("  Calculating quality...")
        df['quality'] = self.factor_calculator.calculate_quality(df)

        logger.info("  Calculating low volatility...")
        df['low_vol'] = self.factor_calculator.calculate_low_volatility(
            df, self.config.volatility_window
        )

        # Log factor availability
        for factor in ['momentum', 'value', 'quality', 'low_vol']:
            valid_pct = (~df[factor].isna()).mean() * 100
            logger.info(f"  {factor}: {valid_pct:.1f}% valid observations")

        return df

    def standardize_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize factors cross-sectionally"""
        logger.info("Standardizing factors...")

        factors = ['momentum', 'value', 'quality', 'low_vol']

        # Group by date for cross-sectional standardization
        def standardize_cs(group):
            for factor in factors:
                if factor in group.columns:
                    # Remove outliers (winsorize at 1% and 99%)
                    factor_data = group[factor].copy()
                    if factor_data.notna().sum() >= 10:
                        q01 = factor_data.quantile(0.01)
                        q99 = factor_data.quantile(0.99)
                        factor_data = factor_data.clip(q01, q99)

                        # Z-score standardization
                        mean = factor_data.mean()
                        std = factor_data.std()
                        if std > 0:
                            group[f'{factor}_z'] = (factor_data - mean) / std
                        else:
                            group[f'{factor}_z'] = 0
                    else:
                        group[f'{factor}_z'] = np.nan
            return group

        df = df.groupby('date').apply(standardize_cs, include_groups=False)

        return df

    def create_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite multi-factor score"""
        logger.info("Creating composite scores...")

        # Use z-scored factors
        factor_cols = ['momentum_z', 'value_z', 'quality_z', 'low_vol_z']
        weights = [
            self.config.momentum_weight,
            self.config.value_weight,
            self.config.quality_weight,
            self.config.low_vol_weight
        ]

        # Normalize weights
        weights = np.array(weights) / sum(weights)

        # Calculate weighted composite
        df['composite_score'] = 0
        for factor, weight in zip(factor_cols, weights):
            if factor in df.columns:
                df['composite_score'] += df[factor].fillna(0) * weight

        # Also calculate factor exposures
        for factor in factor_cols:
            if factor in df.columns:
                df[f'{factor}_rank'] = df.groupby('date')[factor].rank(pct=True)

        return df

    def form_portfolios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Form portfolios based on composite scores"""
        logger.info("Forming portfolios...")

        # Get monthly data
        df['year_month'] = df['date'].dt.to_period('M')

        # Use end-of-month data for portfolio formation
        monthly_data = df.groupby(['symbol', 'year_month']).last().reset_index()

        # Calculate next month returns
        monthly_data = monthly_data.sort_values(['symbol', 'year_month'])
        monthly_data['ret_next'] = monthly_data.groupby('symbol')['close'].pct_change().shift(-1)

        # Form portfolios for each month
        portfolio_results = []

        for month in monthly_data['year_month'].unique():
            month_data = monthly_data[monthly_data['year_month'] == month].copy()

            # Remove stocks without composite scores
            month_data = month_data.dropna(subset=['composite_score'])

            if len(month_data) < 20:
                continue

            # Rank by composite score
            month_data['rank'] = month_data['composite_score'].rank(pct=True)

            # Assign to portfolios
            try:
                month_data['portfolio'] = pd.qcut(
                    month_data['rank'],
                    self.config.n_portfolios,
                    labels=range(1, self.config.n_portfolios + 1)
                )
            except ValueError:
                # If qcut fails due to duplicate values, use regular binning
                month_data['portfolio'] = pd.cut(
                    month_data['rank'],
                    self.config.n_portfolios,
                    labels=range(1, self.config.n_portfolios + 1)
                )

            # Long top, short bottom
            month_data['position'] = 'neutral'
            month_data.loc[month_data['portfolio'] == self.config.n_portfolios, 'position'] = 'long'
            month_data.loc[month_data['portfolio'] == 1, 'position'] = 'short'

            portfolio_results.append(month_data)

        if portfolio_results:
            return pd.concat(portfolio_results, ignore_index=True)
        else:
            return pd.DataFrame()

    def apply_risk_parity(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """Apply risk parity weighting"""
        logger.info("Applying risk parity...")

        # Simple equal weighting for now
        portfolio_df['weight'] = 0.0

        for month in portfolio_df['year_month'].unique():
            month_mask = portfolio_df['year_month'] == month

            # Long portfolio - equal weight
            long_mask = month_mask & (portfolio_df['position'] == 'long')
            if long_mask.any():
                n_long = long_mask.sum()
                portfolio_df.loc[long_mask, 'weight'] = 1.0 / n_long

            # Short portfolio - equal weight
            short_mask = month_mask & (portfolio_df['position'] == 'short')
            if short_mask.any():
                n_short = short_mask.sum()
                portfolio_df.loc[short_mask, 'weight'] = -1.0 / n_short

        # Apply position limits
        portfolio_df['weight'] = portfolio_df['weight'].clip(
            -self.config.max_position_weight,
            self.config.max_position_weight
        )

        return portfolio_df

    def calculate_returns(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate portfolio returns with transaction costs"""
        logger.info("Calculating portfolio returns...")

        # Group by month and calculate returns
        monthly_returns = []

        for month in portfolio_df['year_month'].unique():
            month_data = portfolio_df[portfolio_df['year_month'] == month]

            # Calculate weighted return
            gross_return = (month_data['weight'] * month_data['ret_next']).sum()

            # Count positions
            n_long = (month_data['position'] == 'long').sum()
            n_short = (month_data['position'] == 'short').sum()
            total_positions = (month_data['weight'] != 0).sum()
            turnover = abs(month_data['weight']).sum()

            monthly_returns.append({
                'year_month': month,
                'gross_return': gross_return,
                'n_long': n_long,
                'n_short': n_short,
                'total_positions': total_positions,
                'turnover': turnover
            })

        monthly_returns_df = pd.DataFrame(monthly_returns)

        # Apply transaction costs
        monthly_returns_df['tcost'] = monthly_returns_df['turnover'] * self.config.tcost_bps / 10000
        monthly_returns_df['net_return'] = monthly_returns_df['gross_return'] - monthly_returns_df['tcost']

        # Calculate cumulative returns
        monthly_returns_df['cum_gross'] = (1 + monthly_returns_df['gross_return']).cumprod() - 1
        monthly_returns_df['cum_net'] = (1 + monthly_returns_df['net_return']).cumprod() - 1

        return monthly_returns_df

    def calculate_factor_attribution(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate factor attribution"""
        logger.info("Calculating factor attribution...")

        factors = ['momentum_z', 'value_z', 'quality_z', 'low_vol_z']

        attribution_results = []

        for month in portfolio_df['year_month'].unique():
            month_data = portfolio_df[portfolio_df['year_month'] == month]

            # Calculate factor exposures
            long_data = month_data[month_data['position'] == 'long']
            short_data = month_data[month_data['position'] == 'short']

            if len(long_data) > 0 and len(short_data) > 0:
                exposures = {}
                for factor in factors:
                    if factor in month_data.columns:
                        long_exp = long_data[factor].mean()
                        short_exp = short_data[factor].mean()
                        exposures[f'{factor}_exposure'] = long_exp - short_exp

                exposures['year_month'] = month
                attribution_results.append(exposures)

        if attribution_results:
            return pd.DataFrame(attribution_results)
        else:
            return pd.DataFrame()

    def calculate_metrics(self, returns_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""

        gross_returns = returns_df['gross_return'].values
        net_returns = returns_df['net_return'].values

        metrics = {
            # Returns
            'gross_annual_return': (1 + gross_returns.mean()) ** 12 - 1,
            'net_annual_return': (1 + net_returns.mean()) ** 12 - 1,

            # Risk
            'gross_volatility': gross_returns.std() * np.sqrt(12),
            'net_volatility': net_returns.std() * np.sqrt(12),

            # Risk-adjusted
            'gross_sharpe': ((1 + gross_returns.mean()) ** 12 - 1) / (
                        gross_returns.std() * np.sqrt(12)) if gross_returns.std() > 0 else 0,
            'net_sharpe': ((1 + net_returns.mean()) ** 12 - 1) / (
                        net_returns.std() * np.sqrt(12)) if net_returns.std() > 0 else 0,

            # Drawdown
            'max_drawdown': self._calculate_max_drawdown(net_returns),

            # Other
            'win_rate': (net_returns > 0).mean(),
            'avg_positions': returns_df['total_positions'].mean(),
            'avg_turnover': returns_df['turnover'].mean(),
            'total_tcost': returns_df['tcost'].sum(),

            # Statistical significance
            't_stat': np.sqrt(
                len(net_returns)) * net_returns.mean() / net_returns.std() if net_returns.std() > 0 else 0,
            'p_value': 2 * (1 - stats.norm.cdf(abs(np.sqrt(
                len(net_returns)) * net_returns.mean() / net_returns.std()))) if net_returns.std() > 0 else 1
        }

        return metrics

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def create_visualizations(self, returns_df: pd.DataFrame, attribution_df: pd.DataFrame):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")

        fig = plt.figure(figsize=(15, 10))

        # 1. Cumulative returns
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(returns_df.index, returns_df['cum_gross'] * 100, label='Gross', linewidth=2)
        ax1.plot(returns_df.index, returns_df['cum_net'] * 100, label='Net', linewidth=2, linestyle='--')
        ax1.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Monthly returns distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(returns_df['net_return'] * 100, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Monthly Returns Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')

        # 3. Rolling Sharpe (12-month)
        ax3 = plt.subplot(2, 3, 3)
        rolling_sharpe = (
                returns_df['net_return'].rolling(12).mean() /
                returns_df['net_return'].rolling(12).std() * np.sqrt(12)
        )
        ax3.plot(returns_df.index, rolling_sharpe, color='green', linewidth=1.5)
        ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax3.set_title('Rolling 12-Month Sharpe Ratio', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True, alpha=0.3)

        # 4. Drawdown chart
        ax4 = plt.subplot(2, 3, 4)
        cum_returns = (1 + returns_df['net_return']).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max * 100
        ax4.fill_between(returns_df.index, 0, drawdown, color='red', alpha=0.3)
        ax4.plot(returns_df.index, drawdown, color='red', linewidth=1)
        ax4.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)

        # 5. Number of positions
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(returns_df.index, returns_df['n_long'], label='Long', alpha=0.7, color='green')
        ax5.plot(returns_df.index, returns_df['n_short'], label='Short', alpha=0.7, color='red')
        ax5.set_title('Number of Positions', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Count')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Return vs risk
        ax6 = plt.subplot(2, 3, 6)
        annual_ret = returns_df['net_return'].mean() * 12 * 100
        annual_vol = returns_df['net_return'].std() * np.sqrt(12) * 100
        ax6.scatter(annual_vol, annual_ret, s=100, color='blue', alpha=0.7)
        ax6.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax6.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Volatility (%)')
        ax6.set_ylabel('Return (%)')
        ax6.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        plt.suptitle('Multi-Factor Strategy Performance', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save
        output_dir = Path("outputs/multi_factor")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "multi_factor_analysis.png", dpi=150, bbox_inches='tight')
        logger.info(f"Plots saved to {output_dir / 'multi_factor_analysis.png'}")
        plt.close()

    def run_backtest(self, data_path: str) -> Dict:
        """Run complete multi-factor backtest"""
        logger.info("=" * 60)
        logger.info("MULTI-FACTOR STRATEGY BACKTEST")
        logger.info("=" * 60)

        try:
            # Load and prepare data
            df = self.load_and_prepare_data(data_path)
            if df.empty:
                logger.error("No data available for backtest")
                return {}

            # Calculate all factors
            df = self.calculate_all_factors(df)

            # Standardize factors
            df = self.standardize_factors(df)

            # Create composite scores
            df = self.create_composite_score(df)

            # Form portfolios
            portfolio_df = self.form_portfolios(df)

            if portfolio_df.empty:
                logger.error("Failed to form portfolios!")
                return {}

            # Apply risk parity if configured
            if self.config.use_risk_parity:
                portfolio_df = self.apply_risk_parity(portfolio_df)

            # Calculate returns
            returns_df = self.calculate_returns(portfolio_df)

            # Calculate factor attribution
            attribution_df = self.calculate_factor_attribution(portfolio_df)

            # Calculate metrics
            metrics = self.calculate_metrics(returns_df)

            # Create visualizations
            self.create_visualizations(returns_df, attribution_df)

            # Store results
            self.results = {
                'returns': returns_df,
                'metrics': metrics,
                'attribution': attribution_df,
                'portfolio': portfolio_df
            }

            # Print results
            self.print_results()

            # Save results
            self.save_results()

            return self.results

        except Exception as e:
            logger.error(f"Multi-factor backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def print_results(self):
        """Print formatted results"""
        if not self.results:
            return

        metrics = self.results['metrics']

        print("\n" + "=" * 60)
        print("MULTI-FACTOR STRATEGY RESULTS")
        print("=" * 60)

        print("\nPERFORMANCE METRICS:")
        print("-" * 40)

        print("\nReturns:")
        print(f"  Gross Annual Return: {metrics['gross_annual_return'] * 100:>7.2f}%")
        print(f"  Net Annual Return:   {metrics['net_annual_return'] * 100:>7.2f}%")

        print("\nRisk:")
        print(f"  Annual Volatility:   {metrics['net_volatility'] * 100:>7.2f}%")
        print(f"  Maximum Drawdown:    {metrics['max_drawdown'] * 100:>7.2f}%")

        print("\nRisk-Adjusted:")
        print(f"  Net Sharpe Ratio:    {metrics['net_sharpe']:>7.3f}")
        print(f"  Win Rate:            {metrics['win_rate'] * 100:>7.2f}%")

        print("\nStatistical Significance:")
        print(f"  T-Statistic:         {metrics['t_stat']:>7.3f}")
        print(f"  P-Value:             {metrics['p_value']:>7.4f}")
        significant = "YES" if metrics['p_value'] < 0.05 else "NO"
        print(f"  Significant (5%):    {significant}")

        print("\nPortfolio Statistics:")
        print(f"  Avg Positions:       {metrics['avg_positions']:>7.1f}")
        print(f"  Avg Turnover:        {metrics['avg_turnover']:>7.2f}")
        print(f"  Total Trans Costs:   {metrics['total_tcost'] * 100:>7.2f}%")

        print("\n" + "=" * 60)

    def save_results(self):
        """Save results to files"""
        output_dir = Path("outputs/multi_factor")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save returns
        self.results['returns'].to_csv(output_dir / "returns.csv", index=False)

        # Save metrics
        metrics_df = pd.DataFrame([self.results['metrics']])
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)

        # Save attribution if available
        if not self.results['attribution'].empty:
            self.results['attribution'].to_csv(output_dir / "attribution.csv", index=False)

        logger.info(f"Results saved to {output_dir}")


def main():
    """Main execution"""
    # Configuration
    config = MultiFactorConfig(
        momentum_weight=0.3,
        value_weight=0.25,
        quality_weight=0.25,
        low_vol_weight=0.2,
        use_risk_parity=True,
        tcost_bps=20
    )

    # Initialize strategy
    strategy = MultiFactorStrategy(config)

    # Use cleaned data if available, otherwise fallback to raw
    data_path = "data/processed/prices_clean.parquet"

    if not Path(data_path).exists():
        logger.error(f"Cleaned data file not found: {data_path}")
        logger.info("Please run the full pipeline (run_all.py) first to clean data")

        # Fallback to raw data with warning
        raw_data_paths = [
            "data/raw/price_data.parquet",
            "data/raw/prices_combined.csv"
        ]

        raw_path_found = False
        for raw_data_path in raw_data_paths:
            if Path(raw_data_path).exists():
                logger.warning("Using raw data - results may be unreliable due to data quality issues")
                logger.warning("Consider running 'python run_all.py --refresh' to clean data first")
                data_path = raw_data_path
                raw_path_found = True
                break

        if not raw_path_found:
            logger.error("No data files found! Run fetch_all_prices.py first.")
            return 1

    # Run backtest
    results = strategy.run_backtest(data_path)

    if results:
        logger.info("Multi-factor backtest completed successfully")

        # Quick summary
        metrics = results['metrics']
        print(f"\nQUICK RESULTS SUMMARY:")
        print(f"   Net Sharpe: {metrics['net_sharpe']:.3f}")
        print(f"   Net Return: {metrics['net_annual_return'] * 100:.2f}% annually")
        print(f"   Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
        significant = "YES" if metrics['p_value'] < 0.05 else "NO"
        print(f"   Significant: {significant}")
    else:
        logger.error("Multi-factor backtest failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())