# src/mlmom/backtest.py
"""Backtesting engine for momentum strategies"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from src.mlmom.config import Config

logger = logging.getLogger(__name__)


class Backtester:
    """Backtesting engine for momentum strategies"""

    def __init__(self, config: Config):
        self.config = config
        self.results = {}

    def run_backtest(self, predictions: pd.DataFrame, prices: pd.DataFrame,
                     strategy_type: str = 'cs') -> Dict:
        """Run backtest on predictions"""
        logger.info(f"Running {strategy_type} backtest")

        if strategy_type == 'cs':
            return self._run_cs_backtest(predictions, prices)
        else:
            return self._run_ts_backtest(predictions, prices)

    def _run_cs_backtest(self, predictions: pd.DataFrame, prices: pd.DataFrame) -> Dict:
        """Run cross-sectional momentum backtest"""
        # Merge predictions with prices
        df = predictions.merge(
            prices[['symbol', 'date', 'adj_close', 'feat_amihud']],
            on=['symbol', 'date']
        )

        # Calculate positions
        df = self._calculate_cs_positions(df)

        # Calculate returns
        df['fwd_ret'] = df.groupby('symbol')['adj_close'].pct_change(self.config.labels.horizon).shift(
            -self.config.labels.horizon)

        # Calculate portfolio returns
        df = self._calculate_portfolio_returns(df)

        # Apply transaction costs
        df = self._apply_transaction_costs(df)

        # Calculate metrics
        metrics = self._calculate_metrics(df)

        return {
            'trades': df,
            'metrics': metrics,
            'equity_curve': self._calculate_equity_curve(df)
        }

    def _run_ts_backtest(self, predictions: pd.DataFrame, prices: pd.DataFrame) -> Dict:
        """Run time-series momentum backtest"""
        # Merge predictions with prices
        df = predictions.merge(
            prices[['symbol', 'date', 'adj_close', 'feat_vol_20']],
            on=['symbol', 'date']
        )

        # Calculate positions
        df = self._calculate_ts_positions(df)

        # Calculate returns
        df['fwd_ret'] = df.groupby('symbol')['adj_close'].pct_change(self.config.labels.horizon).shift(
            -self.config.labels.horizon)

        # Calculate portfolio returns
        df['port_ret'] = df['position'] * df['fwd_ret']

        # Apply transaction costs
        df = self._apply_transaction_costs(df)

        # Calculate metrics
        metrics = self._calculate_metrics(df)

        return {
            'trades': df,
            'metrics': metrics,
            'equity_curve': self._calculate_equity_curve(df)
        }

    def _calculate_cs_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cross-sectional positions"""
        # Rank predictions within each date
        df['pred_rank'] = df.groupby('date')['prediction'].rank(pct=True)

        # Determine long/short positions
        n_quantiles = self.config.labels.cs_quantiles
        df['position'] = 0.0

        # Long top quantile
        df.loc[df['pred_rank'] >= (1 - 1 / n_quantiles), 'position'] = 1.0

        # Short bottom quantile (unless long-only)
        if not self.config.portfolio.long_only:
            df.loc[df['pred_rank'] <= 1 / n_quantiles, 'position'] = -1.0

        # Apply position limits
        if self.config.portfolio.max_weight:
            # Equal weight within each group
            df['position'] = df.groupby(['date', 'position'])['position'].transform(
                lambda x: x / len(x) if len(x) > 0 else 0
            )
            # Apply max weight cap
            df['position'] = df['position'].clip(
                lower=-self.config.portfolio.max_weight,
                upper=self.config.portfolio.max_weight
            )

        # Normalize positions to sum to 0 (market neutral) or 1 (long-only)
        if self.config.portfolio.long_only:
            df['position'] = df.groupby('date')['position'].transform(
                lambda x: x / x[x > 0].sum() if x[x > 0].sum() > 0 else 0
            )
        else:
            # Market neutral
            df['position'] = df.groupby('date')['position'].transform(
                lambda x: x / np.abs(x).sum() if np.abs(x).sum() > 0 else 0
            )

        return df

    def _calculate_ts_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-series positions"""
        # Sign of prediction
        df['signal'] = np.sign(df['prediction'] - 0.5)  # Assuming prediction is probability

        # Volatility scaling
        if 'feat_vol_20' in df.columns:
            df['vol_scalar'] = self.config.portfolio.target_vol / (df['feat_vol_20'] + 1e-10)
            df['vol_scalar'] = df['vol_scalar'].clip(upper=1.0)
        else:
            df['vol_scalar'] = 1.0

        # Position sizing
        df['position'] = df['signal'] * df['vol_scalar']

        # Apply position limits
        df['position'] = df['position'].clip(lower=-1.0, upper=1.0)

        return df

    def _calculate_portfolio_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate portfolio returns"""
        # Portfolio return is position-weighted return
        df['port_ret'] = df['position'] * df['fwd_ret']

        # Aggregate across symbols for each date
        portfolio = df.groupby('date').agg({
            'port_ret': 'sum',
            'position': lambda x: np.abs(x).sum()  # Total exposure
        }).rename(columns={'position': 'gross_exposure'})

        df = df.merge(portfolio[['port_ret']], left_on='date', right_index=True,
                      suffixes=('_stock', ''))

        return df

    def _apply_transaction_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transaction costs"""
        # Calculate turnover
        df['position_lag'] = df.groupby('symbol')['position'].shift(1).fillna(0)
        df['turnover'] = np.abs(df['position'] - df['position_lag'])

        # Apply turnover cap if specified
        if self.config.portfolio.turnover_cap:
            df['turnover'] = df['turnover'].clip(upper=self.config.portfolio.turnover_cap)

        # Calculate costs
        if self.config.costs.dynamic_costs and 'feat_roll_spread' in df.columns:
            # Dynamic costs based on Roll spread
            df['trade_cost'] = df['turnover'] * df['feat_roll_spread'] * 100  # Convert to bps
        else:
            # Fixed costs
            df['trade_cost'] = df['turnover'] * self.config.costs.fixed_costs_bps / 10000

        # Apply turnover penalty if specified
        if self.config.costs.turnover_penalty > 0:
            df['trade_cost'] += df['turnover'] * self.config.costs.turnover_penalty / 10000

        # Net returns
        df['port_ret_net'] = df['port_ret'] - df['trade_cost']

        return df

    def _calculate_equity_curve(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cumulative equity curve"""
        # Aggregate by date
        daily_returns = df.groupby('date').agg({
            'port_ret': 'mean',
            'port_ret_net': 'mean',
            'turnover': 'mean',
            'trade_cost': 'mean'
        })

        # Calculate cumulative returns
        daily_returns['cum_ret_gross'] = (1 + daily_returns['port_ret']).cumprod()
        daily_returns['cum_ret_net'] = (1 + daily_returns['port_ret_net']).cumprod()

        # Calculate drawdowns
        daily_returns['peak_gross'] = daily_returns['cum_ret_gross'].expanding().max()
        daily_returns['drawdown_gross'] = (daily_returns['cum_ret_gross'] - daily_returns['peak_gross']) / \
                                          daily_returns['peak_gross']

        daily_returns['peak_net'] = daily_returns['cum_ret_net'].expanding().max()
        daily_returns['drawdown_net'] = (daily_returns['cum_ret_net'] - daily_returns['peak_net']) / daily_returns[
            'peak_net']

        return daily_returns

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        # Get daily returns
        equity = self._calculate_equity_curve(df)

        # Annualization factor
        if self.config.data.frequency == 'daily':
            ann_factor = 252
        else:
            ann_factor = 12

        metrics = {
            # Returns
            'total_return_gross': equity['cum_ret_gross'].iloc[-1] - 1,
            'total_return_net': equity['cum_ret_net'].iloc[-1] - 1,
            'annual_return_gross': (equity['cum_ret_gross'].iloc[-1] ** (ann_factor / len(equity))) - 1,
            'annual_return_net': (equity['cum_ret_net'].iloc[-1] ** (ann_factor / len(equity))) - 1,

            # Risk
            'volatility_gross': equity['port_ret'].std() * np.sqrt(ann_factor),
            'volatility_net': equity['port_ret_net'].std() * np.sqrt(ann_factor),
            'max_drawdown_gross': equity['drawdown_gross'].min(),
            'max_drawdown_net': equity['drawdown_net'].min(),

            # Risk-adjusted
            'sharpe_gross': (equity['port_ret'].mean() / equity['port_ret'].std()) * np.sqrt(ann_factor),
            'sharpe_net': (equity['port_ret_net'].mean() / equity['port_ret_net'].std()) * np.sqrt(ann_factor),
            'sortino_gross': (equity['port_ret'].mean() / equity['port_ret'][equity['port_ret'] < 0].std()) * np.sqrt(
                ann_factor),
            'sortino_net': (equity['port_ret_net'].mean() / equity['port_ret_net'][
                equity['port_ret_net'] < 0].std()) * np.sqrt(ann_factor),

            # Trading
            'avg_turnover': equity['turnover'].mean(),
            'avg_trade_cost_bps': equity['trade_cost'].mean() * 10000,

            # Hit rate
            'hit_rate_gross': (equity['port_ret'] > 0).mean(),
            'hit_rate_net': (equity['port_ret_net'] > 0).mean(),
        }

        # Information Coefficient (IC) for CS strategies
        if 'pred_rank' in df.columns:
            ic_by_date = df.groupby('date').apply(
                lambda x: x['prediction'].corr(x['fwd_ret'], method='spearman')
            )
            metrics['avg_ic'] = ic_by_date.mean()
            metrics['ic_std'] = ic_by_date.std()
            metrics['ic_ir'] = metrics['avg_ic'] / metrics['ic_std'] if metrics['ic_std'] > 0 else 0

        return metrics
