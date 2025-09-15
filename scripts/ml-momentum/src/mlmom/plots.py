# src/mlmom/plots.py
"""Plotting utilities for momentum strategies"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


class MomentumPlotter:
    """Generate plots for momentum strategies"""

    def __init__(self, output_dir: str = "outputs/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_all(self, results: Dict, config: Config):
        """Generate all plots"""
        logger.info("Generating plots")

        # Equity curve
        self.plot_equity_curve(results['equity_curve'])

        # Rolling metrics
        self.plot_rolling_metrics(results['trades'])

        # Feature importance
        if 'feature_importance' in results:
            self.plot_feature_importance(results['feature_importance'])

        # IC analysis for CS
        if config.portfolio.strategy_type == 'cs' and 'ic_series' in results:
            self.plot_ic_analysis(results['ic_series'])

    def plot_equity_curve(self, equity: pd.DataFrame):
        """Plot cumulative returns and drawdowns"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Cumulative returns
        ax = axes[0]
        ax.plot(equity.index, equity['cum_ret_gross'], label='Gross', linewidth=2)
        ax.plot(equity.index, equity['cum_ret_net'], label='Net', linewidth=2, alpha=0.8)
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Momentum Strategy Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Drawdowns
        ax = axes[1]
        ax.fill_between(equity.index, 0, equity['drawdown_gross'],
                        alpha=0.3, label='Gross DD')
        ax.fill_between(equity.index, 0, equity['drawdown_net'],
                        alpha=0.5, label='Net DD')
        ax.set_ylabel('Drawdown')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'equity_curve.png', dpi=100, bbox_inches='tight')
        plt.close()

    def plot_rolling_metrics(self, trades: pd.DataFrame):
        """Plot rolling turnover and costs"""
        # Aggregate by date
        daily = trades.groupby('date').agg({
            'turnover': 'mean',
            'trade_cost': 'mean'
        })

        # Calculate rolling averages
        daily['turnover_ma'] = daily['turnover'].rolling(20).mean()
        daily['cost_ma'] = daily['trade_cost'].rolling(20).mean() * 10000  # Convert to bps

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Turnover
        ax = axes[0]
        ax.plot(daily.index, daily['turnover_ma'], linewidth=1.5)
        ax.set_ylabel('Turnover')
        ax.set_title('Rolling Turnover and Transaction Costs')
        ax.grid(True, alpha=0.3)

        # Costs
        ax = axes[1]
        ax.plot(daily.index, daily['cost_ma'], linewidth=1.5, color='red', alpha=0.7)
        ax.set_ylabel('Cost (bps)')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'rolling_metrics.png', dpi=100, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, importance: pd.DataFrame, top_n: int = 20):
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get top features
        top_features = importance.head(top_n)

        # Create horizontal bar plot
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=100, bbox_inches='tight')
        plt.close()

    def plot_ic_analysis(self, ic_series: pd.Series):
        """Plot Information Coefficient analysis"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Rolling IC
        ax = axes[0]
        ic_ma = ic_series.rolling(20).mean()
        ax.plot(ic_series.index, ic_series, alpha=0.3, label='Daily IC')
        ax.plot(ic_series.index, ic_ma, linewidth=2, label='20-day MA')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylabel('Information Coefficient')
        ax.set_title('Rolling Information Coefficient')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Cumulative IC
        ax = axes[1]
        cum_ic = ic_series.cumsum()
        ax.plot(ic_series.index, cum_ic, linewidth=2)
        ax.set_ylabel('Cumulative IC')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'ic_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
