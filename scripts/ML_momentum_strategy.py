# ============================================================================
# requirements.txt
"""
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
pyyaml>=5.4.0
scipy>=1.7.0
joblib>=1.1.0
pyarrow>=6.0.0
"""

# ============================================================================
# setup.py
"""Setup script for ML Momentum package"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mlmom",
    version="1.0.0",
    author="ML Momentum Team",
    description="Machine Learning Momentum Strategy using OHLCV data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-momentum",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mlmom-cs=scripts.run_cs:main",
            "mlmom-ts=scripts.run_ts:main",
        ],
    },
)

# ============================================================================
# scripts/data_diagnostics.py
"""Diagnostic script to analyze input data quality"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_diagnostics(data_path: str, output_dir: str = "outputs/diagnostics"):
    """Run comprehensive data diagnostics"""
    logger.info("=" * 60)
    logger.info("DATA DIAGNOSTICS")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {data_path}")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    results = {}

    # =========================================================================
    # 1. Data Overview
    # =========================================================================
    logger.info("\n1. Data Overview")
    logger.info("-" * 40)

    overview = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': list(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }

    if 'symbol' in df.columns:
        overview['unique_symbols'] = df['symbol'].nunique()
        overview['symbols_list'] = sorted(df['symbol'].unique().tolist())[:10]  # First 10

    if 'date' in df.columns:
        overview['date_range'] = f"{df['date'].min()} to {df['date'].max()}"
        overview['total_days'] = (df['date'].max() - df['date'].min()).days
        overview['unique_dates'] = df['date'].nunique()

    results['overview'] = overview

    for key, value in overview.items():
        if key != 'symbols_list' and key != 'columns':
            logger.info(f"  {key}: {value}")

    # =========================================================================
    # 2. Missing Data Analysis
    # =========================================================================
    logger.info("\n2. Missing Data Analysis")
    logger.info("-" * 40)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    missing_summary = {}
    for col in df.columns:
        if missing[col] > 0:
            missing_summary[col] = {
                'count': int(missing[col]),
                'percentage': round(missing_pct[col], 2)
            }
            logger.info(f"  {col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")

    if not missing_summary:
        logger.info("  No missing values found!")

    results['missing_data'] = missing_summary

    # =========================================================================
    # 3. Data Quality Checks
    # =========================================================================
    logger.info("\n3. Data Quality Checks")
    logger.info("-" * 40)

    quality_issues = []

    # Check for duplicates
    if 'symbol' in df.columns and 'date' in df.columns:
        duplicates = df.duplicated(subset=['symbol', 'date']).sum()
        if duplicates > 0:
            quality_issues.append(f"Found {duplicates} duplicate (symbol, date) pairs")
            logger.warning(f"  ⚠️ Found {duplicates} duplicate entries")

    # Check OHLC consistency
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        invalid_ohlc = ((df['high'] < df['low']) |
                        (df['high'] < df['open']) |
                        (df['high'] < df['close']) |
                        (df['low'] > df['open']) |
                        (df['low'] > df['close'])).sum()
        if invalid_ohlc > 0:
            quality_issues.append(f"Found {invalid_ohlc} rows with invalid OHLC relationships")
            logger.warning(f"  ⚠️ Found {invalid_ohlc} invalid OHLC relationships")

    # Check for zero/negative prices
    price_cols = ['open', 'high', 'low', 'close', 'adj_close']
    for col in price_cols:
        if col in df.columns:
            zero_prices = (df[col] <= 0).sum()
            if zero_prices > 0:
                quality_issues.append(f"Found {zero_prices} zero/negative values in {col}")
                logger.warning(f"  ⚠️ Found {zero_prices} zero/negative {col} prices")

    # Check for extreme returns (potential data errors)
    if 'close' in df.columns and 'symbol' in df.columns:
        df['daily_return'] = df.groupby('symbol')['close'].pct_change()
        extreme_returns = (df['daily_return'].abs() > 0.5).sum()  # >50% daily move
        if extreme_returns > 0:
            quality_issues.append(f"Found {extreme_returns} extreme daily returns (>50%)")
            logger.warning(f"  ⚠️ Found {extreme_returns} extreme returns")

    if not quality_issues:
        logger.info("  ✓ No quality issues detected")

    results['quality_issues'] = quality_issues

    # =========================================================================
    # 4. Symbol Coverage Analysis
    # =========================================================================
    if 'symbol' in df.columns and 'date' in df.columns:
        logger.info("\n4. Symbol Coverage Analysis")
        logger.info("-" * 40)

        symbol_stats = df.groupby('symbol').agg({
            'date': ['min', 'max', 'count']
        })
        symbol_stats.columns = ['start_date', 'end_date', 'observations']

        coverage_summary = {
            'avg_observations': symbol_stats['observations'].mean(),
            'min_observations': symbol_stats['observations'].min(),
            'max_observations': symbol_stats['observations'].max(),
            'symbols_with_full_history': (symbol_stats['observations'] == df['date'].nunique()).sum(),
            'symbols_with_recent_data': (symbol_stats['end_date'] == df['date'].max()).sum()
        }

        for key, value in coverage_summary.items():
            logger.info(f"  {key}: {value:.0f}")

        results['coverage'] = coverage_summary

        # Find symbols with gaps
        symbols_with_gaps = []
        for symbol in df['symbol'].unique()[:50]:  # Check first 50 symbols
            symbol_data = df[df['symbol'] == symbol]['date'].sort_values()
            expected_days = pd.date_range(symbol_data.min(), symbol_data.max(), freq='D')
            if len(symbol_data) < len(expected_days) * 0.5:  # Less than 50% of expected days
                symbols_with_gaps.append(symbol)

        if symbols_with_gaps:
            logger.warning(f"  ⚠️ {len(symbols_with_gaps)} symbols have significant gaps")
            results['symbols_with_gaps'] = symbols_with_gaps[:10]  # Store first 10

    # =========================================================================
    # 5. Statistical Summary
    # =========================================================================
    logger.info("\n5. Statistical Summary")
    logger.info("-" * 40)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_summary = {}

    for col in numeric_cols:
        if col in ['close', 'volume', 'daily_return']:  # Focus on key columns
            stats = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skew': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
            stats_summary[col] = stats

            logger.info(f"  {col}:")
            logger.info(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            logger.info(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")

    results['statistics'] = stats_summary

    # =========================================================================
    # 6. Save Results
    # =========================================================================
    logger.info("\n6. Saving Results")
    logger.info("-" * 40)

    # Save JSON report
    json_path = output_dir / 'diagnostic_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  Saved JSON report to {json_path}")

    # Save text report
    text_path = output_dir / 'diagnostic_report.txt'
    with open(text_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DATA DIAGNOSTICS REPORT\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 60 + "\n\n")

        for section, data in results.items():
            f.write(f"\n{section.upper()}\n")
            f.write("-" * 40 + "\n")
            f.write(json.dumps(data, indent=2, default=str) + "\n")

    logger.info(f"  Saved text report to {text_path}")

    # Save sample data
    sample_path = output_dir / 'data_sample.csv'
    df.head(1000).to_csv(sample_path, index=False)
    logger.info(f"  Saved data sample to {sample_path}")

    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTICS COMPLETE")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data diagnostics")
    parser.add_argument("--data", type=str, required=True, help="Path to data file")
    parser.add_argument("--output", type=str, default="outputs/diagnostics",
                        help="Output directory")
    args = parser.parse_args()

    run_diagnostics(args.data, args.output)

# ============================================================================
# scripts/hyperparameter_tuning.py
"""Hyperparameter tuning script for momentum models"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import optuna
from sklearn.model_selection import TimeSeriesSplit

from src.mlmom.config import Config, ModelConfig
from src.mlmom.data import DataLoader
from src.mlmom.features import FeatureEngineer
from src.mlmom.labels import LabelGenerator
from src.mlmom.models import CSMomentumModel, TSMomentumModel, calculate_sample_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MomentumOptimizer:
    """Hyperparameter optimizer for momentum strategies"""

    def __init__(self, config: Config, strategy_type: str = 'cs'):
        self.config = config
        self.strategy_type = strategy_type
        self.best_params = None
        self.best_score = None

    def objective(self, trial, X_train, y_train, X_val, y_val, sample_weights=None):
        """Objective function for Optuna"""

        # Suggest hyperparameters based on model type
        if self.config.model.model_type == 'lightgbm':
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            }
        elif self.config.model.model_type == 'xgboost':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }
        else:
            params = self.config.model.params

        # Create model with suggested parameters
        model_config = ModelConfig(
            model_type=self.config.model.model_type,
            params=params,
            use_sample_weights=self.config.model.use_sample_weights,
            weight_type=self.config.model.weight_type
        )

        # Train model
        if self.strategy_type == 'cs':
            model = CSMomentumModel(model_config)
        else:
            model = TSMomentumModel(model_config)

        if sample_weights is not None and self.config.model.use_sample_weights:
            model.fit(X_train, y_train, sample_weights)
        else:
            model.fit(X_train, y_train)

        # Evaluate on validation set
        predictions = model.predict(X_val)

        # Calculate metric (IC for CS, accuracy for TS)
        if self.strategy_type == 'cs':
            # Information Coefficient
            score = pd.Series(predictions).corr(pd.Series(y_val.values), method='spearman')
        else:
            # Classification accuracy
            from sklearn.metrics import roc_auc_score
            score = roc_auc_score(y_val, predictions)

        return score

    def optimize(self, X, y, sample_weights=None, n_trials=100, n_splits=3):
        """Run hyperparameter optimization"""
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")

        # Create time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)

        def objective_cv(trial):
            """Cross-validated objective"""
            scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                if sample_weights is not None:
                    w_train = sample_weights.iloc[train_idx]
                else:
                    w_train = None

                score = self.objective(
                    trial, X_train, y_train, X_val, y_val, w_train
                )
                scores.append(score)

            return np.mean(scores)

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config.seed)
        )

        # Optimize
        study.optimize(objective_cv, n_trials=n_trials, n_jobs=1)

        # Store best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value

        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return study


def main(config_path: str, n_trials: int = 100):
    """Main hyperparameter tuning pipeline"""
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING")
    logger.info("=" * 60)

    # Load configuration
    config = Config.from_yaml(config_path)
    strategy_type = config.portfolio.strategy_type

    # Create output directory
    output_dir = Path(config.output_dir) / "tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    logger.info("Loading data...")
    data_loader = DataLoader(config.data)
    df = data_loader.prepare_data()

    # Create features
    logger.info("Creating features...")
    feature_engineer = FeatureEngineer(config.features)
    df = feature_engineer.create_features(df)

    # Create labels
    logger.info("Creating labels...")
    label_generator = LabelGenerator(config.labels)
    df = label_generator.create_labels(df, strategy_type=strategy_type)

    # Prepare data
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    df = df.dropna(subset=['label_reg' if strategy_type == 'cs' else 'label_sign'])

    X = df[feature_cols]
    y = df['label_reg' if strategy_type == 'cs' else 'label_sign']
    sample_weights = calculate_sample_weights(df, config.model.weight_type)

    # Run optimization
    optimizer = MomentumOptimizer(config, strategy_type)
    study = optimizer.optimize(X, y, sample_weights, n_trials=n_trials)

    # Save results
    results = {
        'best_params': optimizer.best_params,
        'best_score': optimizer.best_score,
        'n_trials': n_trials,
        'strategy_type': strategy_type
    }

    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save study
    study.trials_dataframe().to_csv(output_dir / 'trials.csv', index=False)

    # Update config with best parameters
    config.model.params = optimizer.best_params
    config.save(output_dir / 'optimized_config.yaml')

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("=" * 60)
    logger.info("TUNING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    args = parser.parse_args()

    main(args.config, args.trials)

# ============================================================================
# scripts/compare_strategies.py
"""Compare multiple momentum strategies"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_strategy_results(output_dir: str) -> Dict:
    """Load results from a strategy output directory"""
    output_dir = Path(output_dir)

    results = {}

    # Load metrics
    metrics_path = output_dir / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            results['metrics'] = json.load(f)

    # Load equity curve
    equity_path = output_dir / 'equity_curve.parquet'
    if equity_path.exists():
        results['equity_curve'] = pd.read_parquet(equity_path)

    # Load trades
    trades_path = output_dir / 'trades.parquet'
    if trades_path.exists():
        results['trades'] = pd.read_parquet(trades_path)

    return results


def compare_strategies(strategy_dirs: Dict[str, str], output_dir: str = "outputs/comparison"):
    """Compare multiple strategies"""
    logger.info("=" * 60)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all strategies
    strategies = {}
    for name, path in strategy_dirs.items():
        logger.info(f"Loading {name} from {path}")
        strategies[name] = load_strategy_results(path)

    # =========================================================================
    # 1. Metrics Comparison
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("PERFORMANCE METRICS")
    logger.info("=" * 40)

    metrics_df = pd.DataFrame()
    for name, results in strategies.items():
        if 'metrics' in results:
            metrics_df[name] = pd.Series(results['metrics'])

    # Display key metrics
    key_metrics = [
        'sharpe_net', 'annual_return_net', 'max_drawdown_net',
        'volatility_net', 'avg_turnover', 'hit_rate_net'
    ]

    comparison_table = metrics_df.loc[
        [m for m in key_metrics if m in metrics_df.index]
    ].round(3)

    print("\n" + comparison_table.to_string())

    # Save comparison table
    comparison_table.to_csv(output_dir / 'metrics_comparison.csv')

    # =========================================================================
    # 2. Plot Cumulative Returns
    # =========================================================================
    logger.info("\nGenerating comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Cumulative returns
    ax = axes[0, 0]
    for name, results in strategies.items():
        if 'equity_curve' in results:
            equity = results['equity_curve']
            ax.plot(equity.index, equity['cum_ret_net'], label=name, linewidth=2)
    ax.set_title('Cumulative Returns (Net)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Drawdowns
    ax = axes[0, 1]
    for name, results in strategies.items():
        if 'equity_curve' in results:
            equity = results['equity_curve']
            ax.plot(equity.index, equity['drawdown_net'], label=name, linewidth=1.5)
    ax.set_title('Drawdowns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rolling Sharpe
    ax = axes[1, 0]
    for name, results in strategies.items():
        if 'equity_curve' in results:
            equity = results['equity_curve']
            rolling_sharpe = (
                    equity['port_ret_net'].rolling(252).mean() /
                    equity['port_ret_net'].rolling(252).std() * np.sqrt(252)
            )
            ax.plot(equity.index, rolling_sharpe, label=name, linewidth=1.5)
    ax.set_title('Rolling Sharpe Ratio (252-day)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Metrics bar chart
    ax = axes[1, 1]
    metrics_to_plot = ['sharpe_net', 'annual_return_net']
    plot_data = comparison_table.loc[
        [m for m in metrics_to_plot if m in comparison_table.index]
    ]
    plot_data.T.plot(kind='bar', ax=ax)
    ax.set_title('Key Metrics Comparison')
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Value')
    ax.legend(title='Metric')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'strategy_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 3. Correlation Analysis
    # =========================================================================
    logger.info("\nCalculating strategy correlations...")

    returns_df = pd.DataFrame()
    for name, results in strategies.items():
        if 'equity_curve' in results:
            returns_df[name] = results['equity_curve']['port_ret_net']

    if len(returns_df.columns) > 1:
        corr_matrix = returns_df.corr()

        # Plot correlation heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Strategy Return Correlations')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrix.png', dpi=100, bbox_inches='tight')
        plt.close()

        # Save correlation matrix
        corr_matrix.to_csv(output_dir / 'correlation_matrix.csv')

    # =========================================================================
    # 4. Summary Report
    # =========================================================================
    with open(output_dir / 'comparison_report.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("STRATEGY COMPARISON REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(comparison_table.to_string() + "\n\n")

        if len(returns_df.columns) > 1:
            f.write("CORRELATION MATRIX\n")
            f.write("-" * 40 + "\n")
            f.write(corr_matrix.to_string() + "\n\n")

        # Best strategy by metric
        f.write("BEST STRATEGY BY METRIC\n")
        f.write("-" * 40 + "\n")
        for metric in key_metrics:
            if metric in comparison_table.index:
                if 'drawdown' in metric:
                    best = comparison_table.loc[metric].idxmax()  # Less negative is better
                else:
                    best = comparison_table.loc[metric].idxmax()
                value = comparison_table.loc[metric, best]
                f.write(f"{metric:20s}: {best:15s} ({value:.3f})\n")

    logger.info(f"\nComparison results saved to {output_dir}")
    logger.info("=" * 60)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Example usage
    strategy_dirs = {
        'Cross-Sectional': 'outputs/cs',
        'Time-Series': 'outputs/ts',
    }

    compare_strategies(strategy_dirs)

# ============================================================================
# Makefile
"""
# ML Momentum Strategy Makefile

.PHONY: install test run-cs run-ts tune clean help

help:
	@echo "ML Momentum Strategy - Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test      - Run unit tests"
	@echo "  make run-cs    - Run cross-sectional strategy"
	@echo "  make run-ts    - Run time-series strategy"
	@echo "  make tune-cs   - Tune cross-sectional hyperparameters"
	@echo "  make tune-ts   - Tune time-series hyperparameters"
	@echo "  make diagnose  - Run data diagnostics"
	@echo "  make compare   - Compare strategies"
	@echo "  make clean     - Clean output files"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest tests/ -v

run-cs:
	python scripts/run_cs.py --config configs/cs.yaml

run-ts:
	python scripts/run_ts.py --config configs/ts.yaml

tune-cs:
	python scripts/hyperparameter_tuning.py --config configs/cs.yaml --trials 100

tune-ts:
	python scripts/hyperparameter_tuning.py --config configs/ts.yaml --trials 100

diagnose:
	python scripts/data_diagnostics.py --data "C:/Desktop/BA/PY/my_project/scripts/data/cleaned/ML_data_cleaned.parquet"

compare:
	python scripts/compare_strategies.py

clean:
	rm -rf outputs/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: install test run-cs run-ts compare
"""