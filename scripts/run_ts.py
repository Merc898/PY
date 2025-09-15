# scripts/run_ts.py
"""Run time-series momentum strategy"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from src.mlmom.config import Config
from src.mlmom.data import DataLoader
from src.mlmom.features import FeatureEngineer
from src.mlmom.labels import LabelGenerator
from src.mlmom.models import TSMomentumModel, calculate_sample_weights
from src.mlmom.backtest import Backtester
from src.mlmom.plots import MomentumPlotter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(config_path: str):
    """Main pipeline for time-series momentum"""
    logger.info("=" * 60)
    logger.info("TIME-SERIES MOMENTUM STRATEGY")
    logger.info("=" * 60)

    # Load configuration
    config = Config.from_yaml(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    # Set random seed
    np.random.seed(config.seed)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config.save(output_dir / "configs.json")

    # =========================================================================
    # 1. DATA LOADING
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 40)

    data_loader = DataLoader(config.data)
    df = data_loader.prepare_data()
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Number of symbols: {df['symbol'].nunique()}")

    # =========================================================================
    # 2. FEATURE ENGINEERING
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 40)

    feature_engineer = FeatureEngineer(config.features)
    df = feature_engineer.create_features(df, id_col='symbol', date_col='date')

    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    logger.info(f"Created {len(feature_cols)} features")

    # Save feature list
    with open(output_dir / "feature_list.json", 'w') as f:
        json.dump(feature_cols, f, indent=2)

    # =========================================================================
    # 3. LABEL GENERATION
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 3: GENERATING LABELS")
    logger.info("=" * 40)

    label_generator = LabelGenerator(config.labels)
    df = label_generator.create_labels(df, strategy_type='ts', id_col='symbol', date_col='date')

    # Remove rows with missing labels
    initial_len = len(df)
    df = df.dropna(subset=['label_sign'])
    logger.info(f"Removed {initial_len - len(df)} rows with missing labels")
    logger.info(f"Final dataset shape: {df.shape}")

    # =========================================================================
    # 4. CROSS-VALIDATION & MODEL TRAINING
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 4: CROSS-VALIDATION & TRAINING")
    logger.info("=" * 40)

    # Process each symbol separately for TS
    symbols = df['symbol'].unique()
    all_predictions = []

    for symbol in symbols[:50]:  # Process first 50 symbols for efficiency
        logger.info(f"\nProcessing {symbol}")

        # Filter data for this symbol
        symbol_df = df[df['symbol'] == symbol].copy()

        if len(symbol_df) < 500:  # Skip symbols with insufficient history
            continue

        # Prepare features and labels
        X = symbol_df[feature_cols]
        y = symbol_df['label_sign']

        # Calculate sample weights
        sample_weights = calculate_sample_weights(symbol_df, config.model.weight_type)

        # Simple train-test split (last 20% for test)
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        w_train = sample_weights.iloc[:split_point]

        if len(X_train) < 100 or len(X_test) < 20:
            continue

        # Train model
        model = TSMomentumModel(config.model)
        model.fit(X_train, y_train, w_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Store predictions
        pred_df = symbol_df.iloc[split_point:][['symbol', 'date']].copy()
        pred_df['prediction'] = predictions
        pred_df['actual'] = y_test.values
        all_predictions.append(pred_df)

    # Combine predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # Save predictions
    predictions_df.to_parquet(output_dir / "cv_predictions.parquet")
    logger.info(f"\nSaved predictions to {output_dir / 'cv_predictions.parquet'}")

    # =========================================================================
    # 5. BACKTESTING
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 5: BACKTESTING")
    logger.info("=" * 40)

    # Merge predictions with original data
    backtest_data = predictions_df.merge(
        df[['symbol', 'date', 'adj_close'] + [col for col in df.columns if col.startswith('feat_')]],
        on=['symbol', 'date']
    )

    # Run backtest
    backtester = Backtester(config)
    results = backtester.run_backtest(backtest_data, df, strategy_type='ts')

    # Save results
    results['trades'].to_parquet(output_dir / "trades.parquet")
    results['equity_curve'].to_parquet(output_dir / "equity_curve.parquet")

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(results['metrics'], f, indent=2)

    # Print key metrics
    logger.info("\n" + "=" * 40)
    logger.info("PERFORMANCE METRICS")
    logger.info("=" * 40)
    metrics = results['metrics']
    print(f"Net Sharpe Ratio: {metrics['sharpe_net']:.3f}")
    print(f"Net Annual Return: {metrics['annual_return_net']:.1%}")
    print(f"Max Drawdown: {metrics['max_drawdown_net']:.1%}")
    print(f"Hit Rate: {metrics['hit_rate_net']:.1%}")
    print(f"Average Turnover: {metrics['avg_turnover']:.1%}")

    # =========================================================================
    # 6. PLOTTING
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 6: GENERATING PLOTS")
    logger.info("=" * 40)

    plotter = MomentumPlotter(output_dir / "plots")
    plotter.plot_all(results, config)
    logger.info(f"Plots saved to {output_dir / 'plots'}")

    logger.info("\n" + "=" * 60)
    logger.info("TIME-SERIES MOMENTUM COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run time-series momentum strategy")
    parser.add_argument(
        "--configs",
        type=str,
        default="configs/ts.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    main(args.config)
