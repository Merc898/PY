# scripts/run_cs.py
"""Run cross-sectional momentum strategy"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from src.mlmom.config import Config
from src.mlmom.data import DataLoader
from src.mlmom.features import FeatureEngineer
from src.mlmom.labels import LabelGenerator
from src.mlmom.models import CSMomentumModel, calculate_sample_weights
from src.mlmom.cv import RollingWindowCV
from src.mlmom.backtest import Backtester
from src.mlmom.plots import MomentumPlotter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(config_path: str):
    """Main pipeline for cross-sectional momentum"""
    logger.info("=" * 60)
    logger.info("CROSS-SECTIONAL MOMENTUM STRATEGY")
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
    config.save(output_dir / "config.json")

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
    df = label_generator.create_labels(df, strategy_type='cs', id_col='symbol', date_col='date')

    # Remove rows with missing labels
    initial_len = len(df)
    df = df.dropna(subset=['label_reg'])
    logger.info(f"Removed {initial_len - len(df)} rows with missing labels")
    logger.info(f"Final dataset shape: {df.shape}")

    # =========================================================================
    # 4. CROSS-VALIDATION & MODEL TRAINING
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 4: CROSS-VALIDATION & TRAINING")
    logger.info("=" * 40)

    # Prepare features and labels
    X = df[feature_cols]
    y = df['label_reg']
    groups = df['date']

    # Calculate sample weights
    sample_weights = calculate_sample_weights(df, config.model.weight_type)

    # Initialize cross-validator
    cv = RollingWindowCV(
        initial_train_size=252,  # 1 year for daily data
        test_size=60,  # 3 months test
        refit_frequency=config.cv.refit_frequency,
        expanding=True,
        embargo_periods=config.cv.embargo_periods
    )

    # Store predictions
    all_predictions = []
    fold_metrics = []

    # Run cross-validation
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        logger.info(f"\nFold {fold + 1}")
        logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train = sample_weights.iloc[train_idx]

        # Train model
        model = CSMomentumModel(config.model)
        model.fit(X_train, y_train, w_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Store predictions with metadata
        pred_df = df.iloc[test_idx][['symbol', 'date']].copy()
        pred_df['prediction'] = predictions
        pred_df['actual'] = y_test.values
        pred_df['fold'] = fold
        all_predictions.append(pred_df)

        # Calculate fold metrics
        correlation = pd.Series(predictions).corr(pd.Series(y_test.values))
        fold_metrics.append({
            'fold': fold,
            'correlation': correlation,
            'rmse': np.sqrt(np.mean((predictions - y_test.values) ** 2))
        })
        logger.info(f"Fold {fold + 1} - Correlation: {correlation:.4f}")

    # Combine all predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # Save predictions
    predictions_df.to_parquet(output_dir / "cv_predictions.parquet")
    logger.info(f"\nSaved predictions to {output_dir / 'cv_predictions.parquet'}")

    # =========================================================================
    # 5. FINAL MODEL TRAINING
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 5: TRAINING FINAL MODEL")
    logger.info("=" * 40)

    # Train on all available data
    final_model = CSMomentumModel(config.model)
    final_model.fit(X, y, sample_weights)

    # Save model
    with open(output_dir / "model.pkl", 'wb') as f:
        pickle.dump(final_model, f)
    logger.info(f"Saved model to {output_dir / 'model.pkl'}")

    # Get feature importance
    feature_importance = final_model.get_feature_importance()
    if not feature_importance.empty:
        feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
        logger.info(f"Top 10 important features:")
        print(feature_importance.head(10))

    # =========================================================================
    # 6. BACKTESTING
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 6: BACKTESTING")
    logger.info("=" * 40)

    # Merge predictions with original data
    backtest_data = predictions_df.merge(
        df[['symbol', 'date', 'adj_close'] + [col for col in df.columns if col.startswith('feat_')]],
        on=['symbol', 'date']
    )

    # Run backtest
    backtester = Backtester(config)
    results = backtester.run_backtest(backtest_data, df, strategy_type='cs')

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
    print(f"Average Turnover: {metrics['avg_turnover']:.1%}")
    print(f"Average IC: {metrics.get('avg_ic', 0):.4f}")
    print(f"IC IR: {metrics.get('ic_ir', 0):.3f}")

    # =========================================================================
    # 7. PLOTTING
    # =========================================================================
    logger.info("\n" + "=" * 40)
    logger.info("STEP 7: GENERATING PLOTS")
    logger.info("=" * 40)

    plotter = MomentumPlotter(output_dir / "plots")

    # Add IC series if available
    if 'avg_ic' in metrics:
        ic_by_date = backtest_data.groupby('date').apply(
            lambda x: x['prediction'].corr(x['actual'], method='spearman')
        )
        results['ic_series'] = ic_by_date

    # Add feature importance to results
    if not feature_importance.empty:
        results['feature_importance'] = feature_importance

    plotter.plot_all(results, config)
    logger.info(f"Plots saved to {output_dir / 'plots'}")

    logger.info("\n" + "=" * 60)
    logger.info("CROSS-SECTIONAL MOMENTUM COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cross-sectional momentum strategy")
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/configs/cs.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    main(args.config)
