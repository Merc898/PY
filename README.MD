# ============================================================================
# README.md
# ML Momentum Strategy

Production-grade Python implementation of Machine Learning-based momentum strategies using only OHLCV data.

## Overview

This repository implements both cross-sectional and time-series momentum strategies using machine learning models. The system is designed to work exclusively with split/dividend-adjusted OHLCV data, without requiring any fundamental data, macroeconomic indicators, or external calendars.

## Features

### Strategies
- **Cross-Sectional Momentum**: Rank stocks by predicted returns and go long/short top/bottom quantiles
- **Time-Series Momentum**: Predict each asset's direction and size independently

### Feature Engineering (OHLCV-derived only)
- **Returns**: Multiple period returns with skip-month momentum
- **Trend**: Moving averages, MACD, Donchian channels, linear trends
- **Volatility**: Rolling volatility, downside deviation, Parkinson range, ATR
- **Microstructure**: Amihud illiquidity, Roll spread estimator, overnight gaps
- **Volume**: Dollar volume, OBV, Chaikin Money Flow, VWAP deviation
- **Calendar**: Day-of-week, month effects (from timestamps only)

### Models
- LightGBM, XGBoost, Gradient Boosting
- Linear models (Lasso, Logistic Regression)
- Support for LSTM/TCN (placeholder for deep learning)

### Cross-Validation
- Combinatorial Purged K-Fold with embargo
- Rolling window with expanding/fixed windows
- No data leakage across time

### Risk Management
- Transaction cost modeling (fixed or dynamic)
- Position limits and turnover constraints
- Volatility targeting for time-series strategies
- Sample weighting by liquidity

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ml-momentum

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
pyyaml>=5.4.0
```

## Project Structure

```
ml-momentum/
├── src/
│   └── mlmom/
│       ├── __init__.py
│       ├── config.py       # Configuration management
│       ├── data.py         # Data loading and alignment
│       ├── features.py     # Feature engineering
│       ├── labels.py       # Label generation
│       ├── models.py       # Model implementations
│       ├── cv.py          # Cross-validation utilities
│       ├── backtest.py    # Backtesting engine
│       ├── costs.py       # Transaction cost models
│       └── plots.py       # Visualization utilities
├── scripts/
│   ├── run_cs.py         # Run cross-sectional strategy
│   └── run_ts.py         # Run time-series strategy
├── configs/
│   ├── cs.yaml           # Cross-sectional config
│   └── ts.yaml           # Time-series config
├── tests/
│   └── test_mlmom.py     # Unit tests
├── outputs/              # Results directory (created at runtime)
└── README.md
```
