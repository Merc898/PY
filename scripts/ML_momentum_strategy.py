# src/mlmom/__init__.py
"""Machine Learning Momentum Strategy Package"""
__version__ = "1.0.0"

# ============================================================================
# src/mlmom/configs.py
"""Configuration management for ML momentum strategies"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import json


@dataclass
class DataConfig:
    """Data loading and processing configuration"""
    prices_path: str
    id_column: str = "symbol"
    date_column: str = "date"
    price_columns: List[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    adj_close_column: str = "close"  # Using close as adj_close since data is already adjusted
    frequency: str = "daily"  # daily or monthly
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    currency: str = "USD"


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    return_periods: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 60, 120])
    skip_periods: int = 1  # Skip most recent period for momentum
    ma_periods: List[int] = field(default_factory=lambda: [10, 20, 50, 200])
    volatility_windows: List[int] = field(default_factory=lambda: [20, 60])
    volume_windows: List[int] = field(default_factory=lambda: [20])
    winsorize_quantile: float = 0.01
    standardize: bool = True
    fillna_method: str = "median"


@dataclass
class LabelConfig:
    """Label generation configuration"""
    horizon: int = 20  # Prediction horizon in periods
    cs_quantiles: int = 5  # Number of quantiles for cross-sectional
    cs_top_bottom_only: bool = True  # Use only top/bottom quantiles
    ts_vol_target: float = 0.15  # Target volatility for TS
    vol_lookback: int = 60  # Lookback for volatility estimation


@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str = "lightgbm"  # lightgbm, xgboost, linear, lstm, tcn
    params: Dict[str, Any] = field(default_factory=dict)
    use_sample_weights: bool = True
    weight_type: str = "amihud"  # amihud or equal


@dataclass
class CVConfig:
    """Cross-validation configuration"""
    n_splits: int = 5
    embargo_periods: int = 10
    purge_periods: int = 10
    refit_frequency: int = 60  # Refit model every N periods
    validation_size: float = 0.2
    test_size: float = 0.2


@dataclass
class PortfolioConfig:
    """Portfolio construction configuration"""
    strategy_type: str = "cs"  # cs or ts
    long_only: bool = False
    max_weight: float = 0.05
    target_vol: float = 0.15
    rebalance_frequency: int = 20  # Rebalance every N periods
    turnover_cap: Optional[float] = None


@dataclass
class CostConfig:
    """Transaction cost configuration"""
    fixed_costs_bps: float = 10.0  # Basis points per round trip
    dynamic_costs: bool = False
    cost_model: str = "roll"  # roll or amihud
    turnover_penalty: float = 0.0


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig
    features: FeatureConfig
    labels: LabelConfig
    model: ModelConfig
    cv: CVConfig
    portfolio: PortfolioConfig
    costs: CostConfig
    output_dir: str = "outputs"
    seed: int = 42
    n_jobs: int = -1
    verbose: int = 1

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(
            data=DataConfig(**data.get('data', {})),
            features=FeatureConfig(**data.get('features', {})),
            labels=LabelConfig(**data.get('labels', {})),
            model=ModelConfig(**data.get('model', {})),
            cv=CVConfig(**data.get('cv', {})),
            portfolio=PortfolioConfig(**data.get('portfolio', {})),
            costs=CostConfig(**data.get('costs', {})),
            **{k: v for k, v in data.items()
               if k not in ['data', 'features', 'labels', 'model', 'cv', 'portfolio', 'costs']}
        )

    def to_dict(self) -> Dict:
        """Convert configs to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = {k: v for k, v in value.__dict__.items()}
            else:
                result[key] = value
        return result

    def save(self, path: str):
        """Save configuration to file"""
        path = Path(path)
        if path.suffix == '.yaml':
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# src/mlmom/data.py
"""Data loading and preprocessing utilities"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """OHLCV data loader and preprocessor"""

    def __init__(self, config: DataConfig):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        """Load and validate OHLCV data"""
        logger.info(f"Loading data from {self.config.prices_path}")

        # Load data
        if self.config.prices_path.endswith('.parquet'):
            df = pd.read_parquet(self.config.prices_path)
        else:
            df = pd.read_csv(self.config.prices_path)

        # Ensure date column is datetime
        df[self.config.date_column] = pd.to_datetime(df[self.config.date_column])

        # Filter date range if specified
        if self.config.start_date:
            df = df[df[self.config.date_column] >= pd.to_datetime(self.config.start_date)]
        if self.config.end_date:
            df = df[df[self.config.date_column] <= pd.to_datetime(self.config.end_date)]

        # Sort by id and date
        df = df.sort_values([self.config.id_column, self.config.date_column])

        # Validate required columns
        required = [self.config.id_column, self.config.date_column] + self.config.price_columns
        missing = set(required) - set(df.columns)
        if missing:
            # If adj_close is missing, use close as adj_close
            if 'adj_close' in missing and 'close' in df.columns:
                df['adj_close'] = df['close']
                logger.warning("adj_close not found, using close as adj_close")
            else:
                raise ValueError(f"Missing required columns: {missing}")

        # Ensure we have adj_close
        if 'adj_close' not in df.columns:
            if self.config.adj_close_column in df.columns:
                df['adj_close'] = df[self.config.adj_close_column]
            else:
                df['adj_close'] = df['close']

        logger.info(f"Loaded {len(df)} rows with {df[self.config.id_column].nunique()} unique symbols")
        logger.info(f"Date range: {df[self.config.date_column].min()} to {df[self.config.date_column].max()}")

        return df

    def align_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align data by creating regular grid of dates x symbols"""
        logger.info("Aligning data on regular date grid")

        # Get unique dates and symbols
        dates = df[self.config.date_column].unique()
        symbols = df[self.config.id_column].unique()

        # Create complete grid
        grid = pd.MultiIndex.from_product(
            [symbols, dates],
            names=[self.config.id_column, self.config.date_column]
        )

        # Reindex to complete grid
        df_indexed = df.set_index([self.config.id_column, self.config.date_column])
        df_aligned = df_indexed.reindex(grid)

        # Forward fill prices within reasonable limits (5 days for daily data)
        price_cols = ['open', 'high', 'low', 'close', 'adj_close']
        existing_price_cols = [col for col in price_cols if col in df_aligned.columns]

        if self.config.frequency == 'daily':
            limit = 5
        else:
            limit = 1

        df_aligned[existing_price_cols] = df_aligned.groupby(level=0)[existing_price_cols].ffill(limit=limit)

        # Volume should not be forward filled
        if 'volume' in df_aligned.columns:
            df_aligned['volume'] = df_aligned.groupby(level=0)['volume'].fillna(0)

        df_aligned = df_aligned.reset_index()

        logger.info(f"Aligned data shape: {df_aligned.shape}")
        return df_aligned

    def calculate_returns(self, df: pd.DataFrame, periods: List[int] = [1]) -> pd.DataFrame:
        """Calculate returns for specified periods"""
        for period in periods:
            df[f'ret_{period}'] = df.groupby(self.config.id_column)['adj_close'].pct_change(period)
        return df

    def prepare_data(self) -> pd.DataFrame:
        """Complete data preparation pipeline"""
        df = self.load_data()
        df = self.align_data(df)
        df = self.calculate_returns(df, [1])  # Add basic returns
        return df


# ============================================================================
# src/mlmom/features.py
"""Feature engineering for momentum strategies"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from scipy import stats
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """OHLCV-based feature engineering"""

    def __init__(self, config: FeatureConfig):
        self.config = config

    def create_features(self, df: pd.DataFrame, id_col: str = 'symbol',
                        date_col: str = 'date') -> pd.DataFrame:
        """Create all features from OHLCV data"""
        logger.info("Starting feature engineering")

        # Sort data
        df = df.sort_values([id_col, date_col])

        # Price-based features
        df = self._create_return_features(df, id_col)
        df = self._create_trend_features(df, id_col)
        df = self._create_volatility_features(df, id_col)
        df = self._create_drawdown_features(df, id_col)

        # Volume-based features
        df = self._create_volume_features(df, id_col)
        df = self._create_liquidity_features(df, id_col)

        # Microstructure features
        df = self._create_gap_features(df, id_col)
        df = self._create_spread_proxies(df, id_col)

        # Calendar features
        df = self._create_calendar_features(df, date_col)

        # Clean and standardize
        df = self._clean_features(df, id_col, date_col)

        logger.info(f"Created {len([c for c in df.columns if c.startswith('feat_')])} features")
        return df

    def _create_return_features(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Create return-based momentum features"""
        for period in self.config.return_periods:
            # Simple returns
            df[f'feat_ret_{period}'] = df.groupby(id_col)['adj_close'].pct_change(period)

            # Compound returns
            df[f'feat_cret_{period}'] = df.groupby(id_col)['adj_close'].transform(
                lambda x: (x / x.shift(period)) - 1
            )

            # Skip-month momentum (e.g., 12-2)
            if period > self.config.skip_periods:
                df[f'feat_ret_{period}_{self.config.skip_periods}'] = df.groupby(id_col)['adj_close'].transform(
                    lambda x: (x.shift(self.config.skip_periods) / x.shift(period)) - 1
                )

        return df

    def _create_trend_features(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Create trend and price location features"""
        for period in self.config.ma_periods:
            # Simple moving average
            df[f'feat_sma_{period}'] = df.groupby(id_col)['adj_close'].transform(
                lambda x: x.rolling(period).mean()
            )
            df[f'feat_sma_gap_{period}'] = (df['adj_close'] - df[f'feat_sma_{period}']) / df[f'feat_sma_{period}']

            # Exponential moving average
            df[f'feat_ema_{period}'] = df.groupby(id_col)['adj_close'].transform(
                lambda x: x.ewm(span=period, adjust=False).mean()
            )
            df[f'feat_ema_gap_{period}'] = (df['adj_close'] - df[f'feat_ema_{period}']) / df[f'feat_ema_{period}']

        # MACD
        df['feat_macd_line'] = df.groupby(id_col)['adj_close'].transform(
            lambda x: x.ewm(span=12, adjust=False).mean() - x.ewm(span=26, adjust=False).mean()
        )
        df['feat_macd_signal'] = df.groupby(id_col)['feat_macd_line'].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )
        df['feat_macd_hist'] = df['feat_macd_line'] - df['feat_macd_signal']

        # Distance from 52-week high/low
        df['feat_52w_high'] = df.groupby(id_col)['high'].transform(lambda x: x.rolling(252).max())
        df['feat_52w_low'] = df.groupby(id_col)['low'].transform(lambda x: x.rolling(252).min())
        df['feat_pct_from_52h'] = (df['adj_close'] - df['feat_52w_high']) / df['feat_52w_high']
        df['feat_pct_from_52l'] = (df['adj_close'] - df['feat_52w_low']) / df['feat_52w_low']

        # Donchian channel breakout
        for period in [20, 50]:
            df[f'feat_donchian_high_{period}'] = df.groupby(id_col)['high'].transform(
                lambda x: x.rolling(period).max()
            )
            df[f'feat_donchian_low_{period}'] = df.groupby(id_col)['low'].transform(
                lambda x: x.rolling(period).min()
            )
            df[f'feat_donchian_break_{period}'] = (
                    (df['adj_close'] > df[f'feat_donchian_high_{period}'].shift(1)).astype(int) -
                    (df['adj_close'] < df[f'feat_donchian_low_{period}'].shift(1)).astype(int)
            )

        # Linear trend (OLS slope)
        for period in [20, 60]:
            df[f'feat_trend_slope_{period}'] = df.groupby(id_col)['adj_close'].transform(
                lambda x: x.rolling(period).apply(
                    lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) == period else np.nan
                )
            )

        return df

    def _create_volatility_features(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Create volatility and shape features"""
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df['returns'] = df.groupby(id_col)['adj_close'].pct_change()

        for window in self.config.volatility_windows:
            # Standard deviation
            df[f'feat_vol_{window}'] = df.groupby(id_col)['returns'].transform(
                lambda x: x.rolling(window).std()
            )

            # Downside deviation
            df[f'feat_downvol_{window}'] = df.groupby(id_col)['returns'].transform(
                lambda x: x.rolling(window).apply(
                    lambda y: np.sqrt(np.mean(np.minimum(y, 0) ** 2))
                )
            )

            # Parkinson volatility (high-low range)
            df[f'feat_parkinson_{window}'] = df.groupby(id_col).apply(
                lambda x: ((np.log(x['high'] / x['low']) ** 2).rolling(window).mean() / (4 * np.log(2))) ** 0.5
            ).reset_index(level=0, drop=True)

            # Average True Range (ATR)
            df['true_range'] = df.groupby(id_col).apply(
                lambda x: pd.DataFrame({
                    'tr': np.maximum(
                        x['high'] - x['low'],
                        np.maximum(
                            np.abs(x['high'] - x['close'].shift()),
                            np.abs(x['low'] - x['close'].shift())
                        )
                    )
                })
            ).reset_index(level=0, drop=True)
            df[f'feat_atr_{window}'] = df.groupby(id_col)['true_range'].transform(
                lambda x: x.rolling(window).mean()
            )

            # Volatility of volatility
            df[f'feat_volvol_{window}'] = df.groupby(id_col)[f'feat_vol_{window}'].transform(
                lambda x: x.rolling(window).std()
            )

            # Skewness and kurtosis
            df[f'feat_skew_{window}'] = df.groupby(id_col)['returns'].transform(
                lambda x: x.rolling(window).skew()
            )
            df[f'feat_kurt_{window}'] = df.groupby(id_col)['returns'].transform(
                lambda x: x.rolling(window).kurt()
            )

        return df

    def _create_drawdown_features(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Create drawdown-based features"""
        # Running maximum
        df['running_max'] = df.groupby(id_col)['adj_close'].transform(
            lambda x: x.expanding().max()
        )

        # Drawdown
        df['feat_drawdown'] = (df['adj_close'] - df['running_max']) / df['running_max']

        # Max drawdown over windows
        for window in [60, 120, 252]:
            df[f'feat_max_dd_{window}'] = df.groupby(id_col)['feat_drawdown'].transform(
                lambda x: x.rolling(window).min()
            )

        # Time since max
        df['feat_time_since_max'] = df.groupby(id_col).apply(
            lambda x: pd.Series(
                np.arange(len(x)) - x['adj_close'].expanding().idxmax().fillna(0),
                index=x.index
            )
        ).reset_index(level=0, drop=True)

        # Recovery time (periods in drawdown)
        df['in_drawdown'] = (df['feat_drawdown'] < 0).astype(int)
        df['feat_dd_duration'] = df.groupby([id_col, (df['in_drawdown'] != df['in_drawdown'].shift()).cumsum()])[
            'in_drawdown'
        ].cumsum()

        return df

    def _create_volume_features(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Create volume and liquidity features"""
        # Volume z-score (as proxy for turnover without shares outstanding)
        for window in self.config.volume_windows:
            df[f'feat_volume_zscore_{window}'] = df.groupby(id_col)['volume'].transform(
                lambda x: (x - x.rolling(window).mean()) / x.rolling(window).std()
            )

        # Dollar volume
        df['dollar_volume'] = df['close'] * df['volume']
        df['feat_dollar_volume'] = df.groupby(id_col)['dollar_volume'].transform(
            lambda x: x.rolling(20).mean()
        )

        # On-Balance Volume (OBV)
        df['price_change'] = df.groupby(id_col)['close'].diff()
        df['obv_raw'] = df.apply(
            lambda x: x['volume'] if x['price_change'] > 0 else
            (-x['volume'] if x['price_change'] < 0 else 0), axis=1
        )
        df['feat_obv'] = df.groupby(id_col)['obv_raw'].cumsum()

        # Chaikin Money Flow
        df['money_flow_mult'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df['money_flow_volume'] = df['money_flow_mult'] * df['volume']
        df['feat_cmf'] = df.groupby(id_col).apply(
            lambda x: x['money_flow_volume'].rolling(20).sum() / x['volume'].rolling(20).sum()
        ).reset_index(level=0, drop=True)

        # Price-Volume Trend (PVT)
        df['feat_pvt'] = df.groupby(id_col).apply(
            lambda x: ((x['close'].pct_change() * x['volume']).fillna(0).cumsum())
        ).reset_index(level=0, drop=True)

        # VWAP deviation proxy (using typical price as VWAP proxy)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['feat_vwap_dev'] = (df['close'] - df['typical_price']) / df['typical_price']

        return df

    def _create_liquidity_features(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Create illiquidity and spread proxy features"""
        # Amihud illiquidity measure
        df['feat_amihud'] = np.abs(df['returns']) / (df['dollar_volume'] + 1e-10)
        df['feat_amihud_ma'] = df.groupby(id_col)['feat_amihud'].transform(
            lambda x: x.rolling(20).mean()
        )

        # Roll effective spread estimator
        df['ret_lag'] = df.groupby(id_col)['returns'].shift(1)
        df['feat_roll_spread'] = df.groupby(id_col).apply(
            lambda x: 2 * np.sqrt(np.maximum(-x['returns'].rolling(20).cov(x['ret_lag']), 0))
        ).reset_index(level=0, drop=True)

        return df

    def _create_gap_features(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Create gap and intraday features"""
        # Overnight return (open to previous close)
        df['prev_close'] = df.groupby(id_col)['close'].shift(1)
        df['feat_overnight_ret'] = (df['open'] - df['prev_close']) / df['prev_close']

        # Intraday range
        df['feat_intraday_range'] = (df['high'] - df['low']) / df['close']

        # Close to open gap
        df['feat_close_open_gap'] = (df['close'] - df['open']) / df['open']

        return df

    def _create_spread_proxies(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Create bid-ask spread proxies from OHLCV"""
        # Corwin-Schultz spread estimator
        df['beta'] = df.groupby(id_col).apply(
            lambda x: (np.log(x['high'] / x['low']) ** 2 +
                       np.log(x['high'].shift(1) / x['low'].shift(1)) ** 2)
        ).reset_index(level=0, drop=True)

        df['gamma'] = df.groupby(id_col).apply(
            lambda x: np.log(
                np.maximum(x['high'], x['high'].shift(1)) /
                np.minimum(x['low'], x['low'].shift(1))
            ) ** 2
        ).reset_index(level=0, drop=True)

        df['alpha'] = (np.sqrt(2 * df['beta']) - np.sqrt(df['beta'])) / (3 - 2 * np.sqrt(2)) - np.sqrt(
            df['gamma'] / (3 - 2 * np.sqrt(2)))
        df['feat_cs_spread'] = 2 * (np.exp(df['alpha']) - 1) / (1 + np.exp(df['alpha']))
        df['feat_cs_spread'] = df['feat_cs_spread'].clip(lower=0)

        return df

    def _create_calendar_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Create calendar-based features from timestamps only"""
        # Day of week
        df['feat_dow'] = pd.to_datetime(df[date_col]).dt.dayofweek

        # Month of year
        df['feat_month'] = pd.to_datetime(df[date_col]).dt.month

        # Month-end flag
        df['feat_month_end'] = pd.to_datetime(df[date_col]).dt.is_month_end.astype(int)

        # Trading gap (days since last observation)
        df['date_numeric'] = pd.to_datetime(df[date_col]).astype(np.int64) / 1e9 / 86400  # Days since epoch
        df['feat_trading_gap'] = df.groupby('symbol')['date_numeric'].diff()

        # Holiday proxy (gap > 3 days for daily data)
        df['feat_holiday_gap'] = (df['feat_trading_gap'] > 3).astype(int)

        return df

    def _clean_features(self, df: pd.DataFrame, id_col: str, date_col: str) -> pd.DataFrame:
        """Clean and standardize features"""
        feature_cols = [col for col in df.columns if col.startswith('feat_')]

        # Winsorize features
        for col in feature_cols:
            lower = df[col].quantile(self.config.winsorize_quantile)
            upper = df[col].quantile(1 - self.config.winsorize_quantile)
            df[col] = df[col].clip(lower=lower, upper=upper)

        # Standardize features cross-sectionally
        if self.config.standardize:
            for col in feature_cols:
                df[col] = df.groupby(date_col)[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-10)
                )

        # Fill missing values
        if self.config.fillna_method == 'median':
            for col in feature_cols:
                df[col] = df.groupby(date_col)[col].transform(
                    lambda x: x.fillna(x.median())
                )
        elif self.config.fillna_method == 'zero':
            df[feature_cols] = df[feature_cols].fillna(0)

        # Final fillna with 0 for any remaining NaNs
        df[feature_cols] = df[feature_cols].fillna(0)

        return df


# ============================================================================
# src/mlmom/labels.py
"""Label generation for momentum strategies"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generate labels for cross-sectional and time-series momentum"""

    def __init__(self, config: LabelConfig):
        self.config = config

    def create_labels(self, df: pd.DataFrame, strategy_type: str = 'cs',
                      id_col: str = 'symbol', date_col: str = 'date') -> pd.DataFrame:
        """Create labels based on strategy type"""
        if strategy_type == 'cs':
            return self._create_cs_labels(df, id_col, date_col)
        else:
            return self._create_ts_labels(df, id_col, date_col)

    def _create_cs_labels(self, df: pd.DataFrame, id_col: str, date_col: str) -> pd.DataFrame:
        """Create cross-sectional momentum labels"""
        logger.info("Creating cross-sectional labels")

        # Calculate forward returns
        df['fwd_ret'] = df.groupby(id_col)['adj_close'].transform(
            lambda x: x.shift(-self.config.horizon) / x - 1
        )

        # Standardize returns cross-sectionally
        df['fwd_ret_std'] = df.groupby(date_col)['fwd_ret'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        )

        # Create quantile labels
        df['quantile'] = df.groupby(date_col)['fwd_ret'].transform(
            lambda x: pd.qcut(x, q=self.config.cs_quantiles, labels=False, duplicates='drop')
        )

        # Create binary classification label (top vs bottom quantile)
        if self.config.cs_top_bottom_only:
            df['label_class'] = np.where(
                df['quantile'] == self.config.cs_quantiles - 1, 1,  # Top quantile
                np.where(df['quantile'] == 0, 0, np.nan)  # Bottom quantile
            )
        else:
            df['label_class'] = df['quantile']

        # Regression label
        df['label_reg'] = df['fwd_ret_std']

        # Drop incomplete forward returns
        df = df.dropna(subset=['fwd_ret'])

        logger.info(f"Created CS labels with {df['label_reg'].notna().sum()} valid samples")
        return df

    def _create_ts_labels(self, df: pd.DataFrame, id_col: str, date_col: str) -> pd.DataFrame:
        """Create time-series momentum labels"""
        logger.info("Creating time-series labels")

        # Calculate forward returns
        df['fwd_ret'] = df.groupby(id_col)['adj_close'].transform(
            lambda x: x.shift(-self.config.horizon) / x - 1
        )

        # Calculate rolling volatility for scaling
        df['vol'] = df.groupby(id_col)['returns'].transform(
            lambda x: x.rolling(self.config.vol_lookback).std()
        )

        # Volatility-scaled returns
        df['fwd_ret_scaled'] = df['fwd_ret'] * (self.config.ts_vol_target / (df['vol'] + 1e-10))

        # Sign label (binary classification)
        df['label_sign'] = (df['fwd_ret'] > 0).astype(int)

        # Regression label (volatility-scaled)
        df['label_reg'] = df['fwd_ret_scaled']

        # Drop incomplete forward returns
        df = df.dropna(subset=['fwd_ret'])

        logger.info(f"Created TS labels with {df['label_reg'].notna().sum()} valid samples")
        return df


# ============================================================================
# src/mlmom/models.py
"""Model implementations for momentum strategies"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MomentumModel:
    """Base class for momentum models"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_importance = None

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """Fit the model"""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if self.feature_importance is not None:
            return self.feature_importance
        return pd.DataFrame()


class CSMomentumModel(MomentumModel):
    """Cross-sectional momentum model"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model based on configs"""
        if self.config.model_type == 'lightgbm':
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'num_threads': -1,
                **self.config.params
            }
            self.model = lgb.LGBMRegressor(**params)

        elif self.config.model_type == 'xgboost':
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_jobs': -1,
                **self.config.params
            }
            self.model = xgb.XGBRegressor(**params)

        elif self.config.model_type == 'gbm':
            params = {
                'loss': 'squared_error',
                'learning_rate': 0.05,
                'n_estimators': 100,
                'max_depth': 3,
                'subsample': 0.8,
                **self.config.params
            }
            self.model = GradientBoostingRegressor(**params)

        else:  # linear
            params = {
                'alpha': 1.0,
                'max_iter': 1000,
                **self.config.params
            }
            self.model = Lasso(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """Fit the model"""
        logger.info(f"Fitting {self.config.model_type} model")

        if sample_weight is not None and self.config.use_sample_weights:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)

        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)


class TSMomentumModel(MomentumModel):
    """Time-series momentum model"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model based on configs"""
        if self.config.model_type in ['lstm', 'tcn']:
            # For deep learning models, we'd implement LSTM/TCN here
            # For now, fall back to GBM
            logger.warning(f"{self.config.model_type} not implemented, using GBM")
            self.config.model_type = 'gbm'

        if self.config.model_type == 'lightgbm':
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                **self.config.params
            }
            self.model = lgb.LGBMClassifier(**params)

        elif self.config.model_type == 'xgboost':
            params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                **self.config.params
            }
            self.model = xgb.XGBClassifier(**params)

        elif self.config.model_type == 'gbm':
            params = {
                'loss': 'log_loss',
                'learning_rate': 0.05,
                'n_estimators': 100,
                'max_depth': 3,
                'subsample': 0.8,
                **self.config.params
            }
            self.model = GradientBoostingClassifier(**params)

        else:  # logistic
            params = {
                'penalty': 'l2',
                'C': 1.0,
                'max_iter': 1000,
                **self.config.params
            }
            self.model = LogisticRegression(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """Fit the model"""
        logger.info(f"Fitting {self.config.model_type} model for TS")

        if sample_weight is not None and self.config.use_sample_weights:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)

        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions (probabilities for classification)"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)


def calculate_sample_weights(df: pd.DataFrame, weight_type: str = 'amihud') -> pd.Series:
    """Calculate sample weights based on liquidity"""
    if weight_type == 'amihud' and 'feat_amihud' in df.columns:
        # Down-weight illiquid names
        weights = 1 / np.sqrt(1 + df['feat_amihud'].fillna(df['feat_amihud'].median()))
    else:
        weights = pd.Series(1, index=df.index)

    return weights / weights.mean()  # Normalize


# ============================================================================
# src/mlmom/cv.py
"""Cross-validation utilities with purging and embargo"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Generator
from sklearn.model_selection import TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)


class PurgedKFold:
    """Combinatorial Purged K-Fold Cross-Validation"""

    def __init__(self, n_splits: int = 5, embargo_periods: int = 10, purge_periods: int = 10):
        self.n_splits = n_splits
        self.embargo_periods = embargo_periods
        self.purge_periods = purge_periods

    def split(self, X: pd.DataFrame, y: pd.Series = None,
              groups: pd.Series = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test splits with purging and embargo"""

        if groups is None:
            raise ValueError("groups (dates) must be provided")

        unique_dates = sorted(groups.unique())
        n_dates = len(unique_dates)

        # Use TimeSeriesSplit as base
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        for train_dates_idx, test_dates_idx in tscv.split(unique_dates):
            train_dates = unique_dates[train_dates_idx[0]:train_dates_idx[-1] + 1]
            test_dates = unique_dates[test_dates_idx[0]:test_dates_idx[-1] + 1]

            # Apply embargo (remove dates after train that are too close to test)
            if self.embargo_periods > 0:
                embargo_start = test_dates[0]
                embargo_dates = []
                for i, date in enumerate(unique_dates):
                    if date < embargo_start:
                        idx = unique_dates.index(date)
                        if idx + self.embargo_periods >= unique_dates.index(embargo_start):
                            embargo_dates.append(date)

                train_dates = [d for d in train_dates if d not in embargo_dates]

            # Apply purging (remove dates before test that might leak)
            if self.purge_periods > 0:
                test_start_idx = unique_dates.index(test_dates[0])
                if test_start_idx > self.purge_periods:
                    purge_dates = unique_dates[test_start_idx - self.purge_periods:test_start_idx]
                    train_dates = [d for d in train_dates if d not in purge_dates]

            # Get indices
            train_idx = groups[groups.isin(train_dates)].index.values
            test_idx = groups[groups.isin(test_dates)].index.values

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


class RollingWindowCV:
    """Rolling window cross-validation with expanding or fixed window"""

    def __init__(self, initial_train_size: int = 252, test_size: int = 60,
                 refit_frequency: int = 60, expanding: bool = True,
                 embargo_periods: int = 10):
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.refit_frequency = refit_frequency
        self.expanding = expanding
        self.embargo_periods = embargo_periods

    def split(self, X: pd.DataFrame, y: pd.Series = None,
              groups: pd.Series = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test splits"""

        if groups is None:
            raise ValueError("groups (dates) must be provided")

        unique_dates = sorted(groups.unique())
        n_dates = len(unique_dates)

        # Start from initial_train_size
        test_start = self.initial_train_size

        while test_start + self.test_size <= n_dates:
            # Define train period
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, test_start - self.initial_train_size - self.embargo_periods)

            train_end = test_start - self.embargo_periods
            test_end = min(test_start + self.test_size, n_dates)

            # Get dates
            train_dates = unique_dates[train_start:train_end]
            test_dates = unique_dates[test_start:test_end]

            # Get indices
            train_idx = groups[groups.isin(train_dates)].index.values
            test_idx = groups[groups.isin(test_dates)].index.values

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

            # Move to next test period
            test_start += self.refit_frequency


# ============================================================================
# src/mlmom/backtest.py
"""Backtesting engine for momentum strategies"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

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


# ============================================================================
# src/mlmom/costs.py
"""Transaction cost modeling"""
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CostModel:
    """Transaction cost model"""

    def __init__(self, config: CostConfig):
        self.config = config

    def calculate_costs(self, trades: pd.DataFrame) -> pd.Series:
        """Calculate transaction costs for trades"""
        if self.config.dynamic_costs:
            return self._dynamic_costs(trades)
        else:
            return self._fixed_costs(trades)

    def _fixed_costs(self, trades: pd.DataFrame) -> pd.Series:
        """Fixed transaction costs"""
        # Turnover * fixed cost in bps
        costs = trades['turnover'] * self.config.fixed_costs_bps / 10000

        # Add turnover penalty if specified
        if self.config.turnover_penalty > 0:
            costs += trades['turnover'] * self.config.turnover_penalty / 10000

        return costs

    def _dynamic_costs(self, trades: pd.DataFrame) -> pd.Series:
        """Dynamic transaction costs based on market impact models"""
        if self.config.cost_model == 'roll' and 'feat_roll_spread' in trades.columns:
            # Use Roll spread estimate
            costs = trades['turnover'] * trades['feat_roll_spread'] * 100  # Convert to bps
        elif self.config.cost_model == 'amihud' and 'feat_amihud' in trades.columns:
            # Use Amihud illiquidity measure
            costs = trades['turnover'] * np.sqrt(trades['feat_amihud']) * 100
        else:
            # Fall back to fixed costs
            logger.warning(f"Dynamic cost model {self.config.cost_model} not available, using fixed costs")
            costs = self._fixed_costs(trades)

        # Add turnover penalty
        if self.config.turnover_penalty > 0:
            costs += trades['turnover'] * self.config.turnover_penalty / 10000

        return costs


# ============================================================================
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


# ============================================================================
# configs/cs.yaml
data:
prices_path: "C:/Desktop/BA/PY/my_project/scripts/data/cleaned/ML_data_cleaned.parquet"
id_column: "symbol"
date_column: "date"
frequency: "daily"
currency: "USD"

features:
return_periods: [1, 5, 20, 60, 120, 252]
skip_periods: 1
ma_periods: [10, 20, 50, 200]
volatility_windows: [20, 60]
volume_windows: [20]
winsorize_quantile: 0.01
standardize: true
fillna_method: "median"

labels:
horizon: 20
cs_quantiles: 5
cs_top_bottom_only: true

model:
model_type: "lightgbm"
params:
num_leaves: 31
learning_rate: 0.03
feature_fraction: 0.8
bagging_fraction: 0.7
bagging_freq: 5
min_child_samples: 20
n_estimators: 200
use_sample_weights: true
weight_type: "amihud"

cv:
n_splits: 5
embargo_periods: 10
purge_periods: 10
refit_frequency: 60
validation_size: 0.2
test_size: 0.2

portfolio:
strategy_type: "cs"
long_only: false
max_weight: 0.05
rebalance_frequency: 20

costs:
fixed_costs_bps: 10.0
dynamic_costs: false
turnover_penalty: 0.0

output_dir: "outputs/cs"
seed: 42
n_jobs: -1
verbose: 1

# ============================================================================
# configs/ts.yaml
data:
prices_path: "C:/Desktop/BA/PY/my_project/scripts/data/cleaned/ML_data_cleaned.parquet"
id_column: "symbol"
date_column: "date"
frequency: "daily"
currency: "USD"

features:
return_periods: [1, 5, 20, 60, 120, 252]
skip_periods: 1
ma_periods: [10, 20, 50, 200]
volatility_windows: [20, 60]
volume_windows: [20]
winsorize_quantile: 0.01
standardize: true
fillna_method: "median"

labels:
horizon: 20
ts_vol_target: 0.15
vol_lookback: 60

model:
model_type: "lightgbm"
params:
objective: "binary"
metric: "auc"
num_leaves: 31
learning_rate: 0.03
feature_fraction: 0.8
bagging_fraction: 0.7
bagging_freq: 5
min_child_samples: 20
n_estimators: 200
use_sample_weights: true
weight_type: "amihud"

cv:
n_splits: 5
embargo_periods: 10
purge_periods: 10
refit_frequency: 60
validation_size: 0.2
test_size: 0.2

portfolio:
strategy_type: "ts"
long_only: false
target_vol: 0.15
rebalance_frequency: 20

costs:
fixed_costs_bps: 10.0
dynamic_costs: false
turnover_penalty: 0.0

output_dir: "outputs/ts"
seed: 42
n_jobs: -1
verbose: 1