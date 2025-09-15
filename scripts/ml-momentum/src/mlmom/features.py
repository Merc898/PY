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
