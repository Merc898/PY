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
