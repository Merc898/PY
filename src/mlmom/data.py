# src/mlmom/data.py
"""Data loading and preprocessing utilities"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging

from src.mlmom.config import DataConfig

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

