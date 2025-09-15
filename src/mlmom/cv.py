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
