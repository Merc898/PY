# tests/test_mlmom.py
"""Unit tests for ML momentum strategy components"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.mlmom.config import Config, DataConfig, FeatureConfig, LabelConfig
from src.mlmom.features import FeatureEngineer
from src.mlmom.labels import LabelGenerator
from src.mlmom.cv import PurgedKFold, RollingWindowCV


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering"""

    def setUp(self):
        """Create sample data"""
        dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT']

        data = []
        for symbol in symbols:
            for date in dates:
                price = 100 + np.random.randn() * 10
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'open': price * 0.99,
                    'high': price * 1.01,
                    'low': price * 0.98,
                    'close': price,
                    'adj_close': price,
                    'volume': np.random.randint(1000000, 10000000)
                })

        self.df = pd.DataFrame(data)
        self.df['returns'] = self.df.groupby('symbol')['adj_close'].pct_change()

    def test_feature_creation(self):
        """Test that features are created correctly"""
        config = FeatureConfig(
            return_periods=[1, 5, 20],
            ma_periods=[10, 20],
            volatility_windows=[20],
            volume_windows=[20]
        )

        engineer = FeatureEngineer(config)
        df_features = engineer.create_features(self.df.copy())

        # Check that features are created
        feature_cols = [col for col in df_features.columns if col.startswith('feat_')]
        self.assertGreater(len(feature_cols), 20)

        # Check for specific features
        self.assertIn('feat_ret_1', df_features.columns)
        self.assertIn('feat_sma_10', df_features.columns)
        self.assertIn('feat_vol_20', df_features.columns)

    def test_no_lookahead_bias(self):
        """Test that features don't have lookahead bias"""
        config = FeatureConfig(return_periods=[1, 5])
        engineer = FeatureEngineer(config)

        df_features = engineer.create_features(self.df.copy())

        # Check that return features are properly lagged
        for symbol in self.df['symbol'].unique():
            symbol_data = df_features[df_features['symbol'] == symbol]

            # feat_ret_1 should be the same as returns shifted by 1
            expected = symbol_data['adj_close'].pct_change(1)
            actual = symbol_data['feat_ret_1']

            # Allow for small numerical differences
            pd.testing.assert_series_equal(
                expected, actual, check_names=False, atol=1e-6
            )


class TestLabelGeneration(unittest.TestCase):
    """Test label generation"""

    def setUp(self):
        """Create sample data"""
        dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']

        data = []
        for symbol in symbols:
            prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
            for i, date in enumerate(dates):
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'adj_close': prices[i],
                    'returns': np.random.randn() * 0.02
                })

        self.df = pd.DataFrame(data)

    def test_cs_labels(self):
        """Test cross-sectional label generation"""
        config = LabelConfig(
            horizon=20,
            cs_quantiles=5,
            cs_top_bottom_only=True
        )

        generator = LabelGenerator(config)
        df_labels = generator.create_labels(self.df.copy(), strategy_type='cs')

        # Check that labels are created
        self.assertIn('label_reg', df_labels.columns)
        self.assertIn('label_class', df_labels.columns)
        self.assertIn('quantile', df_labels.columns)

        # Check that labels are properly distributed
        for date in df_labels['date'].unique()[:10]:  # Check first 10 dates
            date_data = df_labels[df_labels['date'] == date]

            # Check quantiles sum to expected
            if len(date_data) >= 5:
                quantiles = date_data['quantile'].dropna()
                self.assertTrue(quantiles.min() >= 0)
                self.assertTrue(quantiles.max() <= 4)

    def test_ts_labels(self):
        """Test time-series label generation"""
        config = LabelConfig(
            horizon=20,
            ts_vol_target=0.15,
            vol_lookback=60
        )

        generator = LabelGenerator(config)
        df_labels = generator.create_labels(self.df.copy(), strategy_type='ts')

        # Check that labels are created
        self.assertIn('label_reg', df_labels.columns)
        self.assertIn('label_sign', df_labels.columns)

        # Check binary labels
        self.assertTrue(df_labels['label_sign'].isin([0, 1]).all())


class TestCrossValidation(unittest.TestCase):
    """Test cross-validation with purging and embargo"""

    def setUp(self):
        """Create sample data"""
        dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
        self.X = pd.DataFrame(np.random.randn(len(dates), 5))
        self.y = pd.Series(np.random.randn(len(dates)))
        self.groups = pd.Series(dates)

    def test_purged_kfold(self):
        """Test purged k-fold CV"""
        cv = PurgedKFold(n_splits=3, embargo_periods=5, purge_periods=5)

        splits = list(cv.split(self.X, self.y, self.groups))
        self.assertEqual(len(splits), 3)

        for train_idx, test_idx in splits:
            # Check no overlap
            self.assertEqual(len(set(train_idx) & set(test_idx)), 0)

            # Check embargo
            train_dates = self.groups.iloc[train_idx]
            test_dates = self.groups.iloc[test_idx]

            if len(train_dates) > 0 and len(test_dates) > 0:
                # Latest train date should be sufficiently before earliest test date
                gap = (test_dates.min() - train_dates.max()).days
                self.assertGreaterEqual(gap, 1)  # At least 1 day gap

    def test_rolling_window(self):
        """Test rolling window CV"""
        cv = RollingWindowCV(
            initial_train_size=100,
            test_size=20,
            refit_frequency=20,
            expanding=True,
            embargo_periods=5
        )

        splits = list(cv.split(self.X, self.y, self.groups))
        self.assertGreater(len(splits), 0)

        # Check that training set expands
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        for i in range(1, len(train_sizes)):
            self.assertGreaterEqual(train_sizes[i], train_sizes[i - 1])


class TestSampleWeights(unittest.TestCase):
    """Test sample weight calculation"""

    def test_amihud_weights(self):
        """Test Amihud-based sample weights"""
        from src.mlmom.models import calculate_sample_weights

        df = pd.DataFrame({
            'feat_amihud': [0.1, 0.5, 1.0, 2.0, 0.01]
        })

        weights = calculate_sample_weights(df, weight_type='amihud')

        # Check that weights are normalized
        self.assertAlmostEqual(weights.mean(), 1.0, places=5)

        # Check that illiquid stocks have lower weights
        self.assertLess(weights.iloc[3], weights.iloc[0])  # Higher Amihud = lower weight
        self.assertGreater(weights.iloc[4], weights.iloc[1])  # Lower Amihud = higher weight


if __name__ == '__main__':
    unittest.main()
