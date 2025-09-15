
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
