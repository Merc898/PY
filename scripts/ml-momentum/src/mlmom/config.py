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
