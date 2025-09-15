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
