#!/usr/bin/env python3
"""
run_diagnostics.py - Comprehensive diagnostics for data quality and strategy validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict
import warnings
from scipy import stats
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDiagnostics:
    """Comprehensive data quality diagnostics"""

    def __init__(self, output_dir: str = "diagnostics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.issues = []

    def run_all_diagnostics(self, data_path: str) -> Dict:
        """Run complete diagnostic suite"""
        logger.info("=" * 60)
        logger.info("RUNNING COMPREHENSIVE DIAGNOSTICS")
        logger.info("=" * 60)

        # Load data
        df = self.load_data(data_path)

        if df.empty:
            logger.error("No data to analyze!")
            return {}

        # Run diagnostics
        results = {
            'data_overview': self.data_overview(df),
            'data_quality': self.check_data_quality(df),
            'corporate_actions': self.detect_corporate_actions(df),
            'currency_check': self.check_currency_consistency(df),
            'coverage_analysis': self.analyze_coverage(df),
            'return_analysis': self.analyze_returns(df),
            'outlier_analysis': self.detect_outliers(df),
            'survivorship_bias': self.check_survivorship_bias(df)
        }

        # Generate report
        self.generate_report(results)

        # Create visualizations
        self.create_diagnostic_plots(df)

        return results

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data for analysis"""
        try:
            if data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path)

            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"Loaded {len(df):,} observations")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return pd.DataFrame()

    def data_overview(self, df: pd.DataFrame) -> Dict:
        """Get basic data overview"""
        logger.info("\n📊 DATA OVERVIEW")
        logger.info("-" * 40)

        overview = {
            'total_observations': len(df),
            'unique_symbols': df['symbol'].nunique(),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'total_days': (df['date'].max() - df['date'].min()).days,
            'columns': list(df.columns)
        }

        if 'currency' in df.columns:
            currency_counts = df.groupby('currency')['symbol'].nunique().to_dict()
            overview['symbols_by_currency'] = currency_counts
            if len(currency_counts) > 1:
                self.issues.append("⚠️ Multiple currencies detected in dataset")

        print(f"Total observations: {overview['total_observations']:,}")
        print(f"Unique symbols: {overview['unique_symbols']}")
        print(f"Date range: {overview['date_range']}")

        if 'symbols_by_currency' in overview:
            print("\nSymbols by currency:")
            for curr, count in overview['symbols_by_currency'].items():
                print(f"  {curr}: {count}")

        return overview

    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        """Check for data quality issues"""
        logger.info("\n🔍 DATA QUALITY CHECK")
        logger.info("-" * 40)

        quality_issues = {}

        missing = df.isnull().sum()
        if missing.any():
            quality_issues['missing_values'] = missing[missing > 0].to_dict()
            self.issues.append(f"❌ Missing values in {len(missing[missing > 0])} columns")

        duplicates = df.duplicated(subset=['symbol', 'date']).sum()
        quality_issues['duplicate_entries'] = duplicates
        if duplicates > 0:
            self.issues.append(f"❌ {duplicates} duplicate (symbol, date) entries")

        price_cols = ['close', 'open', 'high', 'low']
        for col in price_cols:
            if col in df.columns:
                negative = (df[col] <= 0).sum()
                if negative > 0:
                    quality_issues[f'negative_{col}'] = negative
                    self.issues.append(f"❌ {negative} non-positive {col} prices")

        if all(col in df.columns for col in ['high', 'low', 'close']):
            inconsistent = ((df['high'] < df['low']) |
                            (df['close'] > df['high']) |
                            (df['close'] < df['low'])).sum()
            if inconsistent > 0:
                quality_issues['price_inconsistencies'] = inconsistent
                self.issues.append(f"❌ {inconsistent} price inconsistencies")

        return quality_issues

    def detect_corporate_actions(self, df: pd.DataFrame) -> Dict:
        logger.info("\n🔄 CORPORATE ACTIONS DETECTION")
        logger.info("-" * 40)

        if 'close' not in df.columns:
            return {}

        df_sorted = df.sort_values(['symbol', 'date'])
        df_sorted['daily_return'] = df_sorted.groupby('symbol')['close'].pct_change()

        potential = df_sorted[(df_sorted['daily_return'] > 0.45) |
                              (df_sorted['daily_return'] < -0.30)]
        corporate_actions = {
            'potential_splits': len(potential),
            'affected_symbols': potential['symbol'].nunique()
        }
        if len(potential) > 0:
            self.issues.append(f"⚠️ {len(potential)} potential splits/corporate actions detected")
        return corporate_actions

    def check_currency_consistency(self, df: pd.DataFrame) -> Dict:
        logger.info("\n💱 CURRENCY CONSISTENCY CHECK")
        logger.info("-" * 40)
        if 'currency' not in df.columns:
            return {}

        currencies = df['currency'].unique()
        result = {'unique_currencies': list(currencies)}

        if len(currencies) > 1:
            if 'close_usd' not in df.columns:
                self.issues.append("❌ Multiple currencies without FX adjustment")
        return result

    def analyze_coverage(self, df: pd.DataFrame) -> Dict:
        logger.info("\n📈 COVERAGE ANALYSIS")
        logger.info("-" * 40)
        coverage = {}
        symbol_cov = df.groupby('symbol')['date'].agg(['min', 'max', 'count'])
        coverage['avg_obs_per_symbol'] = symbol_cov['count'].mean()
        return coverage

    def analyze_returns(self, df: pd.DataFrame) -> Dict:
        logger.info("\n📊 RETURN ANALYSIS")
        logger.info("-" * 40)
        if 'close' not in df.columns:
            return {}
        df_sorted = df.sort_values(['symbol', 'date'])
        df_sorted['daily_return'] = df_sorted.groupby('symbol')['close'].pct_change()
        r = df_sorted['daily_return'].dropna()
        stats_dict = {
            'mean': r.mean(),
            'std': r.std(),
            'skew': stats.skew(r),
            'kurtosis': stats.kurtosis(r)
        }
        return stats_dict

    def detect_outliers(self, df: pd.DataFrame) -> Dict:
        logger.info("\n🎯 OUTLIER DETECTION")
        logger.info("-" * 40)
        outliers = {}
        if 'volume' in df.columns:
            df['volume_z'] = df.groupby('symbol')['volume'].transform(
                lambda x: np.abs((x - x.mean()) / x.std()))
            outliers['volume_outliers'] = int((df['volume_z'] > 5).sum())
        return outliers

    def check_survivorship_bias(self, df: pd.DataFrame) -> Dict:
        logger.info("\n🔎 SURVIVORSHIP BIAS CHECK")
        logger.info("-" * 40)
        recent_date = df['date'].max() - timedelta(days=90)
        symbols_with_recent = df[df['date'] > recent_date]['symbol'].unique()
        missing = set(df['symbol'].unique()) - set(symbols_with_recent)
        return {'symbols_missing_recent': len(missing)}

    def create_diagnostic_plots(self, df: pd.DataFrame):
        logger.info("\n📊 Creating diagnostic plots...")
        fig, ax = plt.subplots(figsize=(10, 5))
        daily_symbols = df.groupby('date')['symbol'].nunique()
        ax.plot(daily_symbols.index, daily_symbols.values)
        ax.set_title('Number of Symbols Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Symbols')
        plt.tight_layout()
        plot_path = self.output_dir / "diagnostic_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Plots saved to {plot_path}")

    def generate_report(self, results: Dict):
        logger.info("\n📝 Generating diagnostic report...")
        report_path = self.output_dir / "diagnostic_report.txt"

        with open(report_path, 'w', encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("DATA DIAGNOSTICS REPORT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")
            f.write("SUMMARY OF ISSUES FOUND:\n")
            f.write("-" * 40 + "\n")
            if self.issues:
                for issue in self.issues:
                    f.write(f"• {issue}\n")
            else:
                f.write("✅ No critical issues found\n")

        json_path = self.output_dir / "diagnostic_results.json"
        with open(json_path, 'w', encoding="utf-8") as jf:
            json.dump(results, jf, indent=2, default=str)
        logger.info(f"Report saved to {report_path}")
        logger.info(f"JSON results saved to {json_path}")


def main():
    diagnostics = DataDiagnostics()
    data_paths = [
        "data/processed/prices_clean.parquet",
        "data/raw/prices_combined.csv",
        "data/raw/price_data.parquet"
    ]
    data_path = next((p for p in data_paths if Path(p).exists()), None)
    if not data_path:
        logger.error("No data file found! Run fetch_all_prices.py first")
        return 1
    logger.info(f"Using data file: {data_path}")
    diagnostics.run_all_diagnostics(data_path)
    return 0


if __name__ == "__main__":
    exit(main())
