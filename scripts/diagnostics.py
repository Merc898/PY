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
from typing import Dict, List, Tuple
import warnings
from scipy import stats
from datetime import datetime, timedelta

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

        # Currency breakdown if available
        if 'currency' in df.columns:
            currency_counts = df.groupby('currency')['symbol'].nunique().to_dict()
            overview['symbols_by_currency'] = currency_counts

            # Check for currency mixing
            if len(currency_counts) > 1:
                self.issues.append("⚠️ Multiple currencies detected in dataset")

        # Print overview
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

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            quality_issues['missing_values'] = missing[missing > 0].to_dict()
            self.issues.append(f"❌ Missing values found in {len(missing[missing > 0])} columns")

        # Check for duplicate entries
        duplicates = df.duplicated(subset=['symbol', 'date']).sum()
        quality_issues['duplicate_entries'] = duplicates
        if duplicates > 0:
            self.issues.append(f"❌ {duplicates} duplicate (symbol, date) entries")

        # Check for negative prices
        price_cols = ['close', 'open', 'high', 'low']
        for col in price_cols:
            if col in df.columns:
                negative = (df[col] <= 0).sum()
                if negative > 0:
                    quality_issues[f'negative_{col}'] = negative
                    self.issues.append(f"❌ {negative} negative/zero {col} prices")

        # Check for price consistency (high >= low, close within high/low)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            inconsistent = ((df['high'] < df['low']) |
                            (df['close'] > df['high']) |
                            (df['close'] < df['low'])).sum()
            quality_issues['price_inconsistencies'] = inconsistent
            if inconsistent > 0:
                self.issues.append(f"❌ {inconsistent} price inconsistencies (high/low/close)")

        # Check for stale prices
        if 'close' in df.columns:
            df_sorted = df.sort_values(['symbol', 'date'])
            df_sorted['price_unchanged'] = df_sorted.groupby('symbol')['close'].transform(
                lambda x: (x == x.shift(1))
            )

            # Count consecutive unchanged prices
            stale_symbols = []
            for symbol in df_sorted['symbol'].unique():
                symbol_data = df_sorted[df_sorted['symbol'] == symbol]
                max_consecutive = 0
                current_consecutive = 0

                for unchanged in symbol_data['price_unchanged']:
                    if unchanged:
                        current_consecutive += 1
                        max_consecutive = max(max_consecutive, current_consecutive)
                    else:
                        current_consecutive = 0

                if max_consecutive >= 10:  # 10+ consecutive days unchanged
                    stale_symbols.append((symbol, max_consecutive))

            if stale_symbols:
                quality_issues['stale_prices'] = stale_symbols[:10]  # Top 10
                self.issues.append(f"⚠️ {len(stale_symbols)} symbols with stale prices (10+ days unchanged)")

        # Print summary
        if not quality_issues:
            print("✅ No major data quality issues detected")
        else:
            print(f"❌ Found {len(quality_issues)} types of quality issues")
            for issue, details in quality_issues.items():
                if isinstance(details, int):
                    print(f"  - {issue}: {details}")
                elif isinstance(details, list) and len(details) > 0:
                    print(f"  - {issue}: {len(details)} cases")

        return quality_issues

    def detect_corporate_actions(self, df: pd.DataFrame) -> Dict:
        """Detect potential corporate actions (splits, dividends)"""
        logger.info("\n🔄 CORPORATE ACTIONS DETECTION")
        logger.info("-" * 40)

        if 'close' not in df.columns:
            return {}

        # Calculate daily returns
        df_sorted = df.sort_values(['symbol', 'date'])
        df_sorted['daily_return'] = df_sorted.groupby('symbol')['close'].pct_change()

        # Detect potential splits (returns > 45% or < -30%)
        potential_splits = df_sorted[
            (df_sorted['daily_return'] > 0.45) |
            (df_sorted['daily_return'] < -0.30)
            ]

        corporate_actions = {
            'potential_splits': len(potential_splits),
            'affected_symbols': potential_splits['symbol'].nunique()
        }

        if len(potential_splits) > 0:
            self.issues.append(f"⚠️ {len(potential_splits)} potential splits/corporate actions detected")

            # Sample of potential splits
            sample = potential_splits.nlargest(10, 'daily_return')[['symbol', 'date', 'daily_return']]
            corporate_actions['sample_splits'] = sample.to_dict('records')

            print(f"Detected {len(potential_splits)} potential corporate actions")
            print(f"Affected symbols: {potential_splits['symbol'].nunique()}")
            print("\nSample of largest moves:")
            for row in sample.head(5).itertuples():
                print(f"  {row.symbol} on {row.date}: {row.daily_return * 100:.1f}%")
        else:
            print("✅ No obvious corporate actions detected")

        return corporate_actions

    def check_currency_consistency(self, df: pd.DataFrame) -> Dict:
        """Check for currency mixing issues"""
        logger.info("\n💱 CURRENCY CONSISTENCY CHECK")
        logger.info("-" * 40)

        if 'currency' not in df.columns:
            print("No currency information available")
            return {}

        currency_analysis = {}

        # Count unique currencies
        currencies = df['currency'].unique()
        currency_analysis['unique_currencies'] = list(currencies)

        if len(currencies) > 1:
            print(f"⚠️ WARNING: {len(currencies)} different currencies in dataset")

            # Check if FX adjustment was done
            if 'close_usd' in df.columns:
                print("✅ USD-adjusted prices available")
                currency_analysis['fx_adjusted'] = True
            else:
                print("❌ No USD-adjusted prices found")
                currency_analysis['fx_adjusted'] = False
                self.issues.append("❌ CRITICAL: Multiple currencies without FX adjustment")

            # Show currency breakdown
            currency_counts = df.groupby('currency')['symbol'].nunique()
            print("\nSymbols by currency:")
            for curr, count in currency_counts.items():
                print(f"  {curr}: {count} symbols")

            currency_analysis['symbol_counts'] = currency_counts.to_dict()
        else:
            print(f"✅ Single currency dataset: {currencies[0]}")
            currency_analysis['single_currency'] = True

        return currency_analysis

    def analyze_coverage(self, df: pd.DataFrame) -> Dict:
        """Analyze data coverage and completeness"""
        logger.info("\n📈 COVERAGE ANALYSIS")
        logger.info("-" * 40)

        coverage = {}

        # Coverage by symbol
        symbol_coverage = df.groupby('symbol').agg({
            'date': ['min', 'max', 'count']
        })
        symbol_coverage.columns = ['first_date', 'last_date', 'n_observations']

        # Calculate trading days coverage
        symbol_coverage['total_days'] = (
                symbol_coverage['last_date'] - symbol_coverage['first_date']
        ).dt.days
        symbol_coverage['coverage_ratio'] = (
                symbol_coverage['n_observations'] / (symbol_coverage['total_days'] * 5 / 7)
        )

        coverage['avg_observations_per_symbol'] = symbol_coverage['n_observations'].mean()
        coverage['median_observations_per_symbol'] = symbol_coverage['n_observations'].median()

        # Find symbols with poor coverage
        poor_coverage = symbol_coverage[symbol_coverage['coverage_ratio'] < 0.8]
        coverage['poor_coverage_symbols'] = len(poor_coverage)

        if len(poor_coverage) > 0:
            self.issues.append(f"⚠️ {len(poor_coverage)} symbols with <80% coverage")

        # Time series coverage
        daily_coverage = df.groupby('date')['symbol'].nunique()
        coverage['avg_symbols_per_day'] = daily_coverage.mean()
        coverage['min_symbols_per_day'] = daily_coverage.min()
        coverage['max_symbols_per_day'] = daily_coverage.max()

        # Print summary
        print(f"Average observations per symbol: {coverage['avg_observations_per_symbol']:.0f}")
        print(f"Average symbols per day: {coverage['avg_symbols_per_day']:.0f}")
        print(f"Symbols with poor coverage (<80%): {coverage['poor_coverage_symbols']}")

        # Check for survivorship bias indicators
        recent_date = df['date'].max() - timedelta(days=30)
        recent_symbols = df[df['date'] > recent_date]['symbol'].unique()
        all_symbols = df['symbol'].unique()

        missing_recent = len(set(all_symbols) - set(recent_symbols))
        if missing_recent > 0:
            coverage['symbols_missing_recent_data'] = missing_recent
            print(f"\n⚠️ {missing_recent} symbols have no data in last 30 days (possible delistings)")

        return coverage

    def analyze_returns(self, df: pd.DataFrame) -> Dict:
        """Analyze return distributions"""
        logger.info("\n📊 RETURN ANALYSIS")
        logger.info("-" * 40)

        if 'close' not in df.columns:
            return {}

        # Calculate returns
        df_sorted = df.sort_values(['symbol', 'date'])
        df_sorted['daily_return'] = df_sorted.groupby('symbol')['close'].pct_change()

        # Remove outliers for statistics
        returns = df_sorted['daily_return'].dropna()
        returns_clean = returns[abs(returns) < 0.5]  # Remove >50% moves

        return_stats = {
            'mean': returns_clean.mean(),
            'std': returns_clean.std(),
            'skewness': stats.skew(returns_clean),
            'kurtosis': stats.kurtosis(returns_clean),
            'percentile_1': np.percentile(returns_clean, 1),
            'percentile_99': np.percentile(returns_clean, 99),
            'observations': len(returns_clean)
        }

        # Print statistics
        print(f"Daily Return Statistics (outliers removed):")
        print(f"  Mean:     {return_stats['mean'] * 100:.4f}%")
        print(f"  Std Dev:  {return_stats['std'] * 100:.2f}%")
        print(f"  Skewness: {return_stats['skewness']:.3f}")
        print(f"  Kurtosis: {return_stats['kurtosis']:.3f}")
        print(f"  1st percentile:  {return_stats['percentile_1'] * 100:.2f}%")
        print(f"  99th percentile: {return_stats['percentile_99'] * 100:.2f}%")

        # Check for return anomalies
        extreme_positive = (returns > 1.0).sum()
        extreme_negative = (returns < -0.9).sum()

        if extreme_positive > 0 or extreme_negative > 0:
            print(f"\n⚠️ Extreme returns detected:")
            print(f"  Returns > 100%: {extreme_positive}")
            print(f"  Returns < -90%: {extreme_negative}")
            self.issues.append(f"⚠️ {extreme_positive + extreme_negative} extreme returns detected")

        return return_stats

    def detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect various types of outliers"""
        logger.info("\n🎯 OUTLIER DETECTION")
        logger.info("-" * 40)

        outliers = {}

        if 'close' in df.columns:
            # Price outliers (using IQR method)
            for symbol in df['symbol'].unique()[:100]:  # Check first 100 symbols
                symbol_data = df[df['symbol'] == symbol]['close']
                if len(symbol_data) < 100:
                    continue

                Q1 = symbol_data.quantile(0.25)
                Q3 = symbol_data.quantile(0.75)
                IQR = Q3 - Q1

                outlier_condition = ((symbol_data < Q1 - 3 * IQR) |
                                     (symbol_data > Q3 + 3 * IQR))

                if outlier_condition.any():
                    if 'price_outliers' not in outliers:
                        outliers['price_outliers'] = []
                    outliers['price_outliers'].append(symbol)

        if 'volume' in df.columns:
            # Volume outliers
            df['volume_zscore'] = df.groupby('symbol')['volume'].transform(
                lambda x: np.abs((x - x.mean()) / x.std())
            )
            volume_outliers = df[df['volume_zscore'] > 5]
            outliers['volume_outliers'] = len(volume_outliers)

        # Print summary
        if outliers:
            print(f"Outliers detected:")
            if 'price_outliers' in outliers:
                print(f"  Symbols with price outliers: {len(outliers['price_outliers'])}")
            if 'volume_outliers' in outliers:
                print(f"  Volume outliers: {outliers['volume_outliers']}")
        else:
            print("✅ No significant outliers detected")

        return outliers

    def check_survivorship_bias(self, df: pd.DataFrame) -> Dict:
        """Check for potential survivorship bias"""
        logger.info("\n🔎 SURVIVORSHIP BIAS CHECK")
        logger.info("-" * 40)

        bias_indicators = {}

        # Check if all symbols have recent data
        recent_date = df['date'].max() - timedelta(days=90)

        symbols_with_recent = df[df['date'] > recent_date]['symbol'].unique()
        all_symbols = df['symbol'].unique()

        missing_recent = set(all_symbols) - set(symbols_with_recent)
        bias_indicators['symbols_missing_recent'] = len(missing_recent)
        bias_indicators['pct_missing_recent'] = len(missing_recent) / len(all_symbols) * 100

        if len(missing_recent) > 0:
            print(f"⚠️ WARNING: Potential survivorship bias detected")
            print(f"  {len(missing_recent)} symbols ({bias_indicators['pct_missing_recent']:.1f}%) have no recent data")
            print(f"  This suggests these may be delisted companies")
            self.issues.append(f"⚠️ Potential survivorship bias: {len(missing_recent)} missing recent data")

            # Sample of missing symbols
            sample_missing = list(missing_recent)[:10]
            print(f"\n  Sample of potentially delisted: {sample_missing}")
        else:
            print("✅ All symbols have recent data")

        # Check for suspiciously good performance in old symbols
        old_date = df['date'].min() + timedelta(days=365)
        old_symbols = df[df['date'] < old_date]['symbol'].unique()

        # Check if old symbols are overrepresented in recent data
        old_symbols_surviving = set(old_symbols) & set(symbols_with_recent)
        survival_rate = len(old_symbols_surviving) / len(old_symbols) * 100

        bias_indicators['old_symbol_survival_rate'] = survival_rate

        if survival_rate > 90:
            print(f"\n⚠️ Suspiciously high survival rate: {survival_rate:.1f}%")
            print("  This may indicate survivorship bias in historical data")

        return bias_indicators

    def create_diagnostic_plots(self, df: pd.DataFrame):
        """Create diagnostic visualizations"""
        logger.info("\n📊 Creating diagnostic plots...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Number of symbols over time
        ax = axes[0, 0]
        daily_symbols = df.groupby('date')['symbol'].nunique()
        ax.plot(daily_symbols.index, daily_symbols.values)
        ax.set_title('Number of Symbols Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Symbols')
        ax.grid(True, alpha=0.3)

        # 2. Return distribution
        ax = axes[0, 1]
        if 'close' in df.columns:
            df_sorted = df.sort_values(['symbol', 'date'])
            returns = df_sorted.groupby('symbol')['close'].pct_change()
            returns_clean = returns[abs(returns) < 0.1]  # Remove extreme
            ax.hist(returns_clean * 100, bins=100, alpha=0.7, edgecolor='black')
            ax.set_title('Daily Return Distribution')
            ax.set_xlabel('Return (%)')
            ax.set_ylabel('Frequency')
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)

        # 3. Currency breakdown
        ax = axes[0, 2]
        if 'currency' in df.columns:
            currency_counts = df.groupby('currency')['symbol'].nunique()
            ax.pie(currency_counts.values, labels=currency_counts.index, autopct='%1.1f%%')
            ax.set_title('Symbols by Currency')
        else:
            ax.text(0.5, 0.5, 'No currency data', ha='center', va='center')
            ax.set_title('Currency Breakdown')

        # 4. Coverage heatmap (sample of symbols)
        ax = axes[1, 0]
        sample_symbols = df['symbol'].unique()[:20]
        coverage_matrix = []

        for symbol in sample_symbols:
            symbol_data = df[df['symbol'] == symbol].set_index('date')
            daily_range = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
            coverage = [1 if date in symbol_data.index else 0 for date in daily_range[:252]]
            coverage_matrix.append(coverage)

        if coverage_matrix:
            im = ax.imshow(coverage_matrix, aspect='auto', cmap='RdYlGn')
            ax.set_title('Data Coverage (First 20 Symbols, 252 Days)')
            ax.set_xlabel('Days')
            ax.set_ylabel('Symbols')

        # 5. Price evolution (sample)
        ax = axes[1, 1]
        sample_symbols = df['symbol'].unique()[:5]
        for symbol in sample_symbols:
            symbol_data = df[df['symbol'] == symbol].sort_values('date')
            if 'close' in symbol_data.columns:
                normalized = symbol_data['close'] / symbol_data['close'].iloc[0]
                ax.plot(symbol_data['date'], normalized, label=symbol, alpha=0.7)
        ax.set_title('Normalized Price Evolution (Sample)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Price')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 6. Volume distribution
        ax = axes[1, 2]
        if 'volume' in df.columns:
            volumes = df['volume'][df['volume'] > 0]
            ax.hist(np.log10(volumes), bins=50, alpha=0.7, edgecolor='black')
            ax.set_title('Log Volume Distribution')
            ax.set_xlabel('Log10(Volume)')
            ax.set_ylabel('Frequency')

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "diagnostic_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plots saved to {plot_path}")
        plt.close()

    def generate_report(self, results: Dict):
        """Generate comprehensive diagnostic report"""
        logger.info("\n📝 Generating diagnostic report...")

        report_path = self.output_dir / "diagnostic_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DATA DIAGNOSTICS REPORT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")

            # Summary of issues
            f.write("SUMMARY OF ISSUES FOUND:\n")
            f.write("-" * 40 + "\n")
            if self.issues:
                for issue in self.issues:
                    f.write(f"• {issue}\n")
            else:
                f.write("✅ No critical issues found\n")

            f.write("\n" + "=" * 60 + "\n")

            # Detailed results
            for section, data in results.items():
                f.write(f"\n{section.upper()}:\n")
                f.write("-" * 40 + "\n")
                if isinstance(data, dict):
                    for key, value in data.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {data}\n")
                f.write("\n")

        logger.info(f"Report saved to {report_path}")

        # Also save as JSON for programmatic access
        import json
        json_path = self.output_dir / "diagnostic_results.json"

        # Convert non-serializable objects
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    try:
                        json.dumps(v)
                        serializable_results[key][k] = v
                    except:
                        serializable_results[key][k] = str(v)
            else:
                serializable_results[key] = str(value)

        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"JSON results saved to {json_path}")


def main():
    """Main execution"""
    # Initialize diagnostics
    diagnostics = DataDiagnostics()

    # Check for data file
    data_paths = [
        "data/processed/prices_clean.parquet",
        "data/raw/prices_combined.csv",
        "data/raw/price_data.parquet"
    ]

    data_path = None
    for path in data_paths:
        if Path(path).exists():
            data_path = path
            break

    if not data_path:
        logger.error("No data file found!")
        logger.info("Please run fetch_all_prices.py first")
        return 1

    logger.info(f"Using data file: {data_path}")

    # Run diagnostics
    results = diagnostics.run_all_diagnostics(data_path)

    # Print final summary
    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)

    if diagnostics.issues:
        print(f"\n⚠️ Found {len(diagnostics.issues)} issues that need attention:")

        # Categorize issues
        critical = [i for i in diagnostics.issues if '❌' in i]
        warnings = [i for i in diagnostics.issues if '⚠️' in i]

        if critical:
            print(f"\nCRITICAL ISSUES ({len(critical)}):")
            for issue in critical[:5]:  # Show first 5
                print(f"  {issue}")

        if warnings:
            print(f"\nWARNINGS ({len(warnings)}):")
            for issue in warnings[:5]:  # Show first 5
                print(f"  {issue}")

        print(f"\nFull report saved to: {diagnostics.output_dir}")
    else:
        print("\n✅ All diagnostics passed!")

    return 0


if __name__ == "__main__":
    exit(main())