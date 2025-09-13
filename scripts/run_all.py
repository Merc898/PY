#!/usr/bin/env python3
"""
run_all.py - Master script to run the complete factor investing pipeline
"""

import sys
import os
from pathlib import Path
import logging
import time
from datetime import datetime
import subprocess
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FactorPipeline:
    """Complete factor investing pipeline orchestrator"""

    def __init__(self, quick: bool = False):
        self.start_time = time.time()
        self.results = {}
        # Quick mode limits data fetching for faster runs
        self.quick = quick

    def check_environment(self) -> bool:
        """Check if all required packages are installed"""
        logger.info("Checking environment...")

        required_packages = [
            'pandas', 'numpy', 'yfinance', 'matplotlib',
            'seaborn', 'scipy', 'requests', 'lxml'
        ]

        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if missing:
            logger.error(f"Missing packages: {missing}")
            logger.info("Installing missing packages...")

            for package in missing:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                except subprocess.CalledProcessError:
                    logger.error(f"Failed to install {package}")
                    return False

        logger.info("✅ Environment check passed")
        return True

    def create_directory_structure(self):
        """Create necessary directories"""
        directories = [
            "data/raw",
            "data/processed",
            "outputs/momentum",
            "outputs/value",
            "outputs/combined",
            "diagnostics",
            "logs",
            "config"
        ]

        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        logger.info("✅ Directory structure created")

    def run_data_fetch(self, force_refresh: bool = False) -> bool:
        """Run data fetching script"""
        logger.info("=" * 60)
        logger.info("STEP 1: DATA FETCHING")
        logger.info("=" * 60)

        # Check if data already exists
        data_file = Path("data/processed/prices_clean.parquet")

        if data_file.exists() and not force_refresh:
            logger.info("Data already exists. Skipping fetch (use --refresh to force)")
            return True

        try:
            # Import and run fetch script
            from fetch_all_prices import DataFetcher

            fetcher = DataFetcher(max_tickers=5 if self.quick else None)
            data = fetcher.fetch_all_data(use_cache=not force_refresh)

            if data['prices'].empty:
                logger.error("Data fetch failed!")
                return False

            self.results['data_fetch'] = {
                'success': True,
                'rows': len(data['prices']),
                'symbols': data['prices']['symbol'].nunique()
            }

            logger.info("✅ Data fetch completed")
            return True

        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            return False

    def run_diagnostics(self) -> bool:
        """Run diagnostic checks"""
        logger.info("=" * 60)
        logger.info("STEP 2: DATA DIAGNOSTICS")
        logger.info("=" * 60)

        try:
            from diagnostics import DataDiagnostics

            diagnostics = DataDiagnostics()

            # Find data file
            data_paths = [
                "data/processed/prices_clean.parquet",
                "data/raw/price_data.parquet"
            ]

            data_path = None
            for path in data_paths:
                if Path(path).exists():
                    data_path = path
                    break

            if not data_path:
                logger.error("No data file found for diagnostics")
                return False

            results = diagnostics.run_all_diagnostics(data_path)

            # Check for critical issues
            critical_issues = [i for i in diagnostics.issues if '❌' in i]

            self.results['diagnostics'] = {
                'success': len(critical_issues) == 0,
                'critical_issues': len(critical_issues),
                'warnings': len([i for i in diagnostics.issues if '⚠️' in i]),
                'issues': diagnostics.issues[:5]  # First 5 issues
            }

            if critical_issues:
                logger.warning(f"⚠️ {len(critical_issues)} critical issues found")
                logger.info("Continuing despite issues (in production, these should be fixed)")
            else:
                logger.info("✅ Diagnostics completed - no critical issues")

            return True

        except Exception as e:
            logger.error(f"Diagnostics failed: {e}")
            return False

    def run_momentum_strategy(self) -> bool:
        """Run momentum strategy backtest"""
        logger.info("=" * 60)
        logger.info("STEP 3: MOMENTUM STRATEGY")
        logger.info("=" * 60)

        try:
            from momentum_strategy import MomentumStrategy, StrategyConfig

            # Configure strategy
            config = StrategyConfig(
                lookback_months=12,
                skip_months=2,
                n_portfolios=10,
                spread_cost_bps=10,
                market_impact_bps=5,
                brokerage_cost_bps=2
            )

            # Run backtest
            strategy = MomentumStrategy(config)
            results = strategy.run_backtest("data/processed/prices_clean.parquet")

            if not results:
                logger.error("Momentum strategy failed")
                return False

            self.results['momentum'] = {
                'success': True,
                'net_sharpe': results['metrics']['net_sharpe'],
                'net_return': results['metrics']['net_annual_return'],
                'max_drawdown': results['metrics']['max_drawdown'],
                'is_significant': results['metrics']['is_significant']
            }

            logger.info("✅ Momentum strategy completed")
            return True

        except Exception as e:
            logger.error(f"Momentum strategy failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_summary_report(self):
        """Generate final summary report"""
        logger.info("=" * 60)
        logger.info("GENERATING SUMMARY REPORT")
        logger.info("=" * 60)

        report_path = Path("outputs/pipeline_summary.txt")

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("FACTOR INVESTING PIPELINE - SUMMARY REPORT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Total Runtime: {(time.time() - self.start_time) / 60:.2f} minutes\n")
            f.write("=" * 60 + "\n\n")

            # Data Fetch Results
            if 'data_fetch' in self.results:
                f.write("DATA FETCH:\n")
                f.write("-" * 40 + "\n")
                df_results = self.results['data_fetch']
                f.write(f"  Status: {'✅ Success' if df_results['success'] else '❌ Failed'}\n")
                if df_results['success']:
                    f.write(f"  Rows: {df_results['rows']:,}\n")
                    f.write(f"  Symbols: {df_results['symbols']}\n")
                f.write("\n")

            # Diagnostics Results
            if 'diagnostics' in self.results:
                f.write("DIAGNOSTICS:\n")
                f.write("-" * 40 + "\n")
                diag_results = self.results['diagnostics']
                f.write(f"  Status: {'✅ Passed' if diag_results['success'] else '⚠️ Issues Found'}\n")
                f.write(f"  Critical Issues: {diag_results['critical_issues']}\n")
                f.write(f"  Warnings: {diag_results['warnings']}\n")
                if diag_results['issues']:
                    f.write("  Top Issues:\n")
                    for issue in diag_results['issues'][:3]:
                        f.write(f"    • {issue}\n")
                f.write("\n")

            # Momentum Strategy Results
            if 'momentum' in self.results:
                f.write("MOMENTUM STRATEGY:\n")
                f.write("-" * 40 + "\n")
                mom_results = self.results['momentum']
                f.write(f"  Status: {'✅ Success' if mom_results['success'] else '❌ Failed'}\n")
                if mom_results['success']:
                    f.write(f"  Net Annual Return: {mom_results['net_return'] * 100:.2f}%\n")
                    f.write(f"  Net Sharpe Ratio: {mom_results['net_sharpe']:.3f}\n")
                    f.write(f"  Max Drawdown: {mom_results['max_drawdown'] * 100:.2f}%\n")
                    f.write(f"  Statistically Significant: {'✅ Yes' if mom_results['is_significant'] else '❌ No'}\n")
                f.write("\n")

            # Overall Assessment
            f.write("=" * 60 + "\n")
            f.write("OVERALL ASSESSMENT:\n")
            f.write("-" * 40 + "\n")

            all_success = all(r.get('success', False) for r in self.results.values())

            if all_success:
                f.write("✅ Pipeline completed successfully!\n")
            else:
                f.write("⚠️ Pipeline completed with issues\n")

            # Recommendations
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")

            if 'diagnostics' in self.results:
                if self.results['diagnostics']['critical_issues'] > 0:
                    f.write("1. ❌ Fix critical data quality issues before production use\n")
                else:
                    f.write("1. ✅ Data quality acceptable for research\n")

            if 'momentum' in self.results:
                if self.results['momentum'].get('is_significant'):
                    f.write("2. ✅ Momentum strategy shows statistical significance\n")
                else:
                    f.write("2. ⚠️ Momentum strategy not statistically significant\n")

                if self.results['momentum'].get('net_sharpe', 0) > 0.5:
                    f.write("3. ✅ Sharpe ratio acceptable for further research\n")
                else:
                    f.write("3. ⚠️ Sharpe ratio below typical threshold (0.5)\n")

        logger.info(f"Summary report saved to {report_path}")

        # Also save as JSON
        json_path = Path("outputs/pipeline_results.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"JSON results saved to {json_path}")

    def run(self, refresh_data: bool = False):
        """Run complete pipeline"""
        logger.info("=" * 60)
        logger.info("FACTOR INVESTING PIPELINE - STARTING")
        logger.info("=" * 60)
        logger.info(f"Start time: {datetime.now()}")
        logger.info("")

        # Check environment
        if not self.check_environment():
            logger.error("Environment check failed!")
            return 1

        # Create directories
        self.create_directory_structure()

        # Run pipeline steps
        steps = [
            ("Data Fetch", lambda: self.run_data_fetch(refresh_data)),
            ("Diagnostics", self.run_diagnostics),
            ("Momentum Strategy", self.run_momentum_strategy)
        ]

        for step_name, step_func in steps:
            logger.info(f"\nRunning: {step_name}")
            success = step_func()

            if not success:
                logger.error(f"Step failed: {step_name}")
                logger.info("Continuing with remaining steps...")

        # Generate summary
        self.generate_summary_report()

        # Final summary
        runtime = (time.time() - self.start_time) / 60
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total runtime: {runtime:.2f} minutes")

        # Print quick summary
        print("\n📊 QUICK SUMMARY:")
        print("-" * 40)

        if 'data_fetch' in self.results:
            print(f"Data: {self.results['data_fetch'].get('symbols', 0)} symbols")

        if 'diagnostics' in self.results:
            issues = self.results['diagnostics']['critical_issues']
            print(f"Data Quality: {issues} critical issues")

        if 'momentum' in self.results:
            mom = self.results['momentum']
            print(f"Momentum Sharpe: {mom.get('net_sharpe', 0):.3f}")
            print(f"Momentum Return: {mom.get('net_return', 0) * 100:.2f}%")
            print(f"Significant: {'Yes ✅' if mom.get('is_significant') else 'No ❌'}")

        print("-" * 40)
        print(f"Full report: outputs/pipeline_summary.txt")

        return 0


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Run factor investing pipeline')
    parser.add_argument('--refresh', action='store_true',
                        help='Force refresh of data (ignore cache)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run with subset of data')

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = FactorPipeline()
    return pipeline.run(refresh_data=args.refresh)


if __name__ == "__main__":
    exit(main())