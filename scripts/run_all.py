#!/usr/bin/env python3
"""
enhanced_run_all.py - Complete enhanced factor investing pipeline with comprehensive analytics
"""

import sys
import os
from pathlib import Path
import logging
import time
from datetime import datetime
import subprocess
import json
import pandas as pd

# Setup enhanced logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"enhanced_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedFactorPipeline:
    """Complete enhanced factor investing pipeline with comprehensive analytics"""

    def __init__(self, quick: bool = False):
        self.start_time = time.time()
        self.results = {}
        self.quick = quick
        self.pipeline_stats = {}

    def check_environment(self) -> bool:
        """Check and install required packages"""
        logger.info("🔧 Checking environment and dependencies...")

        required_packages = [
            'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
            'yfinance', 'requests', 'lxml', 'plotly', 'scikit-learn'
        ]

        missing = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing.append(package)

        if missing:
            logger.info(f"Installing missing packages: {missing}")
            for package in missing:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    logger.info(f"✅ Installed {package}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ Failed to install {package}: {e}")
                    return False

        # Check for optional advanced packages
        optional_packages = ['plotly', 'statsmodels', 'arch']
        for package in optional_packages:
            try:
                __import__(package)
                logger.info(f"✅ Optional package {package} available")
            except ImportError:
                logger.warning(f"⚠️ Optional package {package} not available - some features may be limited")

        logger.info("✅ Environment check passed")
        return True

    def create_directory_structure(self):
        """Create comprehensive directory structure"""
        directories = [
            "data/raw",
            "data/processed",
            "data/cleaned",
            "outputs/momentum",
            "outputs/multi_factor",
            "outputs/enhanced_momentum",
            "outputs/analytics",
            "diagnostics/comprehensive",
            "diagnostics/corporate_actions",
            "diagnostics/cleaning",
            "logs",
            "config",
            "reports"
        ]

        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        logger.info("✅ Enhanced directory structure created")

    def prompt_enhanced_data_source(self) -> tuple[bool, bool, bool]:
        """Enhanced interactive dialogue for data source selection"""
        logger.info("=" * 70)
        logger.info("ENHANCED DATA SOURCE SELECTION")
        logger.info("=" * 70)

        # Check what data files exist
        raw_data_files = [
            "data/raw/price_data.parquet",
            "data/raw/prices_combined.csv",
        ]

        clean_data_file = "data/processed/prices_clean.parquet"
        enhanced_clean_file = "data/processed/prices_enhanced_clean.parquet"

        existing_raw = [f for f in raw_data_files if Path(f).exists()]
        has_clean = Path(clean_data_file).exists()
        has_enhanced_clean = Path(enhanced_clean_file).exists()

        print("\n📊 DATA FILE STATUS:")
        print("-" * 50)

        if existing_raw:
            print("✅ Raw data files found:")
            for file in existing_raw:
                size_mb = Path(file).stat().st_size / (1024 * 1024)
                modified = datetime.fromtimestamp(Path(file).stat().st_mtime)
                print(f"   • {file} ({size_mb:.1f}MB, {modified.strftime('%Y-%m-%d %H:%M')})")
        else:
            print("❌ No raw data files found")

        if has_enhanced_clean:
            size_mb = Path(enhanced_clean_file).stat().st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(Path(enhanced_clean_file).stat().st_mtime)
            print(f"✅ Enhanced clean data found:")
            print(f"   • {enhanced_clean_file} ({size_mb:.1f}MB, {modified.strftime('%Y-%m-%d %H:%M')})")
        elif has_clean:
            size_mb = Path(clean_data_file).stat().st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(Path(clean_data_file).stat().st_mtime)
            print(f"⚠️ Basic clean data found (consider upgrading):")
            print(f"   • {clean_data_file} ({size_mb:.1f}MB, {modified.strftime('%Y-%m-%d %H:%M')})")
        else:
            print("❌ No clean data files found")

        print("\n🚀 WHAT WOULD YOU LIKE TO DO?")
        print("-" * 50)
        print("1. 🔄 Fetch fresh data + full enhanced cleaning (recommended)")
        print("2. 🧹 Enhanced cleaning of existing raw data")
        print("3. ⚡ Use existing enhanced clean data (fastest)")
        print("4. 📊 Run comprehensive diagnostics only")
        print("5. 🔬 Run strategies with basic analytics")
        print("6. 💎 Full pipeline with all enhancements")

        while True:
            try:
                choice = input("\nEnter your choice (1-6): ").strip()

                if choice == '1':
                    print("🔄 Will fetch fresh data and run full enhanced cleaning")
                    return True, True, True  # fetch_fresh, enhanced_cleaning, comprehensive_analytics

                elif choice == '2':
                    if not existing_raw:
                        print("❌ No raw data available! Choose option 1.")
                        continue
                    print("🧹 Will run enhanced cleaning on existing data")
                    return False, True, True

                elif choice == '3':
                    if not has_enhanced_clean:
                        print("❌ No enhanced clean data available!")
                        if has_clean:
                            print("Basic clean data found - would you like to use it? (y/n)")
                            use_basic = input().strip().lower()
                            if use_basic == 'y':
                                print("⚡ Will use existing basic clean data")
                                return False, False, True
                        continue
                    print("⚡ Will use existing enhanced clean data")
                    return False, False, True

                elif choice == '4':
                    print("📊 Will run comprehensive diagnostics only")
                    return False, False, False  # diagnostics_only mode

                elif choice == '5':
                    print("🔬 Will run strategies with basic analytics")
                    return False, False, False  # basic analytics mode

                elif choice == '6':
                    print("💎 Will run full enhanced pipeline with all features")
                    return True, True, True

                else:
                    print("Please enter 1, 2, 3, 4, 5, or 6")

            except KeyboardInterrupt:
                print("\n\n👋 Cancelled by user")
                sys.exit(0)
            except Exception as e:
                print(f"Invalid input: {e}")

    def run_data_fetch(self, force_refresh: bool = False) -> bool:
        """Enhanced data fetching with progress tracking"""
        logger.info("=" * 70)
        logger.info("STEP 1: ENHANCED DATA FETCHING")
        logger.info("=" * 70)

        data_files = [
            "data/raw/price_data.parquet",
            "data/raw/prices_combined.csv"
        ]

        existing_data = [f for f in data_files if Path(f).exists()]

        if existing_data and not force_refresh:
            logger.info(f"Data already exists: {existing_data[0]}")
            self.results['data_fetch'] = {
                'success': True,
                'source': 'existing',
                'file': existing_data[0]
            }
            return True

        try:
            # Check if we have the fetch script
            if not Path("scripts/fetch_all_prices.py").exists():
                logger.warning("fetch_all_prices.py not found - creating minimal data for testing")
                return self._create_synthetic_data()

            from fetch_all_prices import DataFetcher

            fetcher = DataFetcher(max_tickers=10 if self.quick else None)
            data = fetcher.fetch_all_data(use_cache=not force_refresh)

            if data['prices'].empty:
                logger.error("Data fetch returned empty results!")
                return False

            self.results['data_fetch'] = {
                'success': True,
                'rows': len(data['prices']),
                'symbols': data['prices']['symbol'].nunique(),
                'source': 'fresh',
                'memory_mb': data['prices'].memory_usage(deep=True).sum() / (1024 * 1024)
            }

            logger.info("✅ Enhanced data fetch completed successfully")
            return True

        except Exception as e:
            logger.error(f"Enhanced data fetch failed: {e}")
            logger.info("Attempting to create synthetic data for testing...")
            return self._create_synthetic_data()

    def _create_synthetic_data(self) -> bool:
        """Create synthetic data for testing purposes"""
        logger.info("🧪 Creating synthetic data for testing...")

        try:
            np.random.seed(42)

            # Create synthetic price data
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'SPY', 'QQQ']
            dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')

            data_list = []

            for symbol in symbols:
                # Generate realistic price path
                returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
                price = 100 * np.exp(np.cumsum(returns))  # Price path

                volume = np.random.lognormal(15, 1, len(dates))  # Volume

                for i, date in enumerate(dates):
                    if np.random.random() > 0.99:  # Skip some days randomly
                        continue

                    # Create OHLC from close price
                    close = price[i]
                    high = close * (1 + abs(np.random.normal(0, 0.01)))
                    low = close * (1 - abs(np.random.normal(0, 0.01)))
                    open_price = close * (1 + np.random.normal(0, 0.005))

                    data_list.append({
                        'date': date,
                        'symbol': symbol,
                        'open': max(0.01, open_price),
                        'high': max(high, close, open_price),
                        'low': min(low, close, open_price),
                        'close': max(0.01, close),
                        'volume': max(1, int(volume[i])),
                        'currency': 'USD'
                    })

            df = pd.DataFrame(data_list)

            # Save synthetic data
            output_path = "data/raw/synthetic_data.parquet"
            df.to_parquet(output_path, index=False)

            self.results['data_fetch'] = {
                'success': True,
                'rows': len(df),
                'symbols': df['symbol'].nunique(),
                'source': 'synthetic',
                'file': output_path
            }

            logger.info(f"✅ Created synthetic data: {len(df):,} observations")
            return True

        except Exception as e:
            logger.error(f"Failed to create synthetic data: {e}")
            return False

    def run_enhanced_data_cleaning(self) -> bool:
        """Run enhanced data cleaning with corporate actions correction"""
        logger.info("=" * 70)
        logger.info("STEP 2: ENHANCED DATA CLEANING")
        logger.info("=" * 70)

        try:
            from enhanced_data_cleaner import clean_factor_data_enhanced

            # Find input data
            input_paths = [
                "data/raw/price_data.parquet",
                "data/raw/prices_combined.csv",
                "data/raw/synthetic_data.parquet"
            ]

            input_path = None
            for path in input_paths:
                if Path(path).exists():
                    input_path = path
                    break

            if not input_path:
                logger.error("No raw data found for enhanced cleaning")
                return False

            output_path = "data/processed/prices_enhanced_clean.parquet"

            # Check if already cleaned and recent
            if Path(output_path).exists():
                clean_time = Path(output_path).stat().st_mtime
                raw_time = Path(input_path).stat().st_mtime

                if clean_time > raw_time:
                    logger.info("Enhanced clean data is up-to-date")
                    clean_df = pd.read_parquet(output_path)
                    self.results['data_cleaning'] = {
                        'success': True,
                        'rows': len(clean_df),
                        'symbols': clean_df['symbol'].nunique(),
                        'input_file': input_path,
                        'output_file': output_path,
                        'status': 'up-to-date'
                    }
                    return True

            # Run enhanced cleaning
            logger.info(f"Starting enhanced cleaning: {input_path} -> {output_path}")
            clean_df = clean_factor_data_enhanced(input_path, output_path)

            self.results['data_cleaning'] = {
                'success': True,
                'rows': len(clean_df),
                'symbols': clean_df['symbol'].nunique(),
                'input_file': input_path,
                'output_file': output_path,
                'status': 'completed',
                'data_quality_score': clean_df.get('data_quality_score', pd.Series([1.0])).mean()
            }

            logger.info("✅ Enhanced data cleaning completed successfully")
            return True

        except Exception as e:
            logger.error(f"Enhanced data cleaning failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_comprehensive_diagnostics(self) -> bool:
        """Run comprehensive data diagnostics"""
        logger.info("=" * 70)
        logger.info("STEP 3: COMPREHENSIVE DATA DIAGNOSTICS")
        logger.info("=" * 70)

        try:
            from enhanced_diagnostics import run_comprehensive_diagnostics

            # Use best available data
            data_paths = [
                "data/processed/prices_enhanced_clean.parquet",
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

            logger.info(f"Running comprehensive diagnostics on: {data_path}")

            # Run comprehensive diagnostics
            diagnostic_results = run_comprehensive_diagnostics(data_path)

            self.results['diagnostics'] = {
                'success': True,
                'data_file': data_path,
                'results': diagnostic_results,
                'comprehensive': True
            }

            logger.info("✅ Comprehensive diagnostics completed")
            return True

        except Exception as e:
            logger.error(f"Comprehensive diagnostics failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_enhanced_momentum_strategy(self) -> bool:
        """Run enhanced momentum strategy"""
        logger.info("=" * 70)
        logger.info("STEP 4: ENHANCED MOMENTUM STRATEGY")
        logger.info("=" * 70)

        try:
            from enhanced_momentum_strategy import EnhancedMomentumStrategy, EnhancedMomentumConfig

            # Enhanced configuration
            config = EnhancedMomentumConfig(
                lookback_months=12,
                skip_months=2,
                n_portfolios=10,
                use_dynamic_sizing=True,
                regime_aware=True,
                volatility_scaling=True,
                momentum_decay=True,
                cross_sectional_neutralization=True,
                spread_cost_bps=10,
                market_impact_bps=5,
                brokerage_cost_bps=2
            )

            strategy = EnhancedMomentumStrategy(config)

            # Use best available data
            data_paths = [
                "data/processed/prices_enhanced_clean.parquet",
                "data/processed/prices_clean.parquet",
                "data/raw/price_data.parquet"
            ]

            data_path = None
            for path in data_paths:
                if Path(path).exists():
                    data_path = path
                    break

            if not data_path:
                logger.error("No data file found for momentum strategy")
                return False

            logger.info(f"Running enhanced momentum strategy on: {data_path}")
            results = strategy.run_comprehensive_backtest(data_path)

            if not results:
                logger.error("Enhanced momentum strategy failed")
                return False

            # Extract key metrics
            analytics = results.get('analytics', {})
            core_metrics = analytics.get('core_metrics', {})
            risk_metrics = analytics.get('risk_analysis', {})
            significance = analytics.get('significance_tests', {})

            self.results['enhanced_momentum'] = {
                'success': True,
                'data_file': data_path,
                'annual_return': core_metrics.get('annual_return', 0),
                'sharpe_ratio': core_metrics.get('sharpe_ratio', 0),
                'max_drawdown': risk_metrics.get('max_drawdown', 0),
                'win_rate': core_metrics.get('win_rate', 0),
                'is_significant': significance.get('sharpe_ratio_test', {}).get('is_significant', False),
                'config_applied': config.__dict__,
                'full_results': results
            }

            logger.info("✅ Enhanced momentum strategy completed")
            return True

        except Exception as e:
            logger.error(f"Enhanced momentum strategy failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_multi_factor_strategy(self) -> bool:
        """Run multi-factor strategy (using fixed version)"""
        logger.info("=" * 70)
        logger.info("STEP 5: ENHANCED MULTI-FACTOR STRATEGY")
        logger.info("=" * 70)

        try:
            from multi_factor_strategy_fixed import MultiFactorStrategy, MultiFactorConfig

            config = MultiFactorConfig(
                momentum_weight=0.3,
                value_weight=0.25,
                quality_weight=0.25,
                low_vol_weight=0.2,
                use_risk_parity=True,
                tcost_bps=20
            )

            strategy = MultiFactorStrategy(config)

            # Use best available data
            data_paths = [
                "data/processed/prices_enhanced_clean.parquet",
                "data/processed/prices_clean.parquet",
                "data/raw/price_data.parquet"
            ]

            data_path = None
            for path in data_paths:
                if Path(path).exists():
                    data_path = path
                    break

            if not data_path:
                logger.error("No data file found for multi-factor strategy")
                return False

            logger.info(f"Running multi-factor strategy on: {data_path}")
            results = strategy.run_backtest(data_path)

            if not results:
                logger.error("Multi-factor strategy failed")
                return False

            # Extract key metrics
            metrics = results.get('metrics', {})

            self.results['multi_factor'] = {
                'success': True,
                'data_file': data_path,
                'annual_return': metrics.get('net_annual_return', 0),
                'sharpe_ratio': metrics.get('net_sharpe', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'is_significant': metrics.get('p_value', 1) < 0.05,
                'full_results': results
            }

            logger.info("✅ Multi-factor strategy completed")
            return True

        except Exception as e:
            logger.error(f"Multi-factor strategy failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_comprehensive_report(self):
        """Generate comprehensive pipeline report"""
        logger.info("=" * 70)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("=" * 70)

        report_path = Path("reports/comprehensive_pipeline_report.txt")
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE FACTOR INVESTING PIPELINE REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Total Runtime: {(time.time() - self.start_time) / 60:.2f} minutes\n")
            f.write("=" * 80 + "\n\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")

            successful_steps = sum(1 for result in self.results.values() if result.get('success', False))
            total_steps = len(self.results)

            f.write(
                f"Pipeline Success Rate: {successful_steps}/{total_steps} ({successful_steps / total_steps * 100:.1f}%)\n")

            if 'enhanced_momentum' in self.results and self.results['enhanced_momentum']['success']:
                momentum_sharpe = self.results['enhanced_momentum']['sharpe_ratio']
                f.write(f"Enhanced Momentum Sharpe: {momentum_sharpe:.3f}\n")

            if 'multi_factor' in self.results and self.results['multi_factor']['success']:
                mf_sharpe = self.results['multi_factor']['sharpe_ratio']
                f.write(f"Multi-Factor Sharpe: {mf_sharpe:.3f}\n")

            f.write("\n")

            # Detailed Results
            for step_name, step_results in self.results.items():
                f.write(f"{step_name.upper().replace('_', ' ')}:\n")
                f.write("-" * 40 + "\n")

                if step_results.get('success'):
                    f.write("Status: SUCCESS\n")

                    if step_name == 'data_fetch':
                        f.write(f"Source: {step_results.get('source', 'unknown')}\n")
                        f.write(f"Rows: {step_results.get('rows', 0):,}\n")
                        f.write(f"Symbols: {step_results.get('symbols', 0)}\n")
                        f.write(f"Memory: {step_results.get('memory_mb', 0):.1f} MB\n")

                    elif step_name == 'data_cleaning':
                        f.write(f"Input: {step_results.get('input_file', 'unknown')}\n")
                        f.write(f"Output: {step_results.get('output_file', 'unknown')}\n")
                        f.write(f"Final Rows: {step_results.get('rows', 0):,}\n")
                        f.write(f"Final Symbols: {step_results.get('symbols', 0)}\n")
                        f.write(f"Quality Score: {step_results.get('data_quality_score', 0):.3f}\n")

                    elif step_name == 'diagnostics':
                        f.write(f"Data File: {step_results.get('data_file', 'unknown')}\n")
                        f.write(f"Comprehensive: {step_results.get('comprehensive', False)}\n")

                    elif step_name in ['enhanced_momentum', 'multi_factor']:
                        f.write(f"Data File: {step_results.get('data_file', 'unknown')}\n")
                        f.write(f"Annual Return: {step_results.get('annual_return', 0) * 100:.2f}%\n")
                        f.write(f"Sharpe Ratio: {step_results.get('sharpe_ratio', 0):.3f}\n")
                        f.write(f"Max Drawdown: {step_results.get('max_drawdown', 0) * 100:.2f}%\n")
                        f.write(f"Win Rate: {step_results.get('win_rate', 0) * 100:.2f}%\n")
                        f.write(
                            f"Statistically Significant: {'YES' if step_results.get('is_significant', False) else 'NO'}\n")
                else:
                    f.write("Status: FAILED\n")

                f.write("\n")

            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")

            recommendations = []

            if 'data_cleaning' in self.results and self.results['data_cleaning']['success']:
                quality_score = self.results['data_cleaning'].get('data_quality_score', 0)
                if quality_score > 0.8:
                    recommendations.append("Data quality is excellent - results are reliable")
                elif quality_score > 0.6:
                    recommendations.append("Data quality is acceptable - monitor for improvements")
                else:
                    recommendations.append("Data quality needs improvement - investigate data sources")

            if 'enhanced_momentum' in self.results and self.results['enhanced_momentum']['success']:
                sharpe = self.results['enhanced_momentum']['sharpe_ratio']
                if sharpe > 1.0:
                    recommendations.append("Momentum strategy shows excellent risk-adjusted returns")
                elif sharpe > 0.5:
                    recommendations.append("Momentum strategy shows good potential - consider refinements")
                else:
                    recommendations.append("Momentum strategy needs significant improvement")

            if not recommendations:
                recommendations.append("Complete the full pipeline to get specific recommendations")

            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")

        # Save JSON results
        json_path = Path("reports/pipeline_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"📝 Comprehensive report saved to {report_path}")
        logger.info(f"📊 JSON results saved to {json_path}")

    def run(self, interactive: bool = True):
        """Run the complete enhanced pipeline"""
        logger.info("=" * 80)
        logger.info("ENHANCED FACTOR INVESTING PIPELINE - STARTING")
        logger.info("=" * 80)
        logger.info(f"Start time: {datetime.now()}")
        logger.info("")

        # Environment check
        if not self.check_environment():
            logger.error("Environment check failed!")
            return 1

        # Directory structure
        self.create_directory_structure()

        # Interactive mode
        if interactive:
            fetch_fresh, enhanced_cleaning, comprehensive_analytics = self.prompt_enhanced_data_source()
        else:
            fetch_fresh, enhanced_cleaning, comprehensive_analytics = True, True, True

        # Define pipeline steps based on user choice
        steps = []

        if fetch_fresh:
            steps.append(("Data Fetch", self.run_data_fetch))

        if enhanced_cleaning:
            steps.append(("Enhanced Data Cleaning", self.run_enhanced_data_cleaning))

        if comprehensive_analytics:
            steps.extend([
                ("Comprehensive Diagnostics", self.run_comprehensive_diagnostics),
                ("Enhanced Momentum Strategy", self.run_enhanced_momentum_strategy),
                ("Multi-Factor Strategy", self.run_multi_factor_strategy)
            ])

        # Execute pipeline
        for step_name, step_func in steps:
            logger.info(f"\n🚀 Running: {step_name}")
            step_start_time = time.time()

            success = step_func()
            step_duration = time.time() - step_start_time

            if success:
                logger.info(f"✅ {step_name} completed in {step_duration:.1f}s")
            else:
                logger.error(f"❌ {step_name} failed after {step_duration:.1f}s")
                logger.info("Continuing with remaining steps...")

        # Generate comprehensive report
        self.generate_comprehensive_report()

        # Final summary
        runtime = (time.time() - self.start_time) / 60
        logger.info("")
        logger.info("=" * 80)
        logger.info("ENHANCED PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total runtime: {runtime:.2f} minutes")

        # Print executive summary
        self.print_executive_summary()

        return 0

    def print_executive_summary(self):
        """Print executive summary of results"""
        print("\n" + "=" * 80)
        print("📊 EXECUTIVE SUMMARY")
        print("=" * 80)

        # Pipeline success
        successful_steps = sum(1 for result in self.results.values() if result.get('success', False))
        total_steps = len(self.results)
        print(f"Pipeline Success: {successful_steps}/{total_steps} steps completed")

        # Data summary
        if 'data_cleaning' in self.results and self.results['data_cleaning']['success']:
            cleaning = self.results['data_cleaning']
            print(f"Data Quality: {cleaning['symbols']} symbols, {cleaning['rows']:,} observations")
            print(f"Quality Score: {cleaning.get('data_quality_score', 0):.3f}/1.0")

        # Strategy results
        if 'enhanced_momentum' in self.results and self.results['enhanced_momentum']['success']:
            momentum = self.results['enhanced_momentum']
            print(
                f"Enhanced Momentum: {momentum['annual_return'] * 100:.2f}% return, {momentum['sharpe_ratio']:.3f} Sharpe")
            print(f"                  {'✅ Significant' if momentum['is_significant'] else '❌ Not significant'}")

        if 'multi_factor' in self.results and self.results['multi_factor']['success']:
            mf = self.results['multi_factor']
            print(f"Multi-Factor: {mf['annual_return'] * 100:.2f}% return, {mf['sharpe_ratio']:.3f} Sharpe")
            print(f"              {'✅ Significant' if mf['is_significant'] else '❌ Not significant'}")

        print("-" * 80)
        print(f"📁 Detailed results: reports/comprehensive_pipeline_report.txt")
        print(f"📊 Visualizations: outputs/*/figures/")
        print(f"🔬 Diagnostics: diagnostics/comprehensive/")
        print("=" * 80)


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Factor Investing Pipeline')
    parser.add_argument('--quick', action='store_true', help='Quick run with subset of data')
    parser.add_argument('--non-interactive', action='store_true', help='Run without user prompts')
    parser.add_argument('--diagnostics-only', action='store_true', help='Run diagnostics only')

    args = parser.parse_args()

    # Create and run enhanced pipeline
    pipeline = EnhancedFactorPipeline(quick=args.quick)

    if args.diagnostics_only:
        # Run diagnostics only
        pipeline.check_environment()
        pipeline.create_directory_structure()
        success = pipeline.run_comprehensive_diagnostics()
        return 0 if success else 1
    else:
        # Run full pipeline
        return pipeline.run(interactive=not args.non_interactive)


if __name__ == "__main__":
    exit(main())