#!/usr/bin/env python3
"""
verify_fixes.py - Test script to verify data quality fixes worked
CREATE THIS AS: scripts/verify_fixes.py (OPTIONAL)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_data_quality():
    """Verify the data cleaning fixes worked properly"""

    logger.info("🔍 VERIFYING DATA QUALITY FIXES")
    logger.info("=" * 50)

    # Check if cleaned data exists
    clean_file = Path("data/processed/prices_clean.parquet")

    if not clean_file.exists():
        logger.error("❌ Cleaned data file not found!")
        logger.info("Run 'python run_all.py --refresh' first")
        return False

    # Load cleaned data
    df = pd.read_parquet(clean_file)
    logger.info(f"📊 Loaded cleaned data: {len(df):,} observations")

    # Test 1: Currency Check
    logger.info("\n1️⃣ CURRENCY CHECK:")
    if 'currency' in df.columns:
        currencies = df['currency'].unique()
        logger.info(f"   Currencies found: {currencies}")

        if len(currencies) == 1 and currencies[0] == 'USD':
            logger.info("   ✅ PASS: Single currency (USD only)")
        else:
            logger.warning(f"   ❌ FAIL: Multiple currencies: {currencies}")
            return False
    else:
        logger.info("   ⚠️  No currency column (assuming USD)")

    # Test 2: Price Consistency Check
    logger.info("\n2️⃣ PRICE CONSISTENCY CHECK:")

    # Check for negative/zero prices
    if 'close' in df.columns:
        negative_prices = (df['close'] <= 0).sum()
        if negative_prices == 0:
            logger.info("   ✅ PASS: No negative/zero prices")
        else:
            logger.warning(f"   ❌ FAIL: {negative_prices} negative/zero prices found")
            return False

    # Check OHLC relationships
    ohlc_cols = ['open', 'high', 'low', 'close']
    if all(col in df.columns for col in ohlc_cols):
        # High >= Low
        hl_issues = (df['high'] < df['low']).sum()
        if hl_issues == 0:
            logger.info("   ✅ PASS: No High < Low issues")
        else:
            logger.warning(f"   ❌ FAIL: {hl_issues} High < Low issues")
            return False

        # High >= Open, Close
        h_issues = ((df['high'] < df['open']) | (df['high'] < df['close'])).sum()
        if h_issues == 0:
            logger.info("   ✅ PASS: High >= Open/Close")
        else:
            logger.warning(f"   ❌ FAIL: {h_issues} High < Open/Close issues")
            return False

    # Test 3: Corporate Actions Check
    logger.info("\n3️⃣ CORPORATE ACTIONS CHECK:")
    df_sorted = df.sort_values(['symbol', 'date'])
    df_sorted['daily_return'] = df_sorted.groupby('symbol')['close'].pct_change()

    # Count extreme returns (should be capped)
    extreme_pos = (df_sorted['daily_return'] > 0.5).sum()
    extreme_neg = (df_sorted['daily_return'] < -0.5).sum()

    if extreme_pos + extreme_neg <= 10:  # Allow a few edge cases
        logger.info(f"   ✅ PASS: Only {extreme_pos + extreme_neg} extreme returns (>50%)")
    else:
        logger.warning(f"   ⚠️  WARNING: {extreme_pos + extreme_neg} extreme returns remain")
        logger.info("   (May indicate remaining split issues)")

    # Test 4: Data Quality Metrics
    logger.info("\n4️⃣ DATA QUALITY METRICS:")

    # Return statistics
    returns = df_sorted['daily_return'].dropna()
    if len(returns) > 0:
        logger.info(f"   Return Mean: {returns.mean():.4f}")
        logger.info(f"   Return Std: {returns.std():.4f}")
        logger.info(f"   Kurtosis: {returns.kurtosis():.2f}")

        # Good data should have kurtosis < 50 (yours was 33)
        if returns.kurtosis() < 50:
            logger.info("   ✅ PASS: Reasonable return distribution")
        else:
            logger.warning("   ⚠️  WARNING: High kurtosis - fat tails remain")

    # Test 5: Data Completeness
    logger.info("\n5️⃣ DATA COMPLETENESS:")
    logger.info(f"   Total observations: {len(df):,}")
    logger.info(f"   Unique symbols: {df['symbol'].nunique()}")
    logger.info(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Check for missing values
    missing = df.isnull().sum()
    critical_missing = missing[missing > 0]

    if len(critical_missing) == 0:
        logger.info("   ✅ PASS: No missing values in critical columns")
    else:
        logger.info(f"   ⚠️  Missing values found:")
        for col, count in critical_missing.items():
            logger.info(f"     {col}: {count:,} missing")

    # Test 6: Future Data Check
    logger.info("\n6️⃣ FUTURE DATA CHECK:")
    now = pd.Timestamp.now()
    future_data = (df['date'] > now).sum()

    if future_data == 0:
        logger.info("   ✅ PASS: No future-dated observations")
    else:
        logger.warning(f"   ❌ FAIL: {future_data} future-dated observations")
        return False

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("🎉 DATA QUALITY VERIFICATION COMPLETE!")
    logger.info("✅ All critical tests passed - data is ready for analysis")

    return True


def compare_before_after():
    """Compare raw vs cleaned data statistics"""

    logger.info("\n📈 BEFORE vs AFTER COMPARISON")
    logger.info("=" * 50)

    raw_files = [
        "data/raw/price_data.parquet",
        "data/raw/prices_combined.csv"
    ]

    raw_file = None
    for file in raw_files:
        if Path(file).exists():
            raw_file = file
            break

    if not raw_file:
        logger.info("No raw data file found for comparison")
        return

    clean_file = "data/processed/prices_clean.parquet"

    if not Path(clean_file).exists():
        logger.info("No clean data file found for comparison")
        return

    # Load both datasets
    if raw_file.endswith('.parquet'):
        raw_df = pd.read_parquet(raw_file)
    else:
        raw_df = pd.read_csv(raw_file)

    clean_df = pd.read_parquet(clean_file)

    logger.info("BEFORE (Raw Data):")
    logger.info(f"  Observations: {len(raw_df):,}")
    logger.info(f"  Symbols: {raw_df['symbol'].nunique()}")

    if 'currency' in raw_df.columns:
        currencies = raw_df['currency'].value_counts()
        logger.info(f"  Currencies: {dict(currencies)}")

    logger.info("\nAFTER (Clean Data):")
    logger.info(f"  Observations: {len(clean_df):,}")
    logger.info(f"  Symbols: {clean_df['symbol'].nunique()}")

    if 'currency' in clean_df.columns:
        currencies = clean_df['currency'].value_counts()
        logger.info(f"  Currencies: {dict(currencies)}")

    # Show what was removed
    removed_obs = len(raw_df) - len(clean_df)
    removed_symbols = raw_df['symbol'].nunique() - clean_df['symbol'].nunique()

    logger.info(f"\nREMOVED:")
    logger.info(f"  Observations: {removed_obs:,} ({removed_obs / len(raw_df) * 100:.1f}%)")
    logger.info(f"  Symbols: {removed_symbols} ({removed_symbols / raw_df['symbol'].nunique() * 100:.1f}%)")


def main():
    """Run verification tests"""

    success = verify_data_quality()

    if success:
        compare_before_after()
        print("\n🎉 VERIFICATION PASSED - Your fixes worked!")
        print("You can now run strategies on the cleaned data with confidence.")
    else:
        print("\n❌ VERIFICATION FAILED - Issues remain")
        print("Check the logs above and ensure all fixes were properly implemented.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())