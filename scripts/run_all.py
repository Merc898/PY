#!/usr/bin/env python3
"""
run_all.py – minimal pipeline runner

Usage examples
--------------
python run_all.py                 # use existing data and rerun strategies
python run_all.py --refresh-data  # fetch & clean new data, rerun strategies
python run_all.py --skip-strats   # refresh data but skip strategies
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from fetch_all_prices import DataFetcher
import momentum_strategy
import reversal_strategy
import multi_factor_strategy

RAW_FILE = Path("data/raw/prices.parquet")
CLEAN_FILE = Path("data/processed/prices_clean.parquet")


def fetch_and_clean() -> None:
    """Fetch raw prices and write a cleaned parquet file."""
    logging.info("Fetching price data…")
    fetcher = DataFetcher()
    df = fetcher.fetch_all_data(use_cache=False)
    if df.empty:
        raise RuntimeError("No data fetched")

    logging.info("Cleaning price data…")
    df = df.dropna().sort_values(["symbol", "date"])
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_FILE, compression="snappy")
    logging.info("Saved cleaned data to %s", CLEAN_FILE)


def run_strategies() -> None:
    """Execute all strategies using the cleaned data."""
    logging.info("Running momentum strategy…")
    momentum_strategy.main()
    logging.info("Running reversal strategy…")
    reversal_strategy.main()
    logging.info("Running multi-factor strategy…")
    multi_factor_strategy.main()


def run_all(refresh_data: bool, rerun_strats: bool) -> None:
    """Orchestrate the full pipeline based on user options."""
    if refresh_data:
        fetch_and_clean()

    if rerun_strats:
        run_strategies()
    else:
        logging.info("Skipping strategy execution")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline runner")
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Fetch and clean new data before running strategies",
    )
    parser.add_argument(
        "--skip-strats",
        action="store_true",
        help="Do not rerun strategies (useful when only refreshing data)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    run_all(refresh_data=args.refresh_data, rerun_strats=not args.skip_strats)


if __name__ == "__main__":
    main()
