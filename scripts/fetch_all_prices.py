#!/usr/bin/env python3
"""
fetch_all_prices.py - Robust price data fetching with proper handling
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import logging
import os
from typing import List, Dict, Tuple
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Robust data fetcher with error handling and validation"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)

    def get_sp500_tickers(self) -> List[str]:
        """Fetch current S&P 500 constituents"""
        try:
            # Get S&P 500 list from Wikipedia
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].str.replace('.', '-').tolist()
            logger.info(f"Retrieved {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 tickers: {e}")
            # Fallback to a smaller list
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                    'BRK-B', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA']

    def get_nasdaq100_tickers(self) -> List[str]:
        """Fetch NASDAQ-100 constituents"""
        try:
            tables = pd.read_html('https://en.wikipedia.org/wiki/NASDAQ-100')
            nasdaq_table = tables[4]  # The index might change
            tickers = nasdaq_table['Ticker'].str.replace('.', '-').tolist()
            logger.info(f"Retrieved {len(tickers)} NASDAQ-100 tickers")
            return tickers
        except Exception as e:
            logger.error(f"Failed to fetch NASDAQ-100 tickers: {e}")
            return []

    def get_euro_stoxx_tickers(self) -> List[str]:
        """Euro Stoxx 50 tickers - with currency tags"""
        # These are traded in EUR - we'll tag them
        euro_tickers = [
            'AIR.PA', 'ALV.DE', 'ASML.AS', 'BAS.DE', 'BBVA.MC',
            'BNP.PA', 'CRH.L', 'CS.PA', 'DTE.DE', 'DB1.DE',
            'DBK.DE', 'ENEL.MI', 'ENGI.PA', 'ENI.MI', 'IBE.MC',
            'INGA.AS', 'ITX.MC', 'MC.PA', 'MUV2.DE', 'OR.PA',
            'PHIA.AS', 'SAF.PA', 'SAN.MC', 'SAP.DE', 'SIE.DE',
            'SU.PA', 'TTE.PA', 'UNA.AS', 'VIV.PA', 'VOW3.DE'
        ]
        return euro_tickers

    def identify_currency(self, ticker: str) -> str:
        """Identify currency based on exchange suffix"""
        eur_suffixes = ['.PA', '.DE', '.AS', '.MC', '.MI', '.MA', '.FR', '.BR', '.LI']
        gbp_suffixes = ['.L', '.LON']

        for suffix in eur_suffixes:
            if ticker.endswith(suffix):
                return 'EUR'
        for suffix in gbp_suffixes:
            if ticker.endswith(suffix):
                return 'GBP'

        return 'USD'  # Default to USD

    def fetch_price_data(self, ticker: str, start_date: str = "2000-01-01") -> pd.DataFrame:
        """Fetch price data for a single ticker with error handling"""
        try:
            # Add delay to avoid rate limiting
            time.sleep(0.1)

            # Download data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=datetime.now(), auto_adjust=True)

            if df.empty:
                logger.warning(f"No data retrieved for {ticker}")
                return pd.DataFrame()

            # Add metadata
            df['symbol'] = ticker
            df['currency'] = self.identify_currency(ticker)

            # Reset index to have date as column
            df = df.reset_index()

            # Select relevant columns
            df = df[['Date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'currency']]
            df.columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'currency']

            # Data quality checks
            df = self.validate_price_data(df, ticker)

            return df

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def validate_price_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Validate and clean price data"""
        if df.empty:
            return df

        initial_count = len(df)

        # Remove rows with invalid prices
        df = df[df['close'] > 0]
        df = df[df['volume'] >= 0]

        # Check for extreme daily returns (likely errors or splits)
        df['daily_return'] = df['close'].pct_change()

        # Flag potential issues but don't remove automatically
        suspicious_returns = df[abs(df['daily_return']) > 0.5]
        if len(suspicious_returns) > 0:
            logger.warning(f"{ticker}: {len(suspicious_returns)} days with >50% returns detected")

        # Remove truly impossible values (>90% daily moves)
        df = df[abs(df['daily_return']) < 0.9]

        # Check for stale prices (same close for 5+ consecutive days)
        df['price_unchanged'] = (df['close'] == df['close'].shift(1))
        stale_runs = df['price_unchanged'].rolling(5).sum()
        if (stale_runs >= 5).any():
            logger.warning(f"{ticker}: Stale prices detected (unchanged for 5+ days)")

        # Drop temporary columns
        df = df.drop(['daily_return', 'price_unchanged'], axis=1)

        final_count = len(df)
        if final_count < initial_count:
            logger.info(f"{ticker}: Removed {initial_count - final_count} invalid rows")

        return df

    def fetch_fx_rates(self, start_date: str = "2000-01-01") -> pd.DataFrame:
        """Fetch FX rates for currency conversion"""
        fx_pairs = {
            'EURUSD=X': 'EUR',
            'GBPUSD=X': 'GBP'
        }

        fx_data = []
        for pair, currency in fx_pairs.items():
            try:
                fx = yf.download(pair, start=start_date, progress=False)
                if not fx.empty:
                    fx_df = fx[['Close']].reset_index()
                    fx_df.columns = ['date', 'fx_rate']
                    fx_df['currency'] = currency
                    fx_data.append(fx_df)
                    logger.info(f"Downloaded FX rates for {currency}")
            except Exception as e:
                logger.error(f"Failed to fetch {pair}: {e}")

        if fx_data:
            return pd.concat(fx_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def fetch_all_data(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """Main function to fetch all data"""

        cache_file = self.data_dir / "raw" / "price_data.parquet"

        # Check cache
        if use_cache and cache_file.exists():
            logger.info("Loading cached data...")
            prices_df = pd.read_parquet(cache_file)
            fx_df = pd.read_parquet(self.data_dir / "raw" / "fx_rates.parquet")
            return {'prices': prices_df, 'fx_rates': fx_df}

        # Get all tickers
        logger.info("Fetching ticker lists...")
        sp500 = self.get_sp500_tickers()
        nasdaq = self.get_nasdaq100_tickers()
        euro = self.get_euro_stoxx_tickers()

        # Combine and deduplicate
        all_tickers = list(set(sp500 + nasdaq + euro))
        logger.info(f"Total unique tickers to fetch: {len(all_tickers)}")

        # Separate by currency for better organization
        us_tickers = [t for t in all_tickers if self.identify_currency(t) == 'USD']
        euro_tickers = [t for t in all_tickers if self.identify_currency(t) == 'EUR']

        logger.info(f"USD tickers: {len(us_tickers)}, EUR tickers: {len(euro_tickers)}")

        # Fetch price data
        all_prices = []
        failed_tickers = []

        total = len(all_tickers)
        for i, ticker in enumerate(all_tickers, 1):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{total} tickers fetched")

            df = self.fetch_price_data(ticker)
            if not df.empty:
                all_prices.append(df)
            else:
                failed_tickers.append(ticker)

        logger.info(f"Successfully fetched: {len(all_prices)} tickers")
        logger.info(f"Failed tickers: {len(failed_tickers)}")

        if failed_tickers:
            logger.warning(f"Failed to fetch: {failed_tickers[:10]}...")  # Show first 10

        # Combine all price data
        if all_prices:
            prices_df = pd.concat(all_prices, ignore_index=True)
            prices_df['date'] = pd.to_datetime(prices_df['date'])
            prices_df = prices_df.sort_values(['symbol', 'date'])

            # Fetch FX rates
            logger.info("Fetching FX rates...")
            fx_df = self.fetch_fx_rates()

            # Save to cache
            prices_df.to_parquet(cache_file, compression='snappy')
            fx_df.to_parquet(self.data_dir / "raw" / "fx_rates.parquet", compression='snappy')

            # Also save as CSV for readability
            prices_df.to_csv(self.data_dir / "raw" / "prices_combined.csv", index=False)
            fx_df.to_csv(self.data_dir / "raw" / "fx_rates.csv", index=False)

            logger.info(f"Data saved to {self.data_dir / 'raw'}")

            # Print summary statistics
            self.print_data_summary(prices_df)

            return {'prices': prices_df, 'fx_rates': fx_df}
        else:
            logger.error("No data fetched!")
            return {'prices': pd.DataFrame(), 'fx_rates': pd.DataFrame()}

    def print_data_summary(self, df: pd.DataFrame):
        """Print summary statistics of fetched data"""
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        print(f"Total observations: {len(df):,}")
        print(f"Unique symbols: {df['symbol'].nunique()}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")

        # Currency breakdown
        currency_counts = df.groupby('currency')['symbol'].nunique()
        print("\nSymbols by currency:")
        for curr, count in currency_counts.items():
            print(f"  {curr}: {count}")

        # Data coverage
        coverage = df.groupby('symbol').agg({
            'date': ['min', 'max', 'count']
        })

        print(f"\nAverage observations per symbol: {coverage[('date', 'count')].mean():.0f}")
        print(f"Symbols with >1000 observations: {(coverage[('date', 'count')] > 1000).sum()}")
        print("=" * 60 + "\n")


def main():
    """Main execution function"""
    # Create data fetcher
    fetcher = DataFetcher()

    # Fetch all data
    logger.info("Starting data fetch process...")
    data = fetcher.fetch_all_data(use_cache=False)  # Set to True to use cached data

    if not data['prices'].empty:
        logger.info("Data fetch completed successfully!")

        # Create a processed version with USD-normalized prices
        logger.info("Creating USD-normalized dataset...")

        prices = data['prices'].copy()
        fx_rates = data['fx_rates'].copy()

        # Merge FX rates
        if not fx_rates.empty:
            fx_rates['date'] = pd.to_datetime(fx_rates['date'])

            # Add USD (rate = 1)
            usd_rates = pd.DataFrame({
                'date': fx_rates['date'].unique(),
                'currency': 'USD',
                'fx_rate': 1.0
            })

            fx_rates = pd.concat([fx_rates, usd_rates], ignore_index=True)

            # Merge prices with FX rates
            prices_adj = prices.merge(fx_rates, on=['date', 'currency'], how='left')

            # Forward fill FX rates for missing dates
            prices_adj['fx_rate'] = prices_adj.groupby('currency')['fx_rate'].fillna(method='ffill')

            # Create USD-adjusted prices
            prices_adj['close_usd'] = prices_adj['close'] * prices_adj['fx_rate']
            prices_adj['open_usd'] = prices_adj['open'] * prices_adj['fx_rate']
            prices_adj['high_usd'] = prices_adj['high'] * prices_adj['fx_rate']
            prices_adj['low_usd'] = prices_adj['low'] * prices_adj['fx_rate']

            # Save processed data
            output_file = fetcher.data_dir / "processed" / "prices_clean.parquet"
            prices_adj.to_parquet(output_file, compression='snappy')
            logger.info(f"Saved processed data to {output_file}")

            print("\nProcessed data preview:")
            print(prices_adj.head())
        else:
            logger.warning("No FX rates available - skipping currency adjustment")
    else:
        logger.error("Data fetch failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())