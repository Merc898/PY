#!/usr/bin/env python3
"""
fetch_all_prices.py - Robust price data fetching with batch downloads
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
from typing import List
from pathlib import Path
import math
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)

    def get_sp500_tickers(self) -> List[str]:
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                attrs={"class": "wikitable"},
                flavor="bs4",
                storage_options={"User-Agent": "Mozilla/5.0"}
            )
            df = tables[0]
            return df["Symbol"].str.replace(".", "-").tolist()
        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 tickers: {e}")
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    def get_nasdaq100_tickers(self) -> List[str]:
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/NASDAQ-100",
                attrs={"class": "wikitable"},
                flavor="bs4",
                storage_options={"User-Agent": "Mozilla/5.0"}
            )
            for tbl in tables:
                for col in ["Ticker", "Symbol", "Company", "Company name"]:
                    if col in tbl.columns:
                        tickers = tbl[col].astype(str).str.extract(r"([A-Z\.\-]+)")[0]
                        tickers = tickers.dropna().str.replace(".", "-").tolist()
                        return tickers
            raise ValueError("No ticker column found in NASDAQ-100 tables")
        except Exception as e:
            logger.error(f"Failed to fetch NASDAQ-100 tickers: {e}")
            return []

    def get_euro_stoxx_tickers(self) -> List[str]:
        return [
            "AIR.PA", "ALV.DE", "ASML.AS", "BAS.DE", "BBVA.MC",
            "BNP.PA", "CRH.L", "CS.PA", "DTE.DE", "DB1.DE",
            "DBK.DE", "ENEL.MI", "ENGI.PA", "ENI.MI", "IBE.MC",
            "INGA.AS", "ITX.MC", "MC.PA", "MUV2.DE", "OR.PA",
            "PHIA.AS", "SAF.PA", "SAN.MC", "SAP.DE", "SIE.DE",
            "SU.PA", "TTE.PA", "UNA.AS", "VIV.PA", "VOW3.DE"
        ]

    def clean_price_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if df.empty:
            return df
        initial = len(df)
        df = df[df["Adj Close"].notna()]
        df = df[df["Adj Close"] > 0]
        removed = initial - len(df)
        if removed > 50:
            logger.info(f"{ticker}: Removed {removed} invalid rows")
        return df

    def fetch_batch(self, tickers: List[str], start_date: str) -> pd.DataFrame:
        """Fetch a batch of tickers via yf.download"""
        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=datetime.now(),
                group_by="ticker",
                auto_adjust=False,
                threads=True
            )
            all_prices = []
            for t in tickers:
                if t not in data:  # some tickers not returned
                    continue
                df = data[t].reset_index()
                df["symbol"] = t
                df = self.clean_price_data(df, t)
                if not df.empty:
                    all_prices.append(df)
            return pd.concat(all_prices, ignore_index=True) if all_prices else pd.DataFrame()
        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            return pd.DataFrame()

    def fetch_all_data(self, start_date: str = "1950-01-01", use_cache: bool = True) -> pd.DataFrame:
        cache_file = self.data_dir / "raw" / "prices.parquet"
        if use_cache and cache_file.exists():
            logger.info("Loading cached data...")
            return pd.read_parquet(cache_file)

        logger.info("Fetching ticker lists...")
        tickers = list(set(
            self.get_sp500_tickers() +
            self.get_nasdaq100_tickers() +
            self.get_euro_stoxx_tickers()
        ))
        logger.info(f"Total unique tickers: {len(tickers)}")

        batch_size = 100
        n_batches = math.ceil(len(tickers) / batch_size)
        all_prices = []
        failed = []

        for i in range(n_batches):
            batch = tickers[i*batch_size:(i+1)*batch_size]
            logger.info(f"Fetching batch {i+1}/{n_batches} ({len(batch)} tickers)")
            df = self.fetch_batch(batch, start_date)
            if df.empty:
                failed.extend(batch)
            else:
                all_prices.append(df)
            time.sleep(1)  # pause to avoid throttling

        # Retry failed tickers individually
        if failed:
            logger.info(f"Retrying {len(failed)} failed tickers individually...")
            for t in failed:
                try:
                    df = yf.download(t, start=start_date, end=datetime.now(), auto_adjust=False)
                    df = df.reset_index()
                    df["symbol"] = t
                    df = self.clean_price_data(df, t)
                    if not df.empty:
                        all_prices.append(df)
                except Exception as e:
                    logger.error(f"Failed {t}: {e}")

        if not all_prices:
            logger.error("No data fetched.")
            return pd.DataFrame()

        prices = pd.concat(all_prices, ignore_index=True)
        prices["Date"] = pd.to_datetime(prices["Date"], utc=True).dt.tz_localize(None)
        prices = prices.rename(columns={
            "Date": "date",
            "Open": "open_raw",
            "High": "high_raw",
            "Low": "low_raw",
            "Close": "close_raw",
            "Adj Close": "close_adj",
            "Volume": "volume"
        })
        prices = prices.sort_values(["symbol", "date"])

        prices.to_parquet(cache_file, compression="snappy")
        prices.to_csv(self.data_dir / "raw" / "prices.csv", index=False)
        logger.info(f"Saved data to {self.data_dir / 'raw'}")

        self.print_summary(prices)
        return prices

    def print_summary(self, df: pd.DataFrame):
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        print(f"Total observations: {len(df):,}")
        print(f"Unique symbols: {df['symbol'].nunique()}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        coverage = df.groupby("symbol").agg({"date": ["min", "max", "count"]})
        print(f"\nAverage obs per symbol: {coverage[('date', 'count')].mean():.0f}")
        print(f"Symbols with >1000 obs: {(coverage[('date', 'count')] > 1000).sum()}")
        print("=" * 60 + "\n")


def main():
    fetcher = DataFetcher()
    logger.info("Starting data fetch process...")
    data = fetcher.fetch_all_data(use_cache=False)
    if data.empty:
        return 1
    logger.info("Data fetch completed.")
    return 0


if __name__ == "__main__":
    exit(main())
