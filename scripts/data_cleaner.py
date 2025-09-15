#!/usr/bin/env python3
"""
diagnostics.py - Data diagnostics with plots and charts

Outputs plots to: <this file's folder>/diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Project scripts/data root
DATA_DIR = Path(r"C:\Desktop\BA\PY\my_project\scripts\data")

random.seed(42)


def get_price_column(df: pd.DataFrame) -> str:
    if "close_adj" in df.columns:
        return "close_adj"
    if "close" in df.columns:
        return "close"
    if "close_raw" in df.columns:
        return "close_raw"
    raise KeyError("No close price column found (expected close_adj/close/close_raw)")


# ── Coverage & Completeness ───────────────────────────────────────────────────
def plot_time_coverage(df: pd.DataFrame):
    logger.info("📈 Plotting time coverage heatmap...")
    price_col = get_price_column(df)

    pivot = df.pivot_table(index="symbol", columns="date", values=price_col, aggfunc="count").fillna(0)
    plt.figure(figsize=(18, 10))
    sns.heatmap(pivot, cmap="viridis", cbar=False)
    plt.title("Time Coverage per Symbol", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Symbol")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_coverage_heatmap.png", dpi=150)
    plt.close()


def plot_hist_series_length(df: pd.DataFrame):
    logger.info("📈 Plotting histogram of series length...")
    counts = df.groupby("symbol")["date"].count()
    plt.figure(figsize=(10, 6))
    sns.histplot(counts, bins=50, kde=False)
    plt.title("Distribution of Series Length per Symbol")
    plt.xlabel("Number of Observations")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "series_length_hist.png", dpi=150)
    plt.close()


def plot_missing_data(df: pd.DataFrame):
    logger.info("📈 Plotting missing data heatmap...")
    price_col = get_price_column(df)
    pivot = df.pivot_table(index="date", columns="symbol", values=price_col, aggfunc="count")
    plt.figure(figsize=(18, 10))
    sns.heatmap(pivot.isna(), cmap="Reds", cbar=False)
    plt.title("Missing Data Heatmap")
    plt.xlabel("Symbol")
    plt.ylabel("Date")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missing_data_heatmap.png", dpi=150)
    plt.close()


def plot_trading_calendar_alignment(df: pd.DataFrame):
    logger.info("📈 Plotting trading calendar alignment...")
    counts = df.groupby("date")["symbol"].count()
    plt.figure(figsize=(14, 6))
    counts.plot()
    plt.title("Number of Active Symbols per Trading Day")
    plt.xlabel("Date")
    plt.ylabel("Active Symbols")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trading_calendar_alignment.png", dpi=150)
    plt.close()


# ── Price & Return Validity ───────────────────────────────────────────────────
def plot_price_examples(df: pd.DataFrame, n=10):
    logger.info("📈 Plotting price series examples...")
    price_col = get_price_column(df)
    uniq = list(df["symbol"].unique())
    if not uniq:
        return
    sample_symbols = random.sample(uniq, min(n, len(uniq)))

    plt.figure(figsize=(14, 8))
    for sym in sample_symbols:
        subset = df[df["symbol"] == sym]
        plt.plot(subset["date"], subset[price_col], label=sym, alpha=0.7)
    if len(sample_symbols) <= 20:
        plt.legend()
    plt.title("Sample Price Series")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "price_examples.png", dpi=150)
    plt.close()


def plot_return_distribution(df: pd.DataFrame):
    logger.info("📈 Plotting return distribution...")
    price_col = get_price_column(df)
    df_sorted = df.sort_values(["symbol", "date"]).copy()
    df_sorted["return"] = df_sorted.groupby("symbol")[price_col].pct_change()

    plt.figure(figsize=(10, 6))
    sns.histplot(df_sorted["return"].dropna(), bins=200, kde=True)
    plt.title("Distribution of Daily Returns")
    plt.xlabel("Daily Return")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "return_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df_sorted["return"].dropna())
    plt.title("Boxplot of Daily Returns")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "return_boxplot.png", dpi=150)
    plt.close()


def plot_cumulative_returns(df: pd.DataFrame, n=5):
    logger.info("📈 Plotting cumulative log returns...")
    price_col = get_price_column(df)
    uniq = list(df["symbol"].unique())
    if not uniq:
        return
    sample_symbols = random.sample(uniq, min(n, len(uniq)))

    plt.figure(figsize=(14, 8))
    for sym in sample_symbols:
        subset = df[df["symbol"] == sym].sort_values("date")
        r = subset[price_col].pct_change().dropna()
        if r.empty:
            continue
        cum_log_ret = np.log1p(r).cumsum()
        plt.plot(subset["date"].iloc[1:], cum_log_ret, label=sym)
    if len(sample_symbols) <= 20:
        plt.legend()
    plt.title("Cumulative Log Returns (Sample)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Log Return")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cumulative_returns.png", dpi=150)
    plt.close()


# ── Outlier Detection ─────────────────────────────────────────────────────────
def plot_top_return_spikes(df: pd.DataFrame):
    logger.info("📈 Plotting top return spikes...")
    price_col = get_price_column(df)
    df_sorted = df.sort_values(["symbol", "date"]).copy()
    df_sorted["return"] = df_sorted.groupby("symbol")[price_col].pct_change()
    abs_returns = df_sorted[["symbol", "date", "return"]].dropna().copy()
    if abs_returns.empty:
        return
    abs_returns["abs_ret"] = abs_returns["return"].abs()
    top = abs_returns.nlargest(20, "abs_ret")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top, x="date", y="abs_ret", hue="symbol", dodge=False)
    plt.xticks(rotation=45)
    plt.title("Top 20 One-Day Return Spikes")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_return_spikes.png", dpi=150)
    plt.close()


def plot_volume_anomalies(df: pd.DataFrame, n=5):
    logger.info("📈 Plotting volume anomalies...")
    if "volume" not in df.columns:
        return
    uniq = list(df["symbol"].unique())
    if not uniq:
        return
    sample_symbols = random.sample(uniq, min(n, len(uniq)))

    plt.figure(figsize=(14, 8))
    for sym in sample_symbols:
        subset = df[df["symbol"] == sym]
        plt.plot(subset["date"], subset["volume"], label=sym, alpha=0.7)
    if len(sample_symbols) <= 20:
        plt.legend()
    plt.title("Trading Volume (Sample)")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "volume_anomalies.png", dpi=150)
    plt.close()


# ── Currency / Market Effects ─────────────────────────────────────────────────
def plot_currency_split(df: pd.DataFrame):
    logger.info("📈 Plotting currency split...")
    if "currency" in df.columns:
        plt.figure(figsize=(6, 6))
        df["currency"].value_counts().plot.pie(autopct="%1.1f%%")
        plt.title("Currency Split of Symbols")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "currency_split.png", dpi=150)
        plt.close()


# ── Main Runner ───────────────────────────────────────────────────────────────
def run_all_diagnostics():
    logger.info("🔍 Running all diagnostics...")

    clean_file = DATA_DIR / "processed" / "prices_clean.parquet"
    if not clean_file.exists():
        logger.error(f"❌ Cleaned data not found at: {clean_file}")
        return

    df = pd.read_parquet(clean_file)

    # ── Export cleaned ML data ────────────────────────────────────────────────
    ml_outdir = DATA_DIR / "cleaned"
    ml_outdir.mkdir(parents=True, exist_ok=True)
    ml_file = ml_outdir / "ML_data_cleaned.parquet"

    wanted_cols = ["date", "symbol", "open_raw", "high_raw", "low_raw",
                   "close_raw", "close_adj", "volume"]
    existing_cols = [c for c in wanted_cols if c in df.columns]

    if not existing_cols:
        logger.error("❌ No OHLCV columns found in cleaned data.")
    else:
        ohlcv = df[existing_cols].copy()
        ohlcv.to_parquet(ml_file, compression="snappy")
        logger.info(f"💾 Cleaned ML dataset saved to {ml_file} with columns {existing_cols}")

    # ── Run diagnostics plots ─────────────────────────────────────────────────
    plot_time_coverage(df)
    plot_hist_series_length(df)
    plot_missing_data(df)
    plot_trading_calendar_alignment(df)
    plot_price_examples(df)
    plot_return_distribution(df)
    plot_cumulative_returns(df)
    plot_top_return_spikes(df)
    plot_volume_anomalies(df)
    plot_currency_split(df)

    logger.info(f"✅ Diagnostics complete. Plots saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    run_all_diagnostics()
