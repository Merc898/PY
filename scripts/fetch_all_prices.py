#!/usr/bin/env python3
"""
diagnostics.py - Comprehensive diagnostics for fetched price data
Outputs:
  data/processed/diagnostic_report.txt
  data/processed/diagnostics_summary.csv
  data/processed/diagnostics_symbol_summary.csv
  data/processed/diagnostics_top_moves.csv
  data/processed/figures/*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from datetime import datetime

# logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

REPORT_DIR = Path("data/processed")
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


class Diagnostics:
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.report_txt = (REPORT_DIR / "diagnostic_report.txt").resolve()
        self.summary_csv = (REPORT_DIR / "diagnostics_summary.csv").resolve()
        self.symbol_csv = (REPORT_DIR / "diagnostics_symbol_summary.csv").resolve()
        self.top_moves_csv = (REPORT_DIR / "diagnostics_top_moves.csv").resolve()

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Using data file: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        df.columns = [c.lower() for c in df.columns]
        # check columns
        need = {"date", "symbol", "close", "volume"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        df["date"] = pd.to_datetime(df["date"])
        return df

    def run_all(self) -> dict:
        df = self.load_data()
        res = {}
        res["overview"] = self.data_overview(df)
        res["quality"] = self.data_quality(df)
        res["corporate"] = self.corporate_actions(df)
        res["currency"] = self.currency_check(df)
        res["coverage"] = self.coverage_analysis(df)
        res["returns"] = self.return_analysis(df)
        res["outliers"] = self.outlier_detection(df)
        res["survivorship"] = self.survivorship_bias(df)
        self.generate_report(res)
        self.export_csvs(df, res)
        self.export_charts(df)
        return res

    # sections
    def data_overview(self, df):
        logger.info("\n📊 DATA OVERVIEW")
        return {
            "total_obs": int(len(df)),
            "symbols": int(df["symbol"].nunique()),
            "date_min": df["date"].min(),
            "date_max": df["date"].max(),
        }

    def data_quality(self, df):
        logger.info("\n🔍 DATA QUALITY CHECK")
        # missing
        missing_counts = df.isna().sum().to_dict()
        # stale prices
        stale = (df.groupby("symbol")["close"]
                   .apply(lambda s: (s == s.shift()).rolling(5).sum().ge(5).any())
                   .sum())
        # zero volumes
        zero_vol = int((df["volume"] == 0).sum())
        # duplicates
        dupes = int(df.duplicated().sum())
        return {
            "stale_price_symbols": int(stale),
            "zero_volume_rows": zero_vol,
            "duplicate_rows": dupes,
            "missing_counts": missing_counts
        }

    def corporate_actions(self, df):
        logger.info("\n🔄 CORPORATE ACTIONS DETECTION")
        df = df.sort_values(["symbol", "date"]).copy()
        df["ret"] = df.groupby("symbol")["close"].pct_change()
        big = df.loc[df["ret"].abs() > 0.70, ["symbol", "date", "ret"]]
        sample = big.reindex(big["ret"].abs().nlargest(10).index)
        if not sample.empty:
            sample.sort_values("date").to_csv(self.top_moves_csv, index=False)
        return {
            "events_gt70pct": int(len(big)),
            "affected_symbols": int(big["symbol"].nunique()),
        }

    def currency_check(self, df):
        logger.info("\n💱 CURRENCY CONSISTENCY CHECK")
        if "currency" in df.columns:
            counts = df.groupby("currency")["symbol"].nunique().to_dict()
        else:
            counts = {"UNKNOWN": int(df["symbol"].nunique())}
        return {"currencies": counts}

    def coverage_analysis(self, df):
        logger.info("\n📈 COVERAGE ANALYSIS")
        cov = df.groupby("symbol")["date"].agg(["min", "max", "count"])
        avg_obs = float(cov["count"].mean())
        poor = int((cov["count"] < 0.8 * cov["count"].max()).sum())
        cov.to_csv(self.symbol_csv)  # full per-symbol summary
        return {
            "avg_obs_per_symbol": avg_obs,
            "symbols_lt80pct_of_max": poor,
        }

    def return_analysis(self, df):
        logger.info("\n📊 RETURN ANALYSIS")
        df = df.sort_values(["symbol", "date"]).copy()
        r = df.groupby("symbol")["close"].pct_change().dropna()
        clipped = r[(r > -0.90) & (r < 1.00)]
        return {
            "mean": float(clipped.mean()),
            "std": float(clipped.std()),
            "skew": float(clipped.skew()),
            "kurtosis": float(clipped.kurtosis()),
            "p1": float(clipped.quantile(0.01)),
            "p99": float(clipped.quantile(0.99)),
            "gt_100pct_moves": int((r > 1.0).sum()),
            "lt_-90pct_moves": int((r < -0.9).sum()),
        }

    def outlier_detection(self, df):
        logger.info("\n🎯 OUTLIER DETECTION")
        price_outlier_syms = (df.groupby("symbol")["close"]
                                .apply(lambda s: (s.pct_change().abs() > 0.5).any())
                                .sum())
        vol_outliers = int((df["volume"] > df["volume"].quantile(0.999)).sum())
        return {
            "price_outlier_symbols": int(price_outlier_syms),
            "volume_outlier_rows": vol_outliers,
        }

    def survivorship_bias(self, df):
        logger.info("\n🔎 SURVIVORSHIP BIAS CHECK")
        latest = df["date"].max()
        active_max = df.groupby("symbol")["date"].max()
        survival_rate = float((active_max >= latest - pd.Timedelta(days=30)).mean())
        return {"survival_rate": survival_rate}

    # outputs
    def generate_report(self, res: dict):
        logger.info("\n📝 Generating diagnostic report...")
        with open(self.report_txt, "w", encoding="utf-8") as f:
            f.write("============================================================\n")
            f.write("DIAGNOSTIC REPORT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("============================================================\n\n")
            for section, content in res.items():
                f.write(f"## {section.upper()}\n")
                for k, v in content.items():
                    f.write(f"• {k}: {v}\n")
                f.write("\n")
        logger.info(f"Report written to {self.report_txt}")

    def export_csvs(self, df, res):
        rows = []
        for sec, content in res.items():
            for k, v in content.items():
                rows.append({"section": sec, "metric": k, "value": v})
        pd.DataFrame(rows).to_csv(self.summary_csv, index=False)
        logger.info(f"Summary CSV -> {self.summary_csv}")

    def export_charts(self, df):
        logger.info("📈 Exporting charts...")
        # active symbols per day
        active_per_day = df.groupby("date")["symbol"].nunique()
        plt.figure(figsize=(10,5))
        active_per_day.plot()
        plt.title("Active Symbols per Day")
        plt.ylabel("Symbols")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "active_symbols.png")
        plt.close()

        # return histogram
        r = df.groupby("symbol")["close"].pct_change().dropna()
        plt.figure(figsize=(8,5))
        r.hist(bins=100, range=(-0.2,0.2))
        plt.title("Distribution of Daily Returns")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "returns_hist.png")
        plt.close()

        # volume distribution
        plt.figure(figsize=(8,5))
        df["volume"].clip(upper=df["volume"].quantile(0.99)).hist(bins=100)
        plt.title("Distribution of Trading Volume (clipped at 99th pct)")
        plt.xlabel("Volume")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "volume_hist.png")
        plt.close()


def main():
    data_path = Path("data/processed/prices_clean.parquet")
    diag = Diagnostics(data_path)
    diag.run_all()
    return 0


if __name__ == "__main__":
    exit(main())
