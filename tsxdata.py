import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

def to_yahoo(tsx_code: str) -> str:
    if not isinstance(tsx_code, str):
        return tsx_code
    s = tsx_code.strip().upper()
    if "-" in s:
        base, suffix = s.rsplit("-", 1)
    else:
        base, suffix = s, ""
    base = base.replace(".", "-")
    out = base + (f"-{suffix}" if suffix else "")
    out = out.replace("-T", ".TO").replace("-V", ".V")
    return out

def compute_mom_12_2(group: pd.DataFrame) -> pd.Series:
    lr = group["logret"]
    mom_log = lr.shift(1).rolling(window=11, min_periods=11).sum()
    return np.expm1(mom_log)

def load_rf_monthly_from_daily(rf_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(rf_csv)
    if "Date" not in df.columns or "rf" not in df.columns:
        raise ValueError("RF CSV must have columns: 'Date' and 'rf'.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df["rf"] = pd.to_numeric(df["rf"], errors="coerce")  # 'Bank holiday' -> NaN
    if df["rf"].dropna().quantile(0.9) > 1.0:
        df["rf"] = df["rf"] / 100.0  # percent -> decimal
    df["month_end"] = df["Date"].dt.to_period("M").dt.to_timestamp("M")
    monthly_ann = df.groupby("month_end")["rf"].mean()
    rf_month = ((1.0 + monthly_ann) ** (1.0 / 12.0) - 1.0).to_frame("rf_month")
    out = rf_month.reset_index().rename(columns={"month_end": "date"})
    return out[["date", "rf_month"]]

def main(args):
    cons = pd.read_csv(args.constituents)
    ticker_col = args.ticker_col if args.ticker_col in cons.columns else cons.columns[0]
    cons["yahoo_ticker"] = cons[ticker_col].apply(to_yahoo)
    tickers = (
        cons["yahoo_ticker"].dropna().astype(str).str.strip().replace({"": np.nan}).dropna().unique().tolist()
    )
    if args.max_tickers is not None:
        tickers = tickers[:args.max_tickers]
    if not tickers:
        raise ValueError("No tickers found after cleaning.")

    print(f"[INFO] Downloading {len(tickers)} tickers from Yahoo Finance...")
    px = yf.download(
        tickers=tickers,
        start=args.start,
        end=args.end,
        auto_adjust=False,
        actions=True,
        group_by="ticker",
        interval="1d",
        threads=True,
        progress=True,
    )

    def get_adj_close(df_tkr: pd.DataFrame) -> pd.Series:
        if "Adj Close" in df_tkr.columns and not df_tkr["Adj Close"].isna().all():
            return df_tkr["Adj Close"].rename("adj_close")
        return df_tkr["Close"].rename("adj_close")

    frames = []
    for t in tickers:
        try:
            df_t = px[t].copy()
        except Exception:
            if isinstance(px, pd.DataFrame) and not px.empty:
                df_t = px.copy()
            else:
                continue
        if df_t.empty:
            continue
        adj = get_adj_close(df_t).dropna()
        adj_m = adj.resample("M").last().to_frame()
        adj_m["ret"] = adj_m["adj_close"].pct_change()
        tmp = adj_m.reset_index()
        tmp = tmp.rename(columns={tmp.columns[0]: "date"})
        tmp["ticker"] = t
        frames.append(tmp)

    if not frames:
        raise RuntimeError("No price data collected. Check tickers and dates.")
    panel = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    panel["date"] = pd.to_datetime(panel["date"])

    panel["logret"] = np.log1p(panel["ret"])
    panel["mom_12_2"] = panel.groupby("ticker", group_keys=False).apply(compute_mom_12_2)
    panel["ret_lead"] = panel.groupby("ticker")["ret"].shift(-1)

    rf_monthly = load_rf_monthly_from_daily(args.rf_csv)
    panel = panel.merge(rf_monthly, on="date", how="left")
    panel["rf_month"] = panel["rf_month"].fillna(method="ffill")
    panel["excess_return_lead"] = panel["ret_lead"] - panel["rf_month"]

    panel["month"] = panel["date"].dt.to_period("M").astype(str)

    def _z(s):
        std = s.std(ddof=0)
        return (s - s.mean()) / std if std and np.isfinite(std) and std > 0 else np.nan
    panel["mom_z"] = panel.groupby("month")["mom_12_2"].transform(_z)

    out = (
        panel.dropna(subset=["mom_12_2", "ret_lead", "rf_month"])
        .loc[:, ["ticker", "date", "month", "ret", "ret_lead", "rf_month",
                 "excess_return_lead", "mom_12_2", "mom_z"]]
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[DONE] Wrote {len(out):,} rows for {out['ticker'].nunique()} tickers -> {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a Canada momentum panel with monthly RF from daily RF CSV.")
    ap.add_argument("--constituents", type=Path, required=False,
        default=Path(r"C:\Users\rfang\Documents\ECO375-Canada-Momentum-and-Value\tsx300.csv"),
        help="Path to TSX constituents CSV.")
    ap.add_argument("--ticker-col", type=str, default="Ticker",
        help="Column name containing tickers in the constituents CSV (default: 'Ticker').")
    ap.add_argument("--rf-csv", type=Path, required=False,
        default=Path(r"C:\Users\rfang\Documents\ECO375-Canada-Momentum-and-Value\canadariskfreedaily.csv"),
        help="Path to DAILY risk-free CSV with columns 'Date' and 'rf' (percent annualized).")
    ap.add_argument("--out", type=Path, default=Path("panel_momentum_ca.csv"),
        help="Output CSV path (default: panel_momentum_ca.csv).")
    ap.add_argument("--start", type=str, default="2015-10-16",
        help="Start date for Yahoo data (default: 2015-10-16).")
    ap.add_argument("--end", type=str, default=None,
        help="End date (default: None = today).")
    ap.add_argument("--max-tickers", type=int, default=120,
        help="Limit tickers for a quick prototype (default: 120; set None for all).")
    args = ap.parse_args()
    main(args)
