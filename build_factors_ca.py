import os
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

PANEL_CSV = "panel_momentum_ca.csv"      # from your tsxdata.py
OUTPUT_MERGED_CSV = "panel_mom_val_ca.csv"
OUTPUT_FACTORS_CSV = "factors_ca.csv"

DATE_COL = "date"
TICKER_COL = "yahoo_ticker"

# Number of portfolios for value / momentum
N_BUCKETS = 10  # deciles


# ---------------------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------------------

def winsorize_series(s: pd.Series,
                     lower_q: float = 0.01,
                     upper_q: float = 0.99) -> pd.Series:
    """
    Two-sided winsorization: clamp values below lower_q to lower_q
    and above upper_q to upper_q.
    """
    if s.dropna().empty:
        return s

    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lo, hi)


def zscore_series(s: pd.Series) -> pd.Series:
    """
    Standardize to mean 0, std 1. If std ~ 0, return NaNs.
    """
    mu = s.mean()
    sigma = s.std(ddof=0)
    if np.isclose(sigma, 0) or np.isnan(sigma):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sigma


def robust_qcut(x: pd.Series, q: int) -> pd.Series:
    """
    Wrapper around pd.qcut that handles:
    - too few unique values
    - constant series
    Returns NaNs if it can't form q buckets.
    """
    try:
        # Use rank to break ties; labels 1..q
        return pd.qcut(x.rank(method="first"), q, labels=False) + 1
    except Exception:
        return pd.Series(np.nan, index=x.index)


# ---------------------------------------------------------------------
# LOAD PANEL
# ---------------------------------------------------------------------

def load_panel(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Panel file not found: {path}")

    panel = pd.read_csv(path)
    panel[DATE_COL] = pd.to_datetime(panel[DATE_COL])

    required_cols = {DATE_COL, TICKER_COL, "adj_close", "ret", "mom_12_2"}
    missing = required_cols - set(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing required columns: {missing}")

    panel = panel.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)

    print(f"[INFO] Loaded panel: {panel.shape[0]:,} rows, "
          f"{panel[TICKER_COL].nunique()} tickers, "
          f"from {panel[DATE_COL].min().date()} to {panel[DATE_COL].max().date()}")

    return panel


# ---------------------------------------------------------------------
# DIVIDEND YIELD AS VALUE SIGNAL
# ---------------------------------------------------------------------

def fetch_dividends_for_ticker(ticker: str,
                               start: pd.Timestamp,
                               end: pd.Timestamp) -> pd.Series:
    """
    Fetch dividend history for a ticker from Yahoo between start and end.
    Returns a Series indexed by date with dividend amounts.
    """
    try:
        tk = yf.Ticker(ticker)
        div = tk.dividends  # Series: index=DatetimeIndex, values=div amount
    except Exception as e:
        print(f"[WARN] Could not fetch dividends for {ticker}: {e}")
        return pd.Series(dtype=float)

    if div is None or div.empty:
        return pd.Series(dtype=float)

    # Normalize timezone to tz-naive to avoid tz-aware/naive comparison issues
    # seen from yfinance (e.g., America/Toronto). Also ensure it's a DatetimeIndex.
    if not isinstance(div.index, pd.DatetimeIndex):
        div.index = pd.to_datetime(div.index)
    if getattr(div.index, "tz", None) is not None:
        # remove timezone information
        div.index = div.index.tz_localize(None)

    # Now we can safely filter by start/end (which are tz-naive)
    div = div[(div.index >= start) & (div.index <= end)]
    return div


def compute_trailing_12m_dy_for_ticker(
    panel_sub: pd.DataFrame
) -> pd.Series:
    """
    For a single ticker's panel (monthly rows), compute trailing 12-month
    dividend yield at each panel date:

        DY_t = (sum of dividends over [t-12m, t]) / price_t

    Returns a Series aligned to panel_sub.index.
    """
    # panel_sub is sorted by DATE already
    dates = panel_sub[DATE_COL].values
    first_date = panel_sub[DATE_COL].min()
    last_date = panel_sub[DATE_COL].max()

    # We need dividends going at least 12 months before the first panel date
    div_start = first_date - pd.DateOffset(months=13)
    div_end = last_date + pd.DateOffset(days=7)

    ticker = panel_sub[TICKER_COL].iloc[0]

    div = fetch_dividends_for_ticker(ticker, div_start, div_end)
    if div.empty:
        # No dividends -> all NaN yields
        return pd.Series(np.nan, index=panel_sub.index)

    # Convert dividends to month-end frequency (sum within calendar month)
    # 'M' is deprecated; use 'ME' (month-end)
    div_m = div.resample("ME").sum().sort_index()

    # Build a monthly trailing 12M sum of dividends
    div12 = div_m.rolling(window=12, min_periods=1).sum()

    # For each panel date (usually month-start), map to that month's month-end
    month_end_idx = (panel_sub[DATE_COL] + pd.offsets.MonthEnd(0))

    # Align the trailing 12M sum to those month-ends (ffill in case we don't
    # have an exact month in div12, but we do monthly resample so should line up)
    div12_aligned = div12.reindex(month_end_idx, method="ffill")

    # Now DY_t = trailing 12M div / price_t (adj_close)
    price = panel_sub["adj_close"].values
    dy_values = div12_aligned.values / price

    dy_series = pd.Series(dy_values, index=panel_sub.index)
    return dy_series


def add_dividend_yield(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each ticker in the panel, compute dividend yield and add column 'dy'.
    """
    panel = panel.copy()

    dy_all = []

    for t, grp in panel.groupby(TICKER_COL):
        grp = grp.sort_values(DATE_COL).copy()
        dy = compute_trailing_12m_dy_for_ticker(grp)
        dy_all.append(dy)

        print(f"[INFO] DY computed for {t}: "
              f"{dy.notna().sum()} non-NaN months, "
              f"{len(dy) - dy.notna().sum()} NaN months")

    panel["dy"] = pd.concat(dy_all).sort_index()
    return panel


# ---------------------------------------------------------------------
# BUILD VALUE SCORE FROM DIVIDEND YIELD
# ---------------------------------------------------------------------

def build_value_score_from_dy(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each DATE:
      - winsorize dividend yield cross-sectionally
      - z-score it (higher z_dy = higher yield = cheaper)
      - define val_score = z_dy

    This is an Asness-style cross-sectional standardization but using only
    dividend yield as the value metric.
    """
    panel = panel.copy()

    def per_date(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df["dy"].notna().sum() < 5:
            df["val_score"] = np.nan
            return df

        w = winsorize_series(df["dy"])
        z = zscore_series(w)
        df["val_score"] = z
        return df

    panel = panel.groupby(DATE_COL, group_keys=False).apply(per_date)
    return panel


# ---------------------------------------------------------------------
# ASSIGN VALUE & MOMENTUM BUCKETS
# ---------------------------------------------------------------------

def assign_buckets(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each date, assign:
      - value_bucket: 1..N_BUCKETS using val_score
          (1 = expensive / low DY, N = cheap / high DY)
      - mom_bucket: 1..N_BUCKETS using mom_12_2
          (1 = losers, N = winners)
    """
    panel = panel.copy()

    def per_date(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Value buckets
        if df["val_score"].notna().sum() > N_BUCKETS:
            df["value_bucket"] = robust_qcut(df["val_score"], N_BUCKETS)
        else:
            df["value_bucket"] = np.nan

        # Momentum buckets
        if df["mom_12_2"].notna().sum() > N_BUCKETS:
            df["mom_bucket"] = robust_qcut(df["mom_12_2"], N_BUCKETS)
        else:
            df["mom_bucket"] = np.nan

        return df

    panel = panel.groupby(DATE_COL, group_keys=False).apply(per_date)
    return panel


# ---------------------------------------------------------------------
# COMPUTE PORTFOLIO & FACTOR RETURNS
# ---------------------------------------------------------------------

def compute_portfolio_returns(panel: pd.DataFrame
                              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From a panel with:
       date, ticker, ret, value_bucket, mom_bucket
    compute:
       - value decile returns each month (equal-weighted)
       - momentum decile returns each month
    Then construct long-short (top - bottom) factors.

    Returns:
      value_ports: DataFrame with columns date, val_dec_1, ..., val_dec_N, val_lh
      mom_ports:   DataFrame with columns date, mom_dec_1, ..., mom_dec_N, mom_lh
    """
    df = panel.copy()

    # We want return at time t+1 linked to buckets formed at time t.
    # So form buckets at t, then use ret_{t+1}.
    df = df.sort_values([TICKER_COL, DATE_COL])
    df["ret_fwd"] = df.groupby(TICKER_COL)["ret"].shift(-1)
    df = df.dropna(subset=["ret_fwd"])

    def make_ports(bucket_col: str, prefix: str) -> pd.DataFrame:
        valid = df[~df[bucket_col].isna()].copy()
        if valid.empty:
            raise ValueError(f"No valid data for bucket column {bucket_col}")

        valid[bucket_col] = valid[bucket_col].astype(int)

        grouped = (
            valid
            .groupby([DATE_COL, bucket_col])["ret_fwd"]
            .mean()
            .unstack(bucket_col)
        )

        # rename cols
        grouped.columns = [f"{prefix}_dec_{i}" for i in grouped.columns]

        # long-high minus long-low
        grouped[f"{prefix}_lh"] = grouped[f"{prefix}_dec_{N_BUCKETS}"] - grouped[f"{prefix}_dec_1"]

        return grouped.reset_index()

    value_ports = make_ports("value_bucket", "val")
    mom_ports = make_ports("mom_bucket", "mom")

    return value_ports, mom_ports


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("[STEP] Loading panel...")
    panel = load_panel(PANEL_CSV)

    print("[STEP] Computing dividend yields (value signal) from Yahoo...")
    panel = add_dividend_yield(panel)

    print("[STEP] Building cross-sectional value score from DY...")
    panel = build_value_score_from_dy(panel)

    print("[STEP] Assigning value & momentum decile buckets...")
    panel = assign_buckets(panel)

    print(f"[STEP] Saving merged panel with value info to {OUTPUT_MERGED_CSV} ...")
    panel.to_csv(OUTPUT_MERGED_CSV, index=False)

    print("[STEP] Computing value & momentum portfolio returns...")
    value_ports, mom_ports = compute_portfolio_returns(panel)

    print("[STEP] Combining factor returns into a single file...")
    factors = value_ports.merge(mom_ports, on=DATE_COL, how="outer").sort_values(DATE_COL)
    factors.to_csv(OUTPUT_FACTORS_CSV, index=False)

    print(f"[DONE] Wrote:")
    print(f"  - panel with value signal & buckets: {OUTPUT_MERGED_CSV} "
          f"({len(panel):,} rows)")
    print(f"  - factor returns (value & momentum): {OUTPUT_FACTORS_CSV} "
          f"({len(factors):,} months)")


if __name__ == "__main__":
    main()
