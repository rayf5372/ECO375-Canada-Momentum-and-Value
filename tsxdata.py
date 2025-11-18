import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

INPUT_CSV = "tsx300.csv"            # your Barchart file
OUTPUT_CSV = "panel_momentum_ca.csv"

START_DATE = "2010-01-01"           # start of sample
END_DATE = None                     # None = up to today
INTERVAL = "1mo"                    # monthly data
BATCH_SIZE = 80                     # yfinance batch size

# ---------------------------------------------------------------------
# STEP 1: Read & clean constituents
# ---------------------------------------------------------------------


def detect_ticker_column(df: pd.DataFrame) -> str:
    """Guess the ticker column name, e.g. Symbol / Ticker / symbol."""
    candidates = ["Symbol", "Ticker", "symbol", "ticker"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a ticker column among: {candidates}")


def clean_constituents(path: str) -> pd.DataFrame:
    """Load Barchart CSV and return a clean constituents DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    cons = pd.read_csv(path)

    # Normalise column names
    cons.columns = [c.strip() for c in cons.columns]

    ticker_col = detect_ticker_column(cons)

    print(f"[DEBUG] raw cons shape: {cons.shape}")
    print(f"[DEBUG] ticker_col detected: {ticker_col}")

    # Drop NA symbols
    cons = cons[cons[ticker_col].notna()]

    # Drop rows that are actually repeated header lines (Symbol,Name,...)
    cons = cons[cons[ticker_col] != ticker_col]

    # Drop obvious garbage footer / note rows if present
    bad_mask = cons[ticker_col].astype(str).str.contains(
        "DOWNLOADED DATA PROVIDED BY BARCHART", case=False, na=False
    )
    cons = cons[~bad_mask]

    # Clean up whitespace and build raw_ticker
    cons["raw_ticker"] = cons[ticker_col].astype(str).str.strip()

    # Deduplicate on raw ticker
    cons = cons.drop_duplicates(subset=["raw_ticker"]).reset_index(drop=True)

    print(f"[DEBUG] cons after cleaning shape: {cons.shape}")
    print(f"[DEBUG] unique raw tickers: {cons['raw_ticker'].nunique()}")

    return cons


# ---------------------------------------------------------------------
# STEP 2: Map Barchart tickers -> Yahoo tickers
# ---------------------------------------------------------------------


def to_yahoo_ticker(sym: str) -> str:
    """
    Convert Barchart TSX ticker format to Yahoo format.

    Examples:
        AAV-T      -> AAV.TO
        BBD-B-T    -> BBD-B.TO
        AP-UN-T    -> AP-UN.TO
        Already .TO stays as-is.
    """
    s = sym.strip()

    # If it's already a Yahoo-style TSX ticker, keep it
    if s.endswith(".TO"):
        return s

    # Standard TSX tickers in this file end with "-T"
    if s.endswith("-T"):
        return s[:-2] + ".TO"

    # Fallback: leave unchanged (for debugging)
    return s


def add_yahoo_tickers(cons: pd.DataFrame) -> pd.DataFrame:
    cons = cons.copy()
    cons["yahoo_ticker"] = cons["raw_ticker"].apply(to_yahoo_ticker)

    n_raw = cons["raw_ticker"].nunique()
    n_yahoo = cons["yahoo_ticker"].nunique()

    print(f"[DEBUG] unique raw_ticker: {n_raw}")
    print(f"[DEBUG] unique yahoo_ticker: {n_yahoo}")
    print("[DEBUG] first 30 raw -> yahoo:")
    print(cons[["raw_ticker", "yahoo_ticker"]].head(30))

    return cons


# ---------------------------------------------------------------------
# STEP 3: Download prices from Yahoo in batches
# ---------------------------------------------------------------------


def download_batch(tickers, start, end, interval) -> pd.DataFrame:
    """
    Download one batch of tickers with yfinance and return a tidy DataFrame:
    columns = date, yahoo_ticker, adj_close

    Handles both:
    - Multi-ticker MultiIndex columns (Price level, Ticker level)
    - Single-ticker flat columns
    """
    tickers = list(tickers)
    print(f"[INFO] Downloading batch of {len(tickers)} tickers...")

    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        group_by="column",   # <- important: Price-level first, Ticker-level second
        auto_adjust=False,   # keep Adj Close
        progress=True,
        threads=True,
    )

    if data.empty:
        print("[WARN] yfinance returned empty DataFrame for this batch.")
        return pd.DataFrame(columns=["date", "yahoo_ticker", "adj_close"])

    frames = []

    # ---------------------------
    # CASE 1: MultiIndex columns
    # ---------------------------
    if isinstance(data.columns, pd.MultiIndex):
        # Level 0 = price fields, Level 1 = ticker symbols (with names e.g. ("Price","Ticker"))
        level0_vals = list(map(str, data.columns.levels[0]))

        # Decide which price field to use
        if "Adj Close" in level0_vals:
            price_field = "Adj Close"
        elif "Close" in level0_vals:
            price_field = "Close"
        else:
            raise ValueError(
                f"No 'Adj Close' or 'Close' field in downloaded columns: {level0_vals}"
            )

        # Slice out the chosen price field; columns are now just tickers
        price_df = data.xs(price_field, level=0, axis=1)

        for t in tickers:
            if t not in price_df.columns:
                print(f"[WARN] No column for ticker {t} in price_df, skipping.")
                continue

            s = price_df[t].dropna()
            if s.empty:
                print(f"[WARN] No valid prices for {t}, skipping.")
                continue

            df = s.to_frame(name="adj_close")
            df["date"] = df.index
            df["yahoo_ticker"] = t
            frames.append(df.reset_index(drop=True))

    # ---------------------------
    # CASE 2: Single ticker, flat columns
    # ---------------------------
    else:
        cols = list(map(str, data.columns))

        if "Adj Close" in cols:
            price_field = "Adj Close"
        elif "Close" in cols:
            price_field = "Close"
        else:
            raise ValueError(
                f"No 'Adj Close' or 'Close' in single-ticker columns: {cols}"
            )

        t = tickers[0] if len(tickers) == 1 else tickers
        s = data[price_field].dropna()
        df = s.to_frame(name="adj_close")
        df["date"] = df.index
        df["yahoo_ticker"] = t
        frames.append(df.reset_index(drop=True))

    if not frames:
        return pd.DataFrame(columns=["date", "yahoo_ticker", "adj_close"])

    out = pd.concat(frames, ignore_index=True)
    return out


def build_price_panel(cons_with_yahoo: pd.DataFrame) -> pd.DataFrame:
    """
    Loop over Yahoo tickers in batches and build full price panel.
    """
    tickers = cons_with_yahoo["yahoo_ticker"].unique().tolist()
    print(f"[INFO] Requested {len(tickers)} tickers after cleaning.")

    all_frames = []

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        batch_df = download_batch(batch, START_DATE, END_DATE, INTERVAL)
        all_frames.append(batch_df)

    if not all_frames:
        raise RuntimeError("No data downloaded; check tickers and internet connection.")

    panel = pd.concat(all_frames, ignore_index=True)

    # Merge back raw tickers (and any other cons info you want)
    panel = panel.merge(
        cons_with_yahoo[["raw_ticker", "yahoo_ticker"]],
        on="yahoo_ticker",
        how="left",
    )

    # Standardise column order
    panel = panel[["date", "yahoo_ticker", "raw_ticker", "adj_close"]]

    # Sort
    panel = panel.sort_values(["yahoo_ticker", "date"]).reset_index(drop=True)

    print(f"[INFO] Price panel shape: {panel.shape}")
    return panel


# ---------------------------------------------------------------------
# STEP 4: Compute returns & 12–2 momentum
# ---------------------------------------------------------------------


def add_returns_and_momentum(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Given a tidy panel with columns:
        ['date', 'yahoo_ticker', 'ticker', 'adj_close', ...]
    compute:
        - simple returns by ticker
        - 12–2 momentum (Carhart style): (P_{t-2} / P_{t-12}) - 1

    Returns the same DataFrame with extra columns:
        'ret', 'mom_12_2'
    """

    # The panel produced by this script uses 'yahoo_ticker' as the
    # identifier column (from build_price_panel). Use that consistently
    # here instead of an ambiguous 'ticker' column.
    # Make sure we're sorted consistently
    panel = panel.sort_values(["yahoo_ticker", "date"]).reset_index(drop=True)

    # Group by yahoo_ticker once
    g = panel.groupby("yahoo_ticker", group_keys=False)

    # 1) One-month simple return: P_t / P_{t-1} - 1
    panel["ret"] = g["adj_close"].pct_change()

    # 2) 12–2 momentum:
    #    For each ticker, mom_t = P_{t-2} / P_{t-12} - 1
    def _mom_12_2(price: pd.Series) -> pd.Series:
        p_lag2 = price.shift(2)
        p_lag12 = price.shift(12)
        return p_lag2 / p_lag12 - 1

    # Use transform so the returned series is aligned with `panel`'s index
    # (no MultiIndex). transform applies the function to each group and
    # returns a Series with the same index as the original DataFrame.
    mom = g["adj_close"].transform(lambda x: _mom_12_2(x))

    # Now the index matches `panel`'s row index, so this is safe
    panel["mom_12_2"] = mom

    return panel


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():
    print("[STEP] Loading and cleaning constituents...")
    cons = clean_constituents(INPUT_CSV)
    cons = add_yahoo_tickers(cons)

    print("[STEP] Downloading price data...")
    panel = build_price_panel(cons)

    print("[STEP] Computing returns and 12–2 momentum...")
    panel = add_returns_and_momentum(panel)

    print(f"[STEP] Writing panel to {OUTPUT_CSV} ...")
    panel.to_csv(OUTPUT_CSV, index=False)
    print(
        f"[DONE] Wrote {len(panel):,} rows for "
        f"{panel['yahoo_ticker'].nunique()} tickers -> {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()