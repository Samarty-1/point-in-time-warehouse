"""Build the warehouse: membership -> prices -> fundamentals -> quality gates.

    python pipeline.py --tickers-from-index --n-tickers 100 --period 10y
    python pipeline.py --tickers AAPL,MSFT,JPM

Every stage is idempotent, so this is safe to re-run and safe to resume after
a failure. Quality gates run last and raise on any blocking failure, so a bad
load never reaches a consumer.
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd
import yfinance as yf

from src import membership as membership_mod
from src import quality
from src.edgar import DEFAULT_CONCEPTS, EdgarClient, fetch_facts
from src.warehouse import connect, load_facts, load_membership, load_prices, table_counts

warnings.filterwarnings("ignore")


def fetch_prices(tickers: list[str], period: str) -> pd.DataFrame:
    raw = yf.download(tickers, period=period, group_by="ticker", auto_adjust=True,
                      progress=False, threads=True)
    frames = []
    for ticker in tickers:
        if ticker not in raw.columns.get_level_values(0):
            continue
        frame = raw[ticker].dropna(how="all").reset_index()
        if frame.empty:
            continue
        frame["ticker"] = ticker
        frame = frame.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                      "Low": "low", "Close": "close", "Volume": "volume"})
        frames.append(frame[["ticker", "date", "open", "high", "low", "close", "volume"]])
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)

    # yfinance emits partially-populated rows around halts, holidays and the
    # edges of a delisting -- a date row exists but close is NaN or zero.
    # Dropped here rather than by relaxing the prices_positive gate: the gate
    # is meant to catch exactly this, and a gate that tolerates its own
    # failure mode is decoration.
    clean = out[(out["close"] > 0) & out["close"].notna()]
    dropped = len(out) - len(clean)
    if dropped:
        print(f"    dropped {dropped:,} price rows with missing/non-positive close")
    return clean.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", default=None, help="Comma-separated; overrides --tickers-from-index")
    ap.add_argument("--tickers-from-index", action="store_true", default=True)
    ap.add_argument("--n-tickers", type=int, default=100, help="Cap the universe (EDGAR is one request per ticker per concept)")
    ap.add_argument("--period", default="10y")
    ap.add_argument("--years-membership", type=int, default=10)
    ap.add_argument("--concepts", default=",".join(DEFAULT_CONCEPTS))
    ap.add_argument("--db", default="data/warehouse.duckdb")
    args = ap.parse_args()

    con = connect(args.db)

    print("[1] point-in-time index membership")
    start = pd.Timestamp.today() - pd.DateOffset(years=args.years_membership)
    history = membership_mod.membership_history(start)
    inserted = load_membership(con, history)
    print(f"    {len(history):,} membership rows over {history['sample_date'].nunique()} samples "
          f"({inserted:,} new)")

    if args.tickers:
        universe = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        # Ever-members, not current members: the names that LEFT the index are
        # the ones a survivorship-biased universe silently drops.
        universe = membership_mod.ever_members(history)[: args.n_tickers]
    print(f"    universe: {len(universe)} tickers")

    print("[2] prices")
    prices = fetch_prices(universe, args.period)
    inserted = load_prices(con, prices)
    print(f"    {len(prices):,} price rows ({inserted:,} new)")

    print("[3] fundamentals from EDGAR (bitemporal)")
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    facts = fetch_facts(universe, concepts, client=EdgarClient())
    inserted = load_facts(con, facts)
    print(f"    {len(facts):,} facts across {facts['ticker'].nunique() if not facts.empty else 0} "
          f"tickers ({inserted:,} new)")

    print("[4] quality gates")
    gates = [
        quality.gate_no_future_filings(facts),
        quality.gate_filed_after_period_end(facts),
        quality.gate_period_start_before_end(facts),
        quality.gate_no_duplicate_facts(facts),
        quality.gate_membership_has_plausible_size(history),
        quality.gate_prices_positive(prices),
        quality.gate_restatement_rate(facts),
        quality.gate_price_coverage(prices, universe),
    ]
    quality.run_gates(gates)

    print("\n[5] warehouse")
    for table, count in table_counts(con).items():
        print(f"    {table:<22} {count:>10,}")
    con.close()


if __name__ == "__main__":
    main()
