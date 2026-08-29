"""Measure what look-ahead bias in fundamentals is actually worth.

    python -m scripts.measure_lookahead

Runs the SAME profitability strategy twice, changing only which vintage of the
fundamentals it sees:

* **point-in-time** -- on each rebalance date, only facts already filed by
  that date (``facts_as_of``). What an investor could actually have known.
* **restated** -- the most recent value for every period, regardless of when
  it was filed (``latest``). What a conventional fundamentals feed hands you,
  and what most fundamental backtests silently use.

The strategy is deliberately plain: rank the universe by return on equity,
go long the top quintile and short the bottom, rebalance quarterly,
equal-weighted. The point is not the strategy -- it is the gap between the two
arms, which is pure look-ahead and exists in any backtest built on restated
data.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.pit import latest, restatement_summary
from src.warehouse import connect, facts_as_of

warnings.filterwarnings("ignore")

REPORTS = Path(__file__).resolve().parent.parent / "reports"
TRADING_DAYS = 252


def _wide(facts: pd.DataFrame, concept: str) -> pd.DataFrame:
    """Most recent value of `concept` per ticker, from an already-filtered
    fact view."""
    subset = facts[facts["concept"] == concept]
    if subset.empty:
        return pd.DataFrame(columns=["ticker", "value"])
    return (
        subset.sort_values("period_end")
        .groupby("ticker", as_index=False)
        .last()[["ticker", "value"]]
    )


def roe_signal(facts: pd.DataFrame) -> pd.Series:
    """Return on equity per ticker from a fact view."""
    income = _wide(facts, "NetIncomeLoss").rename(columns={"value": "net_income"})
    equity = _wide(facts, "StockholdersEquity").rename(columns={"value": "equity"})
    merged = income.merge(equity, on="ticker", how="inner")
    merged = merged[merged["equity"] > 0]
    if merged.empty:
        return pd.Series(dtype=float)

    roe = (merged["net_income"] / merged["equity"]).set_axis(merged["ticker"])
    # Clip to a plausible band. Beyond ordinary winsorising, this defends
    # against XBRL reporting-scale changes (see pit._is_scale_change): if net
    # income and equity are picked up from filings using different scales, the
    # ratio is off by a factor of 1000 and would otherwise dominate the
    # quintile ranking as a pure artifact.
    return roe.clip(-1.0, 1.0)


def quintile_returns(signal: pd.Series, forward_returns: pd.Series, min_names: int = 20) -> float:
    """Long top quintile, short bottom quintile, equal-weighted."""
    common = signal.index.intersection(forward_returns.dropna().index)
    if len(common) < min_names:
        return np.nan

    s = signal.loc[common]
    r = forward_returns.loc[common]
    cut = max(int(len(common) * 0.2), 1)
    ranked = s.sort_values()
    return float(r.loc[ranked.index[-cut:]].mean() - r.loc[ranked.index[:cut]].mean())


def run_arm(con, prices: pd.DataFrame, rebalance_dates, all_facts: pd.DataFrame, point_in_time: bool) -> pd.Series:
    restated_view = None if point_in_time else latest(all_facts)
    wide_prices = prices.pivot_table(index="date", columns="ticker", values="close")

    out = {}
    for i, date in enumerate(rebalance_dates[:-1]):
        facts = facts_as_of(con, str(date.date())) if point_in_time else restated_view
        signal = roe_signal(facts)
        if signal.empty:
            continue

        window = wide_prices.loc[(wide_prices.index > date) & (wide_prices.index <= rebalance_dates[i + 1])]
        if len(window) < 2:
            continue
        forward = window.iloc[-1] / window.iloc[0] - 1

        spread = quintile_returns(signal, forward)
        if np.isfinite(spread):
            out[rebalance_dates[i + 1]] = spread
    return pd.Series(out).sort_index()


def metrics(returns: pd.Series, periods_per_year: float = 4.0) -> dict:
    r = returns.dropna()
    if len(r) < 2:
        return {"n_periods": len(r), "total_return": 0.0, "annualized_return": 0.0, "sharpe": 0.0}
    total = float((1 + r).prod() - 1)
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    return {
        "n_periods": int(len(r)),
        "total_return": round(total, 4),
        "annualized_return": round(float((1 + total) ** (periods_per_year / len(r)) - 1), 4),
        "annualized_vol": round(float(vol), 4),
        "sharpe": round(float(r.mean() * periods_per_year / vol), 4) if vol > 0 else 0.0,
        "hit_rate": round(float((r > 0).mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/warehouse.duckdb")
    ap.add_argument("--start", default=None, help="First rebalance date (default: earliest with data)")
    args = ap.parse_args()

    con = connect(args.db)
    all_facts = con.execute("SELECT * FROM fundamental_facts").df()
    prices = con.execute("SELECT ticker, date, close FROM prices").df()
    if all_facts.empty or prices.empty:
        raise SystemExit("Warehouse is empty -- run `python pipeline.py` first.")

    all_facts["filed_date"] = pd.to_datetime(all_facts["filed_date"])
    all_facts["period_end"] = pd.to_datetime(all_facts["period_end"])
    all_facts["period_start"] = pd.to_datetime(all_facts["period_start"])
    prices["date"] = pd.to_datetime(prices["date"])

    summary = restatement_summary(all_facts)
    print("Restatement profile of the warehouse")
    for key, value in summary.items():
        print(f"    {key:<32} {value}")

    start = pd.Timestamp(args.start) if args.start else max(
        prices["date"].min(), all_facts["filed_date"].min()
    )
    rebalance_dates = pd.date_range(start, prices["date"].max(), freq="QE")
    print(f"\nRebalancing quarterly: {len(rebalance_dates)} dates, "
          f"{rebalance_dates[0].date()} to {rebalance_dates[-1].date()}")

    pit = run_arm(con, prices, rebalance_dates, all_facts, point_in_time=True)
    naive = run_arm(con, prices, rebalance_dates, all_facts, point_in_time=False)

    pit_metrics, naive_metrics = metrics(pit), metrics(naive)

    print(f"\n{'=' * 70}\nROE quintile long-short, same strategy, two data vintages\n{'=' * 70}")
    print(f"{'metric':<22}{'point-in-time':>16}{'restated':>14}{'look-ahead':>14}")
    for key in ("total_return", "annualized_return", "sharpe", "hit_rate"):
        p, n = pit_metrics.get(key, 0), naive_metrics.get(key, 0)
        print(f"{key:<22}{p:>16.4f}{n:>14.4f}{n - p:>14.4f}")
    print(f"{'periods':<22}{pit_metrics['n_periods']:>16}{naive_metrics['n_periods']:>14}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "lookahead_measurement.json"
    out.write_text(json.dumps({
        "restatement_summary": summary,
        "point_in_time": pit_metrics,
        "restated": naive_metrics,
        "pit_returns": {str(k.date()): v for k, v in pit.items()},
        "restated_returns": {str(k.date()): v for k, v in naive.items()},
    }, indent=2, default=str))
    print(f"\nSaved {out}")
    con.close()


if __name__ == "__main__":
    main()
