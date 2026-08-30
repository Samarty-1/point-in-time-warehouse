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

from src import fundamentals
from src.pit import latest, restatement_summary
from src.warehouse import connect, facts_as_of

warnings.filterwarnings("ignore")

REPORTS = Path(__file__).resolve().parent.parent / "reports"
TRADING_DAYS = 252


def roe_signal(facts: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.Series:
    """Return on equity per ticker from a fact view.

    The numerator is a trailing-twelve-month figure assembled the same way for
    every name (see :mod:`src.fundamentals`). An earlier version took whichever
    ``NetIncomeLoss`` fact had the latest ``period_end``, which silently mixed
    quarterly, half-year, nine-month and annual figures inside one
    cross-section -- and, because the two arms below see different fact sets,
    mixed them *differently between the arms*. The measured gap was then part
    look-ahead and part period-length artifact.
    """
    roe = fundamentals.return_on_equity(facts, as_of=as_of)
    if roe.empty:
        return pd.Series(dtype=float)
    return roe.set_index("ticker")["roe"]


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
        signal = roe_signal(facts, as_of=date)
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

    # Kept in the output permanently. This mix is what a naive "latest
    # period_end" pick would hand the ranking, and it is the defect that made
    # the previous version of this measurement uninterpretable: the arms landed
    # on different period lengths, so the gap between them was part look-ahead
    # and part artifact. Printing it means a regression is visible rather than
    # silent.
    print("\nPeriod-length mix a naive 'latest period_end' pick would produce")
    mixes = {}
    for label, view in (("restated", latest(all_facts)),
                        ("point-in-time @2022-06-30", facts_as_of(con, "2022-06-30"))):
        mix = fundamentals.period_mix(view, "NetIncomeLoss")
        mixes[label] = {str(k): int(v) for k, v in mix.items()}
        print(f"    {label:<28} {mixes[label]}")
    print("    (this measurement uses fundamentals.ttm_flow instead: "
          "one twelve-month window for every name)")

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
        "naive_period_length_mix": mixes,
        "signal": {
            "numerator": "trailing twelve months, fundamentals.ttm_flow",
            "denominator": "latest common equity at or before the rebalance date",
            "max_staleness_days": 400,
            "winsorize_roe": 1.0,
        },
        "point_in_time": pit_metrics,
        "restated": naive_metrics,
        "pit_returns": {str(k.date()): v for k, v in pit.items()},
        "restated_returns": {str(k.date()): v for k, v in naive.items()},
    }, indent=2, default=str))
    print(f"\nSaved {out}")
    con.close()


if __name__ == "__main__":
    main()
