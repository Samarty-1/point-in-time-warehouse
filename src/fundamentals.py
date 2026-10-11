"""Assembling comparable fundamentals out of a bitemporal fact table.

Why this module exists
----------------------
The warehouse stores facts exactly as filed, which is the right thing for a
bitemporal store and the wrong thing to hand straight to a cross-sectional
signal. XBRL tags the same concept over several different period lengths in
the same filing: ``NetIncomeLoss`` for a company arrives as a ~90-day quarter,
a ~180-day half, a ~270-day nine-month figure and a ~365-day year, all with
different ``period_start`` values and all equally valid.

Picking "the fact with the latest ``period_end``" -- the obvious
implementation -- therefore picks a **different period length for different
companies**, because which length ends latest depends on the filer's tagging
habits and on where in the reporting calendar the query lands. Measured on
this warehouse, that one line produced a cross-section holding 73 quarterly
figures, 7 half-year, 5 nine-month and 7 annual (2022-06-30, point-in-time
arm). The names carrying an annual numerator showed a median ROE of 0.109
against 0.039 for the names carrying a quarterly one -- a 2.8x gap that is
purely an artifact of period length.

That gap is not noise. A filer's tagging behaviour is stable over time, so the
contaminated ranking sorts a **persistent group of companies** into the top
quintile and a different persistent group into the bottom. A long-short book
built on it is substantially long "companies whose latest tagged figure is
annual" and short "companies whose latest tagged figure is quarterly".

Worse, for this repo's actual purpose: the point-in-time and restated arms see
different fact sets, so they land on *different* period-length mixes. Any gap
measured between them then mixes look-ahead bias -- the thing being measured --
with a period-length artifact. The headline number is not interpretable until
the numerator is made consistent, which is what this module does.

What it produces
----------------
:func:`ttm_flow` returns one trailing-twelve-month figure per ticker, built
the same way for every name, with the assembly method and the window it
covers reported alongside so a caller can audit or filter on them.
:func:`instant_stock` does the balance-sheet equivalent. Names that cannot
produce a clean twelve-month window are **dropped rather than approximated**:
substituting a quarter for a year is the bug this module exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Period-length classification, in days. XBRL period boundaries wobble by a few
# days around fiscal calendars (52/53-week retailers are the worst offenders),
# so these are bands rather than exact values.
QUARTER = (75, 115)
HALF = (160, 205)
NINE_MONTH = (250, 295)
ANNUAL = (340, 390)

# A twelve-month window assembled from quarters must actually cover a year.
TTM_SPAN = (340, 400)
# Consecutive quarters may leave a small calendar gap at fiscal year ends.
MAX_QUARTER_GAP_DAYS = 8

NULL_PERIOD_START = pd.Timestamp("1900-01-01")


def classify_period(period_start: pd.Series, period_end: pd.Series) -> pd.Series:
    """Label each fact by the length of the period it covers.

    ``instant`` is a balance-sheet fact: XBRL gives it no start date, and
    :mod:`src.pit` normalises that to a sentinel so it stays joinable.
    """
    start = pd.to_datetime(period_start)
    end = pd.to_datetime(period_end)
    span = (end - start).dt.days

    out = pd.Series("other", index=span.index, dtype=object)
    out[start.eq(NULL_PERIOD_START) | start.isna()] = "instant"
    for label, (low, high) in (
        ("quarter", QUARTER), ("half", HALF),
        ("ninemonth", NINE_MONTH), ("annual", ANNUAL),
    ):
        out[(out == "other") & span.between(low, high)] = label
    return out


def _prepare(facts: pd.DataFrame, concept: str) -> pd.DataFrame:
    subset = facts[facts["concept"] == concept].copy()
    if subset.empty:
        return subset
    subset["period_start"] = pd.to_datetime(subset["period_start"], errors="coerce").fillna(NULL_PERIOD_START)
    subset["period_end"] = pd.to_datetime(subset["period_end"], errors="coerce")
    subset["period_kind"] = classify_period(subset["period_start"], subset["period_end"])
    # One value per period. The caller is expected to have already resolved
    # vintage (`as_of` or `latest`); this only guards against a view that still
    # carries several filings of the same period.
    if "filed_date" in subset.columns:
        subset = (
            subset.sort_values(["filed_date", "accession"])
            .groupby(["ticker", "period_start", "period_end"], as_index=False)
            .last()
        )
    return subset


def _ttm_from_quarters(quarters: pd.DataFrame) -> tuple[float, pd.Timestamp, pd.Timestamp] | None:
    """Sum the four most recent consecutive quarters, or None if they do not
    form a clean twelve-month window.

    Requiring contiguity matters: a company that skipped a quarter in the
    warehouse (a gap in EDGAR coverage, or a fiscal-year change) would
    otherwise get a "TTM" spanning fifteen months or missing three, and it
    would look like a profitability outlier rather than a data gap.
    """
    recent = quarters.sort_values("period_end").tail(4)
    if len(recent) < 4:
        return None

    starts = recent["period_start"].tolist()
    ends = recent["period_end"].tolist()
    for previous_end, next_start in zip(ends[:-1], starts[1:]):
        if abs((next_start - previous_end).days) > MAX_QUARTER_GAP_DAYS:
            return None

    span = (ends[-1] - starts[0]).days
    if not TTM_SPAN[0] <= span <= TTM_SPAN[1]:
        return None
    return float(recent["value"].sum()), starts[0], ends[-1]


def ttm_flow(
    facts: pd.DataFrame,
    concept: str,
    as_of: pd.Timestamp | str | None = None,
    max_staleness_days: int = 400,
) -> pd.DataFrame:
    """One trailing-twelve-month figure per ticker, built identically for all.

    Two assembly routes, and the fresher window wins:

    * ``sum4q`` -- the four most recent consecutive quarters summed. Preferred
      when it ends later, because it picks up three quarters of news that the
      last annual filing does not have.
    * ``annual`` -- a single tagged ~365-day fact. Preferred on a tie, being
      one number the filer stood behind rather than four this code added up.

    Returns columns ``ticker, value, method, window_start, window_end``. A
    ticker with neither route available is absent from the result: an
    approximate numerator is worse than a missing one, because a missing name
    drops out of the cross-section while an approximate one gets ranked.
    """
    prepared = _prepare(facts, concept)
    if prepared.empty:
        return pd.DataFrame(columns=["ticker", "value", "method", "window_start", "window_end"])

    # Cap the candidate set BEFORE choosing a window, not after.
    #
    # This is the load-bearing line. A restated fact view carries every period
    # the warehouse ever learned about, including ones that had not ended on
    # `as_of`; picking "the latest window" out of that and only then checking
    # the date does not reject a future period, it rejects the ticker. Worse,
    # unfiltered it silently hands a 2022 rebalance Apple's FY2025 net income.
    # That is a multi-year look-ahead, an entirely cruder thing than the
    # restatement effect this warehouse exists to measure, and it would swamp
    # it. Capping first leaves the two arms differing only in WHICH VINTAGE of
    # an already-ended period they see.
    if as_of is not None:
        prepared = prepared[prepared["period_end"] <= pd.Timestamp(as_of)]
        if prepared.empty:
            return pd.DataFrame(columns=["ticker", "value", "method", "window_start", "window_end"])

    rows = []
    for ticker, group in prepared.groupby("ticker"):
        candidates = []

        quarters = group[group["period_kind"] == "quarter"]
        if len(quarters) >= 4:
            assembled = _ttm_from_quarters(quarters)
            if assembled is not None:
                value, start, end = assembled
                candidates.append((end, 1, value, "sum4q", start))

        annuals = group[group["period_kind"] == "annual"]
        if not annuals.empty:
            latest_annual = annuals.sort_values("period_end").iloc[-1]
            candidates.append((
                latest_annual["period_end"], 0, float(latest_annual["value"]),
                "annual", latest_annual["period_start"],
            ))

        if not candidates:
            continue
        # Later window first; on an identical end date the annual fact (rank 0)
        # sorts ahead of the assembled one (rank 1).
        end, _, value, method, start = sorted(candidates, key=lambda c: (c[0], -c[1]))[-1]
        rows.append({"ticker": ticker, "value": value, "method": method,
                     "window_start": start, "window_end": end})

    result = pd.DataFrame(rows)
    if result.empty or as_of is None:
        return result

    cutoff = pd.Timestamp(as_of) - pd.Timedelta(max_staleness_days, unit="D")
    return result[result["window_end"] >= cutoff].reset_index(drop=True)


def instant_stock(
    facts: pd.DataFrame,
    concept: str,
    as_of: pd.Timestamp | str | None = None,
    max_staleness_days: int = 400,
) -> pd.DataFrame:
    """Latest balance-sheet value per ticker, with the same staleness gate.

    Balance-sheet concepts are instantaneous, so there is no period length to
    reconcile -- but a stale one is just as damaging in a ratio, and a name
    whose equity is three years old should not be ranked against names whose
    equity is one quarter old.
    """
    prepared = _prepare(facts, concept)
    if prepared.empty:
        return pd.DataFrame(columns=["ticker", "value", "as_of_date"])

    instants = prepared[prepared["period_kind"] == "instant"]
    # Same ordering as ttm_flow, and for the same reason: restrict to balance
    # dates that had passed, then pick the latest of those.
    if as_of is not None:
        instants = instants[instants["period_end"] <= pd.Timestamp(as_of)]
    if instants.empty:
        return pd.DataFrame(columns=["ticker", "value", "as_of_date"])

    latest = instants.sort_values("period_end").groupby("ticker", as_index=False).last()
    result = latest[["ticker", "value", "period_end"]].rename(columns={"period_end": "as_of_date"})

    if as_of is None:
        return result.reset_index(drop=True)
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(max_staleness_days, unit="D")
    return result[result["as_of_date"] >= cutoff].reset_index(drop=True)


def return_on_equity(
    facts: pd.DataFrame,
    as_of: pd.Timestamp | str | None = None,
    max_staleness_days: int = 400,
    winsorize: float | None = 1.0,
) -> pd.DataFrame:
    """Trailing-twelve-month net income over latest common equity.

    Both legs come from the same fact view, so a point-in-time view yields a
    point-in-time ROE and a restated view yields a restated one -- which is the
    only difference the two arms of the look-ahead measurement should have.

    ``winsorize`` clips the ratio to +/- that value. It is a tail control, not
    a data-quality fix: with a consistent twelve-month numerator the extreme
    values it used to hide -- ratios off by 4x from period-length mixing, or by
    1000x from a reporting-scale change -- no longer arrive. Clipping is still
    worth keeping because a company with near-zero book equity produces a
    legitimately enormous ROE that would otherwise own the top quintile on its
    own. Pass ``None`` to leave the ratio raw.
    """
    income = ttm_flow(facts, "NetIncomeLoss", as_of, max_staleness_days).rename(
        columns={"value": "net_income", "method": "ni_method",
                 "window_start": "ni_start", "window_end": "ni_end"}
    )
    equity = instant_stock(facts, "StockholdersEquity", as_of, max_staleness_days).rename(
        columns={"value": "equity", "as_of_date": "equity_date"}
    )
    if income.empty or equity.empty:
        return pd.DataFrame(columns=["ticker", "roe", "net_income", "equity", "ni_method"])

    merged = income.merge(equity, on="ticker", how="inner")
    # Negative book equity makes ROE meaningless rather than merely extreme:
    # a loss-making company with negative equity reports a positive ratio.
    merged = merged[merged["equity"] > 0].copy()
    if merged.empty:
        return merged

    merged["roe"] = merged["net_income"] / merged["equity"]
    if winsorize is not None:
        merged["roe"] = merged["roe"].clip(-winsorize, winsorize)
    return merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["roe"]).reset_index(drop=True)


def period_mix(facts: pd.DataFrame, concept: str) -> pd.Series:
    """How many tickers a naive ``latest period_end`` pick would take from each
    period length. Kept as a diagnostic so the defect this module fixes stays
    measurable rather than becoming folklore.
    """
    prepared = _prepare(facts, concept)
    if prepared.empty:
        return pd.Series(dtype=int)
    picked = prepared.sort_values("period_end").groupby("ticker", as_index=False).last()
    return picked["period_kind"].value_counts()
