"""Point-in-time queries over the bitemporal fact table.

Three views of the same data, and the difference between them is the entire
point of the warehouse:

* :func:`as_of` -- what was knowable on a given date. The only view a backtest
  may use.
* :func:`latest` -- the most recently filed value for every period. What a
  conventional fundamentals feed gives you, and what quietly injects
  look-ahead bias into most fundamental backtests.
* :func:`restatements` -- where those two disagree, and by how much.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["cik", "ticker", "concept", "unit", "period_start", "period_end"]

# Instantaneous facts -- balance-sheet items like Assets, StockholdersEquity
# and Cash -- describe a point in time, not a span, so XBRL gives them no
# `start` and they arrive with period_start = NaT.
#
# NaT does not equal NaT in a join or a groupby key. Leaving it in place means
# every balance-sheet concept is silently dropped from any merge on KEY, which
# is how restatement detection came to ignore the entire balance sheet without
# raising anything. Normalising to a sentinel the domain can never produce
# keeps those facts joinable. The same sentinel is used by the DuckDB loader,
# for the same reason (a NULL is distinct in a primary key).
NULL_PERIOD_START = pd.Timestamp("1900-01-01")


def _normalize(facts: pd.DataFrame) -> pd.DataFrame:
    if facts.empty:
        return facts.copy()
    out = facts.copy()
    out["period_start"] = pd.to_datetime(out["period_start"], errors="coerce").fillna(NULL_PERIOD_START)
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce")
    out["filed_date"] = pd.to_datetime(out["filed_date"], errors="coerce")
    return out


def as_of(facts: pd.DataFrame, date: pd.Timestamp | str) -> pd.DataFrame:
    """The fact table as it was knowable on `date`.

    For each (company, concept, period), keeps the most recently filed value
    whose `filed_date` is on or before `date`. Periods first reported *after*
    `date` disappear entirely -- which is correct, and is the part naive
    implementations get wrong: it is not enough to filter the value, the row
    must not exist at all.
    """
    date = pd.Timestamp(date)
    facts = _normalize(facts)
    visible = facts[facts["filed_date"] <= date]
    if visible.empty:
        return visible.copy()

    # Ties on filed_date (an amendment filed the same day as the original)
    # break toward the LATER accession, which is the amended figure.
    ordered = visible.sort_values(["filed_date", "accession"])
    return (
        ordered.groupby(KEY, dropna=False, as_index=False)
        .last()
        .reset_index(drop=True)
    )


def latest(facts: pd.DataFrame) -> pd.DataFrame:
    """The most recently filed value for every period, ignoring when it was
    filed. This is the *restated* view -- convenient, and unusable for any
    backtest that claims to trade before the restatement date."""
    if facts.empty:
        return facts.copy()
    ordered = _normalize(facts).sort_values(["filed_date", "accession"])
    return ordered.groupby(KEY, dropna=False, as_index=False).last().reset_index(drop=True)


def first_reported(facts: pd.DataFrame) -> pd.DataFrame:
    """The value as ORIGINALLY filed for each period, plus its filing date."""
    if facts.empty:
        return facts.copy()
    ordered = _normalize(facts).sort_values(["filed_date", "accession"])
    return ordered.groupby(KEY, dropna=False, as_index=False).first().reset_index(drop=True)


def restatements(facts: pd.DataFrame, min_pct_change: float = 0.0) -> pd.DataFrame:
    """Periods whose reported value changed after first publication.

    Returns one row per restated (company, concept, period) with the original
    and final values, the revision size, and the lag before the correction
    appeared -- `days_to_restate` being how long a backtest using the restated
    figure would have been trading on information nobody had.
    """
    if facts.empty:
        return pd.DataFrame(
            columns=KEY + ["original_value", "final_value", "pct_change",
                           "first_filed", "last_filed", "days_to_restate", "n_filings"]
        )

    facts = _normalize(facts)
    original = first_reported(facts).rename(
        columns={"value": "original_value", "filed_date": "first_filed"}
    )
    final = latest(facts).rename(columns={"value": "final_value", "filed_date": "last_filed"})

    merged = original[KEY + ["original_value", "first_filed"]].merge(
        final[KEY + ["final_value", "last_filed"]], on=KEY, how="inner"
    )
    counts = facts.groupby(KEY, dropna=False).size().rename("n_filings").reset_index()
    merged = merged.merge(counts, on=KEY, how="left")

    changed = merged[merged["original_value"] != merged["final_value"]].copy()
    if changed.empty:
        return changed

    denominator = changed["original_value"].abs().replace(0, np.nan)
    changed["pct_change"] = 100.0 * (changed["final_value"] - changed["original_value"]) / denominator
    changed["days_to_restate"] = (changed["last_filed"] - changed["first_filed"]).dt.days
    changed["is_scale_artifact"] = _is_scale_change(changed["original_value"], changed["final_value"])
    changed["revision_class"] = _classify_revision(
        changed["original_value"], changed["final_value"], changed["is_scale_artifact"]
    )

    if min_pct_change > 0:
        changed = changed[changed["pct_change"].abs() >= min_pct_change]

    return changed.sort_values("pct_change", key=abs, ascending=False).reset_index(drop=True)


def _is_scale_change(original: pd.Series, final: pd.Series, tolerance: float = 0.01) -> pd.Series:
    """Flag value changes that are a power-of-ten rescaling, not a restatement.

    Filers change the `decimals`/scale attribute on an XBRL fact between
    filings -- reporting in thousands one year and in units the next. The
    value then appears to change by exactly 1000x while the underlying
    economics are identical. Atmos Energy's FY2010 net income shows up as
    73,997 in one filing and 73,997,000 in a later one: a 99,900% "restatement"
    that is really a units change.

    Left undetected these dominate every restatement statistic (they are the
    entire right tail) and, worse, they corrupt any ratio built from facts of
    different vintages -- an ROE using net income in thousands over equity in
    units is off by 1000x and looks like a spectacular outlier rather than an
    error.

    Detects same-sign changes whose ratio is within `tolerance` of a non-zero
    integer power of ten.
    """
    ratio = final / original.replace(0, np.nan)
    same_sign = ratio > 0
    exponent = np.log10(ratio.where(same_sign).abs())
    near_integer = (exponent - exponent.round()).abs() < tolerance
    non_zero = exponent.round().abs() >= 1
    return (same_sign & near_integer & non_zero).fillna(False)


IMPLAUSIBLE_REVISION_RATIO = 100.0


def _classify_revision(
    original: pd.Series, final: pd.Series, is_scale_artifact: pd.Series,
    max_ratio: float = IMPLAUSIBLE_REVISION_RATIO,
) -> pd.Series:
    """Sort each value change into what plausibly caused it.

    Three classes:

    * ``power_of_ten_rescale`` -- caught exactly by :func:`_is_scale_change`.
    * ``implausible_ratio`` -- the value moved by a factor of ``max_ratio`` or
      more in either direction. Not a restatement: a company does not revise
      its net income by 100x. These are filer tagging errors, dimensional
      members picked up as if consolidated, or scale changes that also carried
      a genuine revision and so miss the exact power-of-ten test above --
      Amcor's Q1 2019 net income arrives as -34 USD in the 10-Q and as
      112,600,000 USD in the 10-K, a "331,176,570% restatement" that is
      nothing of the kind.
    * ``economic`` -- everything else. On this warehouse that is 969 of 973
      changes, and 951 of them are inside 2x.

    The ratio test is deliberately separate from, and much looser than, the
    power-of-ten one. Tightening it further would start swallowing real
    revisions off a near-zero base, where a small absolute correction produces
    a large ratio honestly.
    """
    ratio = (final / original.replace(0, np.nan)).abs()
    out = pd.Series("economic", index=original.index, dtype=object)
    out[ratio >= max_ratio] = "implausible_ratio"
    out[is_scale_artifact.fillna(False)] = "power_of_ten_rescale"
    return out


def restatement_summary(facts: pd.DataFrame) -> dict:
    """Headline statistics for how much the restated view differs from what
    was actually knowable."""
    all_periods = latest(facts)
    changed_all = restatements(facts)

    # Filing artifacts are not economic revisions and they are the entire right
    # tail. Excluding only exact power-of-ten rescales was not enough: it left
    # a "331,176,570% restatement" (Amcor, see _classify_revision) sitting in
    # max_abs_pct_change, which is precisely the number the exclusion existed
    # to keep out. Both artifact classes are excluded from the headline and
    # counted explicitly, so the exclusion stays auditable rather than silent.
    if len(changed_all):
        classes = changed_all["revision_class"].value_counts().to_dict()
        changed = changed_all[changed_all["revision_class"] == "economic"]
    else:
        classes, changed = {}, changed_all

    n_periods = len(all_periods)
    n_changed = len(changed)
    material = changed[changed["pct_change"].abs() >= 5.0] if n_changed else changed
    absolute = changed["pct_change"].abs() if n_changed else pd.Series(dtype=float)

    return {
        "n_company_concept_periods": int(n_periods),
        "n_restated": int(n_changed),
        "pct_restated": round(100.0 * n_changed / n_periods, 2) if n_periods else 0.0,
        "n_restated_over_5pct": int(len(material)),
        "n_scale_artifacts_excluded": int(classes.get("power_of_ten_rescale", 0)),
        "n_implausible_ratio_excluded": int(classes.get("implausible_ratio", 0)),
        "median_abs_pct_change": round(float(absolute.median()), 2) if n_changed else 0.0,
        "p95_abs_pct_change": round(float(absolute.quantile(0.95)), 2) if n_changed else 0.0,
        "p99_abs_pct_change": round(float(absolute.quantile(0.99)), 2) if n_changed else 0.0,
        "max_abs_pct_change": round(float(absolute.max()), 2) if n_changed else 0.0,
        "median_days_to_restate": int(changed["days_to_restate"].median()) if n_changed else 0,
    }


def lookahead_exposure(facts: pd.DataFrame, trade_date: pd.Timestamp | str) -> pd.DataFrame:
    """Facts that a `latest`-based backtest would use on `trade_date` but that
    were NOT knowable then -- either not yet filed, or later revised.

    This is the concrete audit: point it at any date your strategy trades on
    and it lists exactly which numbers you would have been cheating with.
    """
    trade_date = pd.Timestamp(trade_date)
    facts = _normalize(facts)
    known = as_of(facts, trade_date)
    restated_view = latest(facts)

    merged = restated_view.merge(
        known[KEY + ["value", "filed_date"]],
        on=KEY,
        how="left",
        suffixes=("_restated", "_knowable"),
    )

    not_yet_filed = merged["value_knowable"].isna()
    revised = (~not_yet_filed) & (merged["value_restated"] != merged["value_knowable"])
    exposed = merged[not_yet_filed | revised].copy()
    exposed["issue"] = np.where(
        exposed["value_knowable"].isna(), "not yet filed", "value later revised"
    )
    return exposed.reset_index(drop=True)
