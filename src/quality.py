"""Data quality gates that fail the pipeline instead of warning about it.

A quality check that logs and continues is a check nobody reads. Each gate
here returns a pass/fail with the offending rows attached, and
:func:`run_gates` raises when any BLOCKING gate fails, so a bad load stops
before anything downstream consumes it.

Gates are split into blocking and advisory deliberately. "A fact was filed
before the period it describes ended" is a data-integrity failure and must
stop the pipeline. "This company restated more than 20% of its periods" is
unusual but real, and blocking on it would just mean the pipeline never runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class GateResult:
    name: str
    passed: bool
    blocking: bool
    n_offending: int
    message: str
    offending: pd.DataFrame = field(default_factory=pd.DataFrame)


class QualityGateFailure(RuntimeError):
    def __init__(self, failures: list[GateResult]):
        self.failures = failures
        super().__init__(
            "blocking quality gate(s) failed:\n"
            + "\n".join(f"  - {f.name}: {f.message}" for f in failures)
        )


def gate_no_future_filings(facts: pd.DataFrame, today: pd.Timestamp | None = None) -> GateResult:
    """No fact may be filed in the future. Catches timezone/parsing bugs that
    would silently make data visible before it existed."""
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    bad = facts[facts["filed_date"] > today]
    return GateResult(
        "no_future_filings", bad.empty, True, len(bad),
        f"{len(bad)} facts filed after {today.date()}", bad,
    )


def gate_filed_after_period_end(facts: pd.DataFrame, grace_days: int = 0) -> GateResult:
    """A fact cannot be filed before the period it reports on has ended.

    This is the single most important structural check in the warehouse: a
    fact whose filed_date precedes its period_end would make future
    information visible to `as_of` queries, which is the exact bias the whole
    project exists to prevent.
    """
    bad = facts[facts["filed_date"] < (facts["period_end"] - pd.Timedelta(grace_days, unit="D"))]
    return GateResult(
        "filed_after_period_end", bad.empty, True, len(bad),
        f"{len(bad)} facts filed BEFORE the period they report ended", bad,
    )


def gate_period_start_before_end(facts: pd.DataFrame) -> GateResult:
    dated = facts.dropna(subset=["period_start"])
    bad = dated[dated["period_start"] > dated["period_end"]]
    return GateResult(
        "period_start_before_end", bad.empty, True, len(bad),
        f"{len(bad)} facts with period_start after period_end", bad,
    )


def gate_no_duplicate_facts(facts: pd.DataFrame) -> GateResult:
    """One value per (company, concept, unit, period, filing). A duplicate
    means the same filing was parsed twice, which would double-weight it in
    any aggregate."""
    key = ["cik", "concept", "unit", "period_start", "period_end", "accession"]
    dupes = facts[facts.duplicated(subset=key, keep=False)]
    return GateResult(
        "no_duplicate_facts", dupes.empty, True, len(dupes),
        f"{len(dupes)} duplicate facts on {key}", dupes,
    )


def gate_membership_has_plausible_size(
    membership: pd.DataFrame, expected: int = 500, tolerance: float = 0.15
) -> GateResult:
    """Each membership sample should hold roughly the index's real size.

    A sample that parsed 12 tickers means the scrape silently picked up the
    wrong table -- a failure mode that produces a perfectly well-formed but
    entirely wrong universe, and would otherwise go unnoticed.
    """
    if membership.empty:
        return GateResult("membership_plausible_size", False, True, 0, "membership table is empty")

    sizes = membership.groupby("sample_date").size()
    low, high = expected * (1 - tolerance), expected * (1 + tolerance)
    bad = sizes[(sizes < low) | (sizes > high)]
    return GateResult(
        "membership_plausible_size", bad.empty, True, len(bad),
        f"{len(bad)} membership samples outside {low:.0f}-{high:.0f} names",
        bad.reset_index(name="n_members"),
    )


def gate_prices_positive(prices: pd.DataFrame) -> GateResult:
    bad = prices[(prices["close"] <= 0) | prices["close"].isna()]
    return GateResult(
        "prices_positive", bad.empty, True, len(bad),
        f"{len(bad)} rows with non-positive or missing close", bad,
    )


def gate_restatement_rate(facts: pd.DataFrame, max_pct: float = 25.0) -> GateResult:
    """ADVISORY. A very high restatement rate is more likely a parsing bug
    (mixing units, or treating amended and original filings as one series)
    than a company restating half its history."""
    from src.pit import restatement_summary

    summary = restatement_summary(facts)
    rate = summary["pct_restated"]
    return GateResult(
        "restatement_rate", rate <= max_pct, False, summary["n_restated"],
        f"{rate:.1f}% of periods restated (advisory threshold {max_pct}%)",
    )


def gate_price_coverage(prices: pd.DataFrame, universe: list[str], min_pct: float = 80.0) -> GateResult:
    """ADVISORY. What fraction of the point-in-time universe actually has
    price data. Free sources do not serve delisted names, so this will not
    reach 100% -- it is here to make the residual survivorship bias a reported
    number rather than a silent one."""
    have = set(prices["ticker"].unique()) & set(universe)
    pct = 100.0 * len(have) / max(len(universe), 1)
    missing = sorted(set(universe) - have)
    return GateResult(
        "price_coverage", pct >= min_pct, False, len(missing),
        f"{pct:.1f}% of the universe has price data; {len(missing)} names missing",
        pd.DataFrame({"ticker": missing}),
    )


def run_gates(gates: list[GateResult], raise_on_failure: bool = True) -> list[GateResult]:
    """Report every gate, then raise if any BLOCKING one failed."""
    for gate in gates:
        status = "PASS" if gate.passed else ("FAIL" if gate.blocking else "WARN")
        print(f"  [{status}] {gate.name:<28} {gate.message}")

    failures = [g for g in gates if not g.passed and g.blocking]
    if failures and raise_on_failure:
        raise QualityGateFailure(failures)
    return failures
