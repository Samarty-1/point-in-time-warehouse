"""Tests for the DuckDB layer: idempotency, append-only history, and that the
SQL point-in-time query agrees with the pandas one.

The last of those matters most. Two implementations of the same semantics
that disagree is worse than one -- whichever a caller happens to use decides
whether their backtest leaks the future.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import quality
from src.pit import as_of
from src.warehouse import (
    connect,
    facts_as_of,
    load_facts,
    load_membership,
    load_prices,
    members_as_of,
    restatement_report,
    table_counts,
)


@pytest.fixture
def facts() -> pd.DataFrame:
    base = dict(cik=320193, ticker="AAPL", concept="NetIncomeLoss", unit="USD",
                period_start=pd.Timestamp("2007-09-30"), period_end=pd.Timestamp("2008-09-27"),
                fiscal_year=2009, fiscal_period="FY")
    return pd.DataFrame([
        {**base, "filed_date": pd.Timestamp("2009-10-27"), "value": 4_834_000_000.0,
         "form": "10-K", "accession": "0001"},
        {**base, "filed_date": pd.Timestamp("2010-01-25"), "value": 6_119_000_000.0,
         "form": "10-K/A", "accession": "0002"},
        {**base, "period_start": pd.Timestamp("2008-09-28"), "period_end": pd.Timestamp("2009-09-26"),
         "filed_date": pd.Timestamp("2010-10-27"), "value": 8_235_000_000.0,
         "form": "10-K", "accession": "0003"},
    ])


@pytest.fixture
def con(tmp_path):
    connection = connect(tmp_path / "test.duckdb")
    yield connection
    connection.close()


def test_load_is_idempotent(con, facts):
    """Re-running the loader on the same data must converge, not duplicate.
    This is what lets a failed pipeline be retried without cleanup."""
    first = load_facts(con, facts)
    second = load_facts(con, facts)

    assert first == 3
    assert second == 0
    assert table_counts(con)["fundamental_facts"] == 3


def test_overlapping_backfill_inserts_only_new_rows(con, facts):
    load_facts(con, facts.iloc[:2])
    inserted = load_facts(con, facts)  # overlaps the first two

    assert inserted == 1
    assert table_counts(con)["fundamental_facts"] == 3


def test_restatement_is_appended_not_overwritten(con, facts):
    """Both vintages of the FY2008 figure must survive. Overwriting would
    destroy exactly the history the warehouse exists to keep."""
    load_facts(con, facts)
    values = con.execute(
        "SELECT value FROM fundamental_facts WHERE period_end = '2008-09-27' ORDER BY filed_date"
    ).fetchall()

    assert [v[0] for v in values] == [4_834_000_000.0, 6_119_000_000.0]


def test_sql_and_pandas_point_in_time_agree(con, facts):
    """The two implementations must never diverge."""
    load_facts(con, facts)

    for date in ("2009-11-15", "2010-06-01", "2011-01-01"):
        sql_view = facts_as_of(con, date).sort_values("period_end")["value"].tolist()
        pandas_view = as_of(facts, date).sort_values("period_end")["value"].tolist()
        assert sql_view == pandas_view, f"divergence on {date}"


def test_sql_point_in_time_hides_unfiled_periods(con, facts):
    load_facts(con, facts)
    view = facts_as_of(con, "2009-11-15")

    assert len(view) == 1
    assert view.iloc[0]["value"] == 4_834_000_000.0


def test_load_log_records_offered_and_inserted(con, facts):
    load_facts(con, facts, note="initial")
    load_facts(con, facts, note="rerun")
    log = con.execute("SELECT rows_offered, rows_inserted, note FROM load_log ORDER BY loaded_at").df()

    assert log.iloc[0]["rows_inserted"] == 3
    assert log.iloc[1]["rows_inserted"] == 0
    assert log.iloc[1]["rows_offered"] == 3


def test_members_as_of_never_carries_membership_backward(con):
    membership = pd.DataFrame({
        "sample_date": [pd.Timestamp("2020-03-31")] * 2 + [pd.Timestamp("2020-09-30")] * 2,
        "ticker": ["GOOD", "LEAVER", "GOOD", "JOINER"],
        "revision_id": [1, 1, 2, 2],
    })
    load_membership(con, membership)

    assert members_as_of(con, "2020-06-15") == ["GOOD", "LEAVER"]
    assert members_as_of(con, "2020-10-15") == ["GOOD", "JOINER"]
    assert members_as_of(con, "2020-01-01") == []


def test_restatement_report_matches_expected_revision(con, facts):
    load_facts(con, facts)
    report = restatement_report(con)

    assert len(report) == 1
    assert report.iloc[0]["pct_change"] == pytest.approx(26.58, abs=0.01)
    assert report.iloc[0]["days_to_restate"] == 90


def test_price_load_is_idempotent(con):
    prices = pd.DataFrame({
        "ticker": ["AAPL"] * 3,
        "date": pd.date_range("2020-01-01", periods=3),
        "open": [1.0, 2.0, 3.0], "high": [1.0, 2.0, 3.0],
        "low": [1.0, 2.0, 3.0], "close": [1.0, 2.0, 3.0], "volume": [10.0, 20.0, 30.0],
    })
    assert load_prices(con, prices) == 3
    assert load_prices(con, prices) == 0


def test_null_period_start_does_not_duplicate(con):
    """Instantaneous facts (Assets, Cash) have no period_start. DuckDB treats
    NULLs as distinct in a primary key, so without normalisation every reload
    would insert them again."""
    base = dict(cik=1, ticker="X", concept="Assets", unit="USD",
                period_start=pd.NaT, period_end=pd.Timestamp("2020-12-31"),
                filed_date=pd.Timestamp("2021-02-01"), value=100.0,
                form="10-K", accession="0001", fiscal_year=2020, fiscal_period="FY")
    frame = pd.DataFrame([base])

    assert load_facts(con, frame) == 1
    assert load_facts(con, frame) == 0


def test_blocking_gate_raises(facts):
    """A fact filed before its period ended is a structural failure and must
    stop the pipeline, not warn."""
    broken = facts.copy()
    broken.loc[0, "filed_date"] = pd.Timestamp("2008-01-01")  # before period_end

    gate = quality.gate_filed_after_period_end(broken)
    assert not gate.passed and gate.blocking

    with pytest.raises(quality.QualityGateFailure):
        quality.run_gates([gate])


def test_advisory_gate_does_not_raise(facts):
    gate = quality.gate_restatement_rate(facts, max_pct=1.0)
    assert not gate.passed and not gate.blocking
    quality.run_gates([gate])  # must not raise


def test_membership_size_gate_catches_wrong_table():
    """The failure mode this exists for: a scrape that silently parsed a
    navbox instead of the constituents table, producing well-formed nonsense."""
    tiny = pd.DataFrame({"sample_date": [pd.Timestamp("2020-01-01")] * 12,
                         "ticker": [f"T{i}" for i in range(12)]})
    assert not quality.gate_membership_has_plausible_size(tiny).passed

    ok = pd.DataFrame({"sample_date": [pd.Timestamp("2020-01-01")] * 500,
                       "ticker": [f"T{i}" for i in range(500)]})
    assert quality.gate_membership_has_plausible_size(ok).passed


def test_batch_with_same_day_duplicates_does_not_raise(con):
    """Regression: two rows differing only in the TIME part of a date column
    are distinct in pandas but the same key once stored as DATE. DuckDB raises
    on a conflict inside the inserted batch rather than skipping it, so the
    loader must normalise and dedupe before inserting."""
    membership = pd.DataFrame({
        "sample_date": [pd.Timestamp("2020-03-31 14:23:45"), pd.Timestamp("2020-03-31 00:00:00")],
        "ticker": ["AAPL", "AAPL"],
        "revision_id": [1, 1],
    })
    assert load_membership(con, membership) == 1
    assert members_as_of(con, "2020-04-01") == ["AAPL"]
