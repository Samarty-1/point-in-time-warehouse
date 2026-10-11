"""Tests for point-in-time correctness.

These are the tests the whole warehouse rests on. If `as_of` is wrong, every
backtest built on this data is silently using the future, and nothing about
the output would look wrong.

The fixture is modelled on a real case: Apple's FY2008 net income was
reported as $4.834B in the original 10-K (filed 2009-10-27) and restated to
$6.119B in a 10-K/A (filed 2010-01-25) -- a 26.6% revision that no
conventional fundamentals feed will show you.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.pit import (
    as_of,
    first_reported,
    latest,
    lookahead_exposure,
    restatement_summary,
    restatements,
)

ORIGINAL_FILED = pd.Timestamp("2009-10-27")
RESTATED_FILED = pd.Timestamp("2010-01-25")


@pytest.fixture
def facts() -> pd.DataFrame:
    rows = [
        # FY2008 net income: reported, then restated 90 days later.
        dict(cik=320193, ticker="AAPL", concept="NetIncomeLoss", unit="USD",
             period_start=pd.Timestamp("2007-09-30"), period_end=pd.Timestamp("2008-09-27"),
             filed_date=ORIGINAL_FILED, value=4_834_000_000.0, form="10-K",
             accession="0001", fiscal_year=2009, fiscal_period="FY"),
        dict(cik=320193, ticker="AAPL", concept="NetIncomeLoss", unit="USD",
             period_start=pd.Timestamp("2007-09-30"), period_end=pd.Timestamp("2008-09-27"),
             filed_date=RESTATED_FILED, value=6_119_000_000.0, form="10-K/A",
             accession="0002", fiscal_year=2009, fiscal_period="FY"),
        # A later period, never restated, filed after both of the above.
        dict(cik=320193, ticker="AAPL", concept="NetIncomeLoss", unit="USD",
             period_start=pd.Timestamp("2008-09-28"), period_end=pd.Timestamp("2009-09-26"),
             filed_date=pd.Timestamp("2010-10-27"), value=8_235_000_000.0, form="10-K",
             accession="0003", fiscal_year=2010, fiscal_period="FY"),
    ]
    return pd.DataFrame(rows)


def test_as_of_returns_the_original_value_before_restatement(facts):
    """The number an investor could actually see in November 2009."""
    view = as_of(facts, "2009-11-15")
    fy2008 = view[view["period_end"] == pd.Timestamp("2008-09-27")]

    assert len(fy2008) == 1
    assert fy2008.iloc[0]["value"] == 4_834_000_000.0


def test_as_of_returns_the_restated_value_after_restatement(facts):
    view = as_of(facts, "2010-06-01")
    fy2008 = view[view["period_end"] == pd.Timestamp("2008-09-27")]
    assert fy2008.iloc[0]["value"] == 6_119_000_000.0


def test_as_of_hides_periods_not_yet_filed(facts):
    """The row must not EXIST, not merely carry an old value. Filtering the
    value but keeping the row is the subtle version of this bug."""
    view = as_of(facts, "2009-11-15")
    assert (view["period_end"] == pd.Timestamp("2009-09-26")).sum() == 0
    assert len(view) == 1


def test_as_of_before_any_filing_is_empty(facts):
    assert as_of(facts, "2000-01-01").empty


def test_as_of_on_the_filing_date_itself_includes_it(facts):
    """Boundary: a filing is knowable on the day it is filed, not the day
    after. An off-by-one here shifts every fact by a day."""
    view = as_of(facts, ORIGINAL_FILED)
    assert len(view) == 1
    assert view.iloc[0]["value"] == 4_834_000_000.0


def test_as_of_day_before_restatement_still_original(facts):
    view = as_of(facts, RESTATED_FILED - pd.Timedelta(1, unit="D"))
    assert view[view["period_end"] == pd.Timestamp("2008-09-27")].iloc[0]["value"] == 4_834_000_000.0


def test_latest_is_the_restated_view(facts):
    """What a conventional fundamentals feed gives you -- and why it is
    unusable for a backtest dated before 2010-01-25."""
    view = latest(facts)
    fy2008 = view[view["period_end"] == pd.Timestamp("2008-09-27")]
    assert fy2008.iloc[0]["value"] == 6_119_000_000.0
    assert len(view) == 2  # one row per period, regardless of filing count


def test_first_reported_is_the_original_view(facts):
    view = first_reported(facts)
    fy2008 = view[view["period_end"] == pd.Timestamp("2008-09-27")]
    assert fy2008.iloc[0]["value"] == 4_834_000_000.0


def test_restatements_measures_the_revision(facts):
    changed = restatements(facts)

    assert len(changed) == 1
    row = changed.iloc[0]
    assert row["original_value"] == 4_834_000_000.0
    assert row["final_value"] == 6_119_000_000.0
    assert row["pct_change"] == pytest.approx(26.58, abs=0.01)
    assert row["days_to_restate"] == 90
    assert row["n_filings"] == 2


def test_restatements_can_filter_by_materiality(facts):
    assert len(restatements(facts, min_pct_change=5.0)) == 1
    assert len(restatements(facts, min_pct_change=50.0)) == 0


def test_restatement_summary(facts):
    summary = restatement_summary(facts)
    assert summary["n_company_concept_periods"] == 2
    assert summary["n_restated"] == 1
    assert summary["pct_restated"] == 50.0
    assert summary["n_restated_over_5pct"] == 1


def test_lookahead_exposure_names_both_failure_modes(facts):
    """On 2009-11-15 a restated-data backtest is cheating twice: using a value
    that would not be revised for 90 days, and using a period not yet filed."""
    exposed = lookahead_exposure(facts, "2009-11-15")

    issues = set(exposed["issue"])
    assert issues == {"not yet filed", "value later revised"}
    assert len(exposed) == 2


def test_no_lookahead_exposure_once_everything_is_filed(facts):
    assert lookahead_exposure(facts, "2011-01-01").empty


def test_same_day_amendment_prefers_the_amendment():
    """Tie-break: an amendment filed the same day as the original must win,
    otherwise `as_of` returns the superseded figure."""
    base = dict(cik=1, ticker="X", concept="NetIncomeLoss", unit="USD",
                period_start=pd.Timestamp("2020-01-01"), period_end=pd.Timestamp("2020-12-31"),
                filed_date=pd.Timestamp("2021-02-01"), fiscal_year=2020, fiscal_period="FY")
    facts = pd.DataFrame([
        {**base, "value": 100.0, "form": "10-K", "accession": "0001"},
        {**base, "value": 120.0, "form": "10-K/A", "accession": "0002"},
    ])
    assert as_of(facts, "2021-03-01").iloc[0]["value"] == 120.0


def test_empty_input_is_handled_everywhere():
    empty = pd.DataFrame(columns=[
        "cik", "ticker", "concept", "unit", "period_start", "period_end",
        "filed_date", "value", "form", "accession", "fiscal_year", "fiscal_period",
    ])
    assert as_of(empty, "2020-01-01").empty
    assert latest(empty).empty
    assert restatements(empty).empty


def test_instantaneous_facts_are_not_silently_dropped():
    """Regression: balance-sheet concepts have no period_start, so they arrive
    with NaT. NaT never equals NaT in a merge or groupby key, so every
    Assets / StockholdersEquity / Cash fact used to vanish from restatement
    detection without any error being raised -- the entire balance sheet was
    invisible to the audit while the code looked like it worked.
    """
    base = dict(cik=1, ticker="X", concept="StockholdersEquity", unit="USD",
                period_start=pd.NaT, period_end=pd.Timestamp("2020-12-31"),
                fiscal_year=2020, fiscal_period="FY")
    facts = pd.DataFrame([
        {**base, "filed_date": pd.Timestamp("2021-02-01"), "value": 100.0,
         "form": "10-K", "accession": "0001"},
        {**base, "filed_date": pd.Timestamp("2021-08-01"), "value": 130.0,
         "form": "10-K/A", "accession": "0002"},
    ])

    changed = restatements(facts)
    assert len(changed) == 1, "instantaneous facts must survive the merge"
    assert changed.iloc[0]["pct_change"] == pytest.approx(30.0)

    assert as_of(facts, "2021-03-01").iloc[0]["value"] == 100.0
    assert as_of(facts, "2021-09-01").iloc[0]["value"] == 130.0


def test_mixed_duration_and_instantaneous_facts_coexist():
    """Both kinds in one frame must produce two distinct rows, not collapse
    into one via the sentinel."""
    common = dict(cik=1, ticker="X", unit="USD", period_end=pd.Timestamp("2020-12-31"),
                  filed_date=pd.Timestamp("2021-02-01"), form="10-K", accession="0001",
                  fiscal_year=2020, fiscal_period="FY")
    facts = pd.DataFrame([
        {**common, "concept": "NetIncomeLoss", "period_start": pd.Timestamp("2020-01-01"), "value": 10.0},
        {**common, "concept": "StockholdersEquity", "period_start": pd.NaT, "value": 200.0},
    ])
    assert len(latest(facts)) == 2


def test_scale_change_is_not_counted_as_a_restatement():
    """Regression: filers change the XBRL reporting scale between filings, so
    a value goes from 73,997 (thousands) to 73,997,000 (units). That is a
    99,900% apparent revision and zero economic change. Atmos Energy's FY2010
    net income does exactly this. Undetected, these artifacts are the entire
    right tail of every restatement statistic.
    """
    base = dict(cik=1, ticker="ATO", concept="NetIncomeLoss", unit="USD",
                period_start=pd.Timestamp("2010-01-01"), period_end=pd.Timestamp("2010-12-31"),
                fiscal_year=2010, fiscal_period="FY")
    facts = pd.DataFrame([
        {**base, "filed_date": pd.Timestamp("2011-02-01"), "value": 73_997.0,
         "form": "10-K", "accession": "0001"},
        {**base, "filed_date": pd.Timestamp("2012-11-01"), "value": 73_997_000.0,
         "form": "10-K", "accession": "0002"},
    ])

    changed = restatements(facts)
    assert len(changed) == 1
    assert bool(changed.iloc[0]["is_scale_artifact"]) is True

    summary = restatement_summary(facts)
    assert summary["n_restated"] == 0
    assert summary["n_scale_artifacts_excluded"] == 1


def test_genuine_revision_is_not_flagged_as_a_scale_change():
    """Apple's 26.6% FY2008 revision must survive the artifact filter."""
    base = dict(cik=320193, ticker="AAPL", concept="NetIncomeLoss", unit="USD",
                period_start=pd.Timestamp("2007-09-30"), period_end=pd.Timestamp("2008-09-27"),
                fiscal_year=2009, fiscal_period="FY")
    facts = pd.DataFrame([
        {**base, "filed_date": ORIGINAL_FILED, "value": 4_834_000_000.0,
         "form": "10-K", "accession": "0001"},
        {**base, "filed_date": RESTATED_FILED, "value": 6_119_000_000.0,
         "form": "10-K/A", "accession": "0002"},
    ])
    changed = restatements(facts)
    assert bool(changed.iloc[0]["is_scale_artifact"]) is False
    assert restatement_summary(facts)["n_restated"] == 1


def test_sign_flip_is_not_a_scale_change():
    """A loss becoming a profit is the most material restatement there is and
    must never be filtered out, whatever the magnitude ratio."""
    base = dict(cik=1, ticker="X", concept="NetIncomeLoss", unit="USD",
                period_start=pd.Timestamp("2020-01-01"), period_end=pd.Timestamp("2020-12-31"),
                fiscal_year=2020, fiscal_period="FY")
    facts = pd.DataFrame([
        {**base, "filed_date": pd.Timestamp("2021-02-01"), "value": -100.0,
         "form": "10-K", "accession": "0001"},
        {**base, "filed_date": pd.Timestamp("2021-08-01"), "value": 1000.0,
         "form": "10-K/A", "accession": "0002"},
    ])
    changed = restatements(facts)
    assert bool(changed.iloc[0]["is_scale_artifact"]) is False
    # A 10x sign-flipping revision is material and must stay in the headline.
    assert changed.iloc[0]["revision_class"] == "economic"
    assert restatement_summary(facts)["n_restated"] == 1


def test_implausible_ratio_is_excluded_from_the_headline():
    """Regression: the 331,176,570% "restatement" that survived the filter.

    Amcor's Q1 2019 net income arrives as -34 USD in the 10-Q and as
    112,600,000 USD in the 10-K. It is not a power-of-ten rescale -- the ratio
    is 3.3 million and the sign flips -- so the exact-rescale test passed it
    through, and it landed in max_abs_pct_change: the very statistic the
    artifact filter existed to protect.
    """
    base = dict(cik=1, ticker="AMCR", concept="NetIncomeLoss", unit="USD",
                period_start=pd.Timestamp("2019-01-01"), period_end=pd.Timestamp("2019-03-31"),
                fiscal_year=2019, fiscal_period="Q1")
    facts = pd.DataFrame([
        {**base, "filed_date": pd.Timestamp("2019-05-09"), "value": -34.0,
         "form": "10-Q", "accession": "0001"},
        {**base, "filed_date": pd.Timestamp("2019-09-03"), "value": 112_600_000.0,
         "form": "10-K", "accession": "0002"},
    ])

    changed = restatements(facts)
    assert changed.iloc[0]["revision_class"] == "implausible_ratio"
    assert bool(changed.iloc[0]["is_scale_artifact"]) is False

    summary = restatement_summary(facts)
    assert summary["n_restated"] == 0
    assert summary["n_implausible_ratio_excluded"] == 1
    assert summary["max_abs_pct_change"] == 0.0


def test_a_large_but_plausible_revision_stays_in_the_headline():
    """The ratio gate must not swallow real revisions. A 50x move off a small
    base is unusual but reportable; only 100x and beyond is treated as bad
    data."""
    base = dict(cik=1, ticker="X", concept="NetIncomeLoss", unit="USD",
                period_start=pd.Timestamp("2020-01-01"), period_end=pd.Timestamp("2020-12-31"),
                fiscal_year=2020, fiscal_period="FY")
    facts = pd.DataFrame([
        {**base, "filed_date": pd.Timestamp("2021-02-01"), "value": 1_000_000.0,
         "form": "10-K", "accession": "0001"},
        {**base, "filed_date": pd.Timestamp("2021-08-01"), "value": 50_000_000.0,
         "form": "10-K/A", "accession": "0002"},
    ])
    assert restatements(facts).iloc[0]["revision_class"] == "economic"
    assert restatement_summary(facts)["n_restated"] == 1
