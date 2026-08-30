"""Tests for consistent-period fundamentals assembly.

Each test here corresponds to a defect that was live in the published
look-ahead measurement, so they are written as regressions rather than as
general exercises of the API.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import fundamentals
from src.pit import NULL_PERIOD_START


def _fact(ticker, concept, start, end, filed, value, cik=1, form="10-Q", accession="a"):
    return {
        "cik": cik, "ticker": ticker, "concept": concept, "unit": "USD",
        "period_start": pd.Timestamp(start) if start else NULL_PERIOD_START,
        "period_end": pd.Timestamp(end), "filed_date": pd.Timestamp(filed),
        "value": float(value), "form": form, "accession": accession,
    }


def _quarters(ticker, year, values, filed_lag_days=45, cik=1):
    """Four calendar quarters of NetIncomeLoss for one ticker."""
    bounds = [(f"{year}-01-01", f"{year}-03-31"), (f"{year}-04-01", f"{year}-06-30"),
              (f"{year}-07-01", f"{year}-09-30"), (f"{year}-10-01", f"{year}-12-31")]
    return [
        _fact(ticker, "NetIncomeLoss", s, e,
              pd.Timestamp(e) + pd.Timedelta(filed_lag_days, "D"), v,
              cik=cik, accession=f"{ticker}-{year}-{i}")
        for i, ((s, e), v) in enumerate(zip(bounds, values))
    ]


class TestClassifyPeriod:
    def test_labels_each_period_length(self):
        frame = pd.DataFrame([
            _fact("A", "NetIncomeLoss", "2020-01-01", "2020-03-31", "2020-05-01", 1),
            _fact("A", "NetIncomeLoss", "2020-01-01", "2020-06-30", "2020-08-01", 2),
            _fact("A", "NetIncomeLoss", "2020-01-01", "2020-09-30", "2020-11-01", 3),
            _fact("A", "NetIncomeLoss", "2020-01-01", "2020-12-31", "2021-02-01", 4),
            _fact("A", "StockholdersEquity", None, "2020-12-31", "2021-02-01", 5),
        ])
        kinds = fundamentals.classify_period(frame["period_start"], frame["period_end"])
        assert list(kinds) == ["quarter", "half", "ninemonth", "annual", "instant"]

    def test_tolerates_a_53_week_fiscal_year(self):
        frame = pd.DataFrame([
            _fact("A", "NetIncomeLoss", "2019-12-29", "2021-01-03", "2021-02-01", 1),
        ])
        kinds = fundamentals.classify_period(frame["period_start"], frame["period_end"])
        assert list(kinds) == ["annual"]


class TestTtmFlow:
    def test_sums_four_consecutive_quarters(self):
        frame = pd.DataFrame(_quarters("A", 2020, [10, 20, 30, 40]))
        out = fundamentals.ttm_flow(frame, "NetIncomeLoss")
        assert len(out) == 1
        assert out.loc[0, "value"] == pytest.approx(100.0)
        assert out.loc[0, "method"] == "sum4q"
        assert out.loc[0, "window_end"] == pd.Timestamp("2020-12-31")

    def test_rejects_a_gap_in_the_quarters(self):
        """Three of four quarters present must not become a 'TTM'."""
        rows = _quarters("A", 2020, [10, 20, 30, 40])
        del rows[1]
        rows += _quarters("A", 2021, [11, 21, 31, 41])[:1]
        out = fundamentals.ttm_flow(pd.DataFrame(rows), "NetIncomeLoss")
        # The only clean window left ends 2021-03-31 and needs Q2-Q4 2020 plus
        # Q1 2021 -- Q2 2020 is missing, so nothing qualifies.
        assert out.empty

    def test_prefers_the_fresher_window_over_a_stale_annual(self):
        rows = [_fact("A", "NetIncomeLoss", "2019-01-01", "2019-12-31", "2020-02-15", 999,
                      form="10-K", accession="ann")]
        rows += _quarters("A", 2020, [10, 20, 30, 40])
        out = fundamentals.ttm_flow(pd.DataFrame(rows), "NetIncomeLoss")
        assert out.loc[0, "method"] == "sum4q"
        assert out.loc[0, "value"] == pytest.approx(100.0)

    def test_prefers_the_filed_annual_on_an_identical_window_end(self):
        rows = _quarters("A", 2020, [10, 20, 30, 40])
        rows += [_fact("A", "NetIncomeLoss", "2020-01-01", "2020-12-31", "2021-02-15", 101,
                       form="10-K", accession="ann")]
        out = fundamentals.ttm_flow(pd.DataFrame(rows), "NetIncomeLoss")
        assert out.loc[0, "method"] == "annual"
        assert out.loc[0, "value"] == pytest.approx(101.0)

    def test_never_returns_a_window_ending_after_as_of(self):
        """The defect that let a 2022 rebalance read FY2025 net income."""
        rows = _quarters("A", 2020, [10, 20, 30, 40]) + _quarters("A", 2025, [1, 2, 3, 4])
        out = fundamentals.ttm_flow(pd.DataFrame(rows), "NetIncomeLoss", as_of="2021-06-30")
        assert len(out) == 1
        assert out.loc[0, "window_end"] == pd.Timestamp("2020-12-31")
        assert out.loc[0, "value"] == pytest.approx(100.0)

    def test_drops_a_name_whose_only_window_is_stale(self):
        rows = _quarters("A", 2015, [10, 20, 30, 40])
        out = fundamentals.ttm_flow(pd.DataFrame(rows), "NetIncomeLoss",
                                    as_of="2022-06-30", max_staleness_days=400)
        assert out.empty

    def test_every_name_gets_the_same_period_length(self):
        """The core defect: one cross-section must not mix quarters and years.

        `A` files quarterly only; `B`'s freshest fact is an annual one. A naive
        'latest period_end' pick hands back a quarter for A and a year for B.
        """
        rows = _quarters("A", 2020, [10, 10, 10, 10])
        rows += [_fact("B", "NetIncomeLoss", "2020-01-01", "2020-12-31", "2021-02-15", 400,
                       cik=2, form="10-K", accession="b-ann")]
        rows += _quarters("B", 2019, [90, 90, 90, 90], cik=2)
        frame = pd.DataFrame(rows)

        naive = frame.sort_values("period_end").groupby("ticker").last()
        naive_spans = (naive["period_end"] - naive["period_start"]).dt.days
        assert naive_spans["A"] < 130 and naive_spans["B"] > 300, "precondition: naive pick mixes"

        out = fundamentals.ttm_flow(frame, "NetIncomeLoss").set_index("ticker")
        spans = (out["window_end"] - out["window_start"]).dt.days
        assert spans.min() > 340 and spans.max() < 400
        assert out.loc["A", "value"] == pytest.approx(40.0)
        assert out.loc["B", "value"] == pytest.approx(400.0)


class TestInstantStock:
    def test_takes_the_latest_balance_date_at_or_before_as_of(self):
        frame = pd.DataFrame([
            _fact("A", "StockholdersEquity", None, "2020-12-31", "2021-02-01", 100),
            _fact("A", "StockholdersEquity", None, "2021-12-31", "2022-02-01", 200),
            _fact("A", "StockholdersEquity", None, "2025-12-31", "2026-02-01", 900),
        ])
        out = fundamentals.instant_stock(frame, "StockholdersEquity", as_of="2022-06-30")
        assert len(out) == 1
        assert out.loc[0, "value"] == pytest.approx(200.0)


class TestReturnOnEquity:
    def test_uses_a_twelve_month_numerator(self):
        rows = _quarters("A", 2020, [25, 25, 25, 25])
        rows += [_fact("A", "StockholdersEquity", None, "2020-12-31", "2021-02-15", 1000)]
        out = fundamentals.return_on_equity(pd.DataFrame(rows))
        assert out.loc[0, "roe"] == pytest.approx(0.10)

    def test_drops_negative_equity(self):
        rows = _quarters("A", 2020, [25, 25, 25, 25])
        rows += [_fact("A", "StockholdersEquity", None, "2020-12-31", "2021-02-15", -1000)]
        assert fundamentals.return_on_equity(pd.DataFrame(rows)).empty

    def test_winsorizes_a_near_zero_equity_base(self):
        rows = _quarters("A", 2020, [25, 25, 25, 25])
        rows += [_fact("A", "StockholdersEquity", None, "2020-12-31", "2021-02-15", 1)]
        out = fundamentals.return_on_equity(pd.DataFrame(rows), winsorize=1.0)
        assert out.loc[0, "roe"] == pytest.approx(1.0)
