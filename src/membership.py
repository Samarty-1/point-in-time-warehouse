"""Point-in-time S&P 500 membership, and an honest accounting of survivorship bias.

Why this module exists
----------------------
``data_ingestion.fetch_sp500_tickers()`` scrapes the S&P 500 constituent list
from Wikipedia *as it looks today*, and the cross-sectional pipeline then pulls
ten years of history for those names. That is the textbook survivorship-bias
backtest bug, and it is not small here: comparing the index membership as of
2016-08 against 2026-08, **175 of the 505 names then in the index (34.7%) are
no longer in it**. Companies leave the S&P 500 by going bankrupt, collapsing to
a small-cap, or being acquired -- so the names silently dropped are heavily
skewed toward the losers. Ranking the *survivors* against each other and
calling the result an out-of-sample edge quietly assumes you knew in 2016 which
firms would still exist in 2026.

There is a second, opposite bias in the same line of code, easy to miss because
it points the other way: a company that only joined the index in 2024 still has
its 2016-2023 history pulled and fed into the cross-section for those years,
even though it was not an index member then. Stocks get *added* to the S&P 500
after a strong run, so their pre-inclusion history is selected on past
performance too.

What this module can and cannot fix
-----------------------------------
It reconstructs membership as of any past date from the Wikipedia page's own
revision history (free, no API key, and reproducible -- each answer is pinned
to a specific revision id), which fixes the *look-ahead inclusion* bias
completely: a ticker is only allowed into the cross-section on dates it was
genuinely an index member.

It **cannot** fully fix survivorship, and this module does not pretend to.
Yahoo Finance does not serve price history for most delisted tickers -- of a
spot check of eleven names removed from the index since 2016 (CHK, SIVB, FRC,
ANTM, CTXS, ATVI, XLNX, CELG, ABC, ...), nine return no data at all. The bias
is therefore *bounded and reported* rather than eliminated:
:func:`coverage_report` states outright what fraction of true point-in-time
members the price data is missing, so the backtest's headline numbers carry
their own caveat instead of being silently optimistic.

It also fixes a trap that appears the moment you try to fix survivorship
naively. Simply adding the removed tickers to the download list is *worse* than
leaving them out, because **exchanges recycle ticker symbols**. ``BBBY`` was
Bed Bath & Beyond, an index member until 2022; it went bankrupt in 2023, and
Yahoo now serves data for that symbol starting 2026-07-17 -- a completely
unrelated company. Pulling "BBBY" for a 2016-2022 backtest and getting 2026
prices for a different firm is a silent data-integrity failure.
:func:`drop_recycled_tickers` catches exactly this by requiring a ticker's
price history to actually overlap the window in which it was an index member.
"""

from __future__ import annotations

import io
import json
import re
import urllib.parse
import urllib.request
from datetime import date as _date
from pathlib import Path

import pandas as pd

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAGE = "List of S&P 500 companies"
# Wikipedia blocks the default urllib User-Agent; their API policy asks for a
# descriptive one that identifies the client.
USER_AGENT = "point-in-time-warehouse (leighton09lee@gmail.com)"

CACHE_DIR = Path("data/membership_cache")
# Column header for the ticker has been renamed over the years ("Ticker
# symbol" in 2016, "Symbol" today), so match on any of the known spellings
# rather than a fixed name.
_TICKER_COLUMNS = {"symbol", "ticker symbol", "ticker"}
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,6}$")
# The constituents table is the only one on the page with hundreds of rows;
# revisions also contain a "recent changes" table and navboxes.
_MIN_CONSTITUENTS = 400


def _http_get(url: str, timeout: int = 30) -> bytes:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout
    ).read()


def revision_asof(as_of: pd.Timestamp | str) -> tuple[int, str]:
    """Return (revision_id, timestamp) of the last edit to the constituents page
    at or before `as_of`. Pinning to a revision id is what makes membership
    reproducible -- re-running this next year returns the same revision, and
    therefore the same answer, for a given historical date."""
    as_of = pd.Timestamp(as_of)
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": WIKI_PAGE,
        "rvlimit": 1,
        "rvstart": as_of.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rvdir": "older",
        "rvprop": "ids|timestamp",
        "format": "json",
    }
    payload = json.loads(_http_get(f"{WIKI_API}?{urllib.parse.urlencode(params)}"))
    page = next(iter(payload["query"]["pages"].values()))
    revisions = page.get("revisions")
    if not revisions:
        raise RuntimeError(f"No revision of {WIKI_PAGE!r} found at or before {as_of.date()}")
    return int(revisions[0]["revid"]), revisions[0]["timestamp"]


def _parse_constituents(html: bytes) -> set[str]:
    for table in pd.read_html(io.BytesIO(html)):
        for column in table.columns:
            if not isinstance(column, str) or column.strip().lower() not in _TICKER_COLUMNS:
                continue
            symbols = table[column].dropna().astype(str).str.strip()
            symbols = symbols[symbols.map(lambda s: bool(_TICKER_RE.match(s)))]
            if len(symbols) >= _MIN_CONSTITUENTS:
                # yfinance uses '-' where Wikipedia uses '.' for share classes
                # (BRK.B -> BRK-B).
                return set(symbols.str.replace(".", "-", regex=False))
    raise RuntimeError("Could not find the constituents table in that revision")


def members_asof(as_of: pd.Timestamp | str, cache_dir: Path | str = CACHE_DIR) -> set[str]:
    """S&P 500 membership as it stood on `as_of`, from the Wikipedia revision
    current at that date. Cached on disk by revision id -- historical revisions
    are immutable, so a cache hit is always correct, and a 10-year quarterly
    membership history costs ~40 fetches once rather than on every run."""
    cache_dir = Path(cache_dir)
    revid, _ = revision_asof(as_of)
    cache_file = cache_dir / f"sp500_{revid}.json"
    if cache_file.exists():
        return set(json.loads(cache_file.read_text()))

    members = _parse_constituents(_http_get(f"https://en.wikipedia.org/w/index.php?oldid={revid}"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(sorted(members)))
    return members


def membership_history(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str | None = None,
    step: str = "QE",
    cache_dir: Path | str = CACHE_DIR,
) -> pd.DataFrame:
    """Sampled point-in-time membership between `start` and `end`.

    Returns columns: sample_date, ticker. `step` is a pandas offset alias --
    "QE" (quarter-end, the default) matches the S&P's own scheduled
    rebalance cadence and costs ~40 revision fetches for a 10-year window;
    "ME" is more precise about intra-quarter changes at 3x the fetches.

    Sampling means a name added and removed *between* two samples is missed
    entirely. That is a real limitation, but it errs toward the conservative
    side for this use: such a name was in the index only briefly, so excluding
    it loses a little data rather than inventing membership that never existed.
    """
    # normalize() matters: pd.Timestamp.today() carries a time component, so an
    # un-normalized start can land on the same calendar day as a date_range
    # point while comparing unequal -- producing two samples for one date, and
    # a primary-key collision once they are truncated to DATE downstream.
    start = pd.Timestamp(start).normalize()
    end = (pd.Timestamp(end) if end is not None else pd.Timestamp(_date.today())).normalize()
    sample_dates = pd.date_range(start, end, freq=step).union(pd.DatetimeIndex([start, end])).unique()

    rows = []
    for sample_date in sample_dates:
        revid, _ = revision_asof(sample_date)
        for ticker in members_asof(sample_date, cache_dir=cache_dir):
            rows.append({"sample_date": sample_date, "ticker": ticker, "revision_id": revid})
    return pd.DataFrame(rows, columns=["sample_date", "ticker", "revision_id"])


def sampling_interval(history: pd.DataFrame) -> pd.Timedelta:
    """Median gap between consecutive membership samples."""
    samples = pd.Series(sorted(history["sample_date"].unique()))
    if len(samples) < 2:
        return pd.Timedelta(0)
    return samples.diff().dropna().median()


def membership_windows(history: pd.DataFrame) -> pd.DataFrame:
    """Collapse sampled membership into one window per ticker. Used to
    sanity-check price data against membership (see
    :func:`drop_recycled_tickers`).

    Each sample date stands for the whole interval it opens, not for a single
    instant, so `last_member` is extended by one sampling interval. Without
    that extension a ticker seen in exactly one sample gets a zero-length
    window, and any overlap test against it fails by construction -- which
    wrongly flagged legitimate names as recycled (a stock that joined the
    index at the final sample, or left right after the first one).
    """
    windows = (
        history.groupby("ticker")["sample_date"]
        .agg(first_member="min", last_member="max")
        .reset_index()
    )
    windows["last_member"] = windows["last_member"] + sampling_interval(history)
    return windows


def ever_members(history: pd.DataFrame) -> list[str]:
    """Every ticker that was an index member at any sampled date in the window
    -- the correct download list, as opposed to only today's members."""
    return sorted(history["ticker"].unique())


def coverage_report(history: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """How much of the true point-in-time universe the price data actually
    covers. This is the number that bounds the residual survivorship bias, and
    it belongs in the results write-up next to any Sharpe ratio computed on
    this universe -- a backtest missing a third of its historical universe,
    skewed toward the failures, is optimistic by an unknown but non-zero
    amount."""
    universe = set(ever_members(history))
    with_prices = set(prices["ticker"].unique()) & universe
    missing = sorted(universe - with_prices)
    still_current = set(history.loc[history["sample_date"] == history["sample_date"].max(), "ticker"])
    return {
        "n_ever_members": len(universe),
        "n_with_price_data": len(with_prices),
        "n_missing_price_data": len(missing),
        "pct_missing": round(100 * len(missing) / max(len(universe), 1), 1),
        "n_current_members": len(still_current),
        "n_left_index_during_window": len(universe - still_current),
        "missing_tickers": missing,
    }


def drop_recycled_tickers(
    prices: pd.DataFrame, history: pd.DataFrame, min_overlap_days: int = 20
) -> tuple[pd.DataFrame, list[str]]:
    """Remove tickers whose price history doesn't overlap their index
    membership -- i.e. the symbol has been reassigned to a different company.

    The motivating case: BBBY (Bed Bath & Beyond) was an index member until
    2022 and went bankrupt in 2023, but Yahoo serves ~30 rows for that symbol
    beginning 2026-07-17, belonging to an unrelated firm. Without this check,
    naively widening the download list to include removed names -- the obvious
    "fix" for survivorship bias -- silently injects one company's prices under
    another company's identity, which is a worse error than the bias it was
    meant to cure.

    Returns the filtered prices and the list of dropped tickers.
    """
    windows = membership_windows(history).set_index("ticker")
    span = prices.groupby("ticker")["date"].agg(["min", "max"])
    # pd.Timedelta(days=N) emits a NumPy "generic unit" DeprecationWarning
    # under pandas 2.3 + numpy 2.5; the explicit unit= form does not.
    required = pd.Timedelta(min_overlap_days, unit="D")

    dropped = []
    for ticker, row in span.iterrows():
        if ticker not in windows.index:
            continue
        member_from = pd.Timestamp(windows.loc[ticker, "first_member"])
        member_to = pd.Timestamp(windows.loc[ticker, "last_member"])
        overlap = pd.Timestamp(min(row["max"], member_to)) - pd.Timestamp(max(row["min"], member_from))
        # A ticker that was only ever an index member for a short window can
        # never show `required` days of overlap, so hold it to its own window
        # length instead. Only a genuinely disjoint price history -- the
        # recycled-symbol case -- should be dropped.
        threshold = min(required, member_to - member_from)
        if overlap <= pd.Timedelta(0) or overlap < threshold:
            dropped.append(ticker)

    if not dropped:
        return prices, []
    return prices[~prices["ticker"].isin(dropped)].reset_index(drop=True), sorted(dropped)


def apply_point_in_time_membership(
    df: pd.DataFrame, history: pd.DataFrame, date_column: str = "date"
) -> pd.DataFrame:
    """Keep only rows where the ticker was actually an index member on that date.

    Sampled membership is stepped forward to daily with a merge_asof on the
    most recent sample at or before each row's date -- i.e. membership is held
    from the sample that established it until the next sample, never
    interpolated forward from a *future* sample (that would be exactly the
    look-ahead this function exists to remove).
    """
    if df.empty:
        return df

    samples = sorted(history["sample_date"].unique())
    sample_frame = pd.DataFrame({"sample_date": samples}).sort_values("sample_date")
    rows = df[[date_column]].drop_duplicates().sort_values(date_column)
    mapped = pd.merge_asof(
        rows, sample_frame, left_on=date_column, right_on="sample_date", direction="backward"
    )
    # Rows before the first sample have no established membership; fall back to
    # the earliest sample rather than dropping them outright.
    mapped["sample_date"] = mapped["sample_date"].fillna(samples[0])

    keyed = df.merge(mapped, on=date_column, how="left")
    member_pairs = set(zip(history["sample_date"], history["ticker"]))
    is_member = [
        (sample_date, ticker) in member_pairs
        for sample_date, ticker in zip(keyed["sample_date"], keyed["ticker"])
    ]
    return keyed[pd.Series(is_member, index=keyed.index)].drop(columns=["sample_date"]).reset_index(drop=True)
