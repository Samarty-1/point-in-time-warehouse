"""Bitemporal financial facts from SEC EDGAR's XBRL API.

Why bitemporal
--------------
Every commercial and free fundamentals feed I have used answers one question:
"what is Apple's FY2008 net income?" The answer they give is **$6.119
billion**. That is the restated figure, filed in a 10-K/A on 2010-01-25.

What Apple actually reported in its original FY2008 10-K, on 2009-10-27, was
**$4.834 billion** — 26.6% lower. Anyone trading on fundamentals in November
2009 saw $4.834B, because the restatement did not exist yet.

A backtest that uses the restated number is using information that was not
available on the date it claims to trade. This is a look-ahead bias, it is
invisible in the data (nothing is missing or malformed — the number is simply
the *wrong vintage*), and essentially no free dataset lets you avoid it.

So facts here carry **two** time dimensions:

* **valid time** — the period the fact describes (`period_start`, `period_end`)
* **transaction time** — when the fact became public (`filed_date`)

which is what makes ``as_of(date)`` answerable: for each concept and period,
the most recently filed value *whose filing date is on or before* `date`.
That is the definition of point-in-time correctness, and it is the whole
reason this warehouse exists.

Source
------
EDGAR's XBRL ``companyconcept`` API. Free, no key, no rate-limit tier — but
the SEC requires a descriptive User-Agent with contact information and asks
for no more than 10 requests/second. Both are honoured here.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{concept}.json"

# The SEC's fair-access policy requires a User-Agent that identifies the
# requester. Requests without one are blocked.
DEFAULT_USER_AGENT = "point-in-time-warehouse (leighton09lee@gmail.com)"
MIN_REQUEST_INTERVAL = 0.11  # ~9 req/s, under the SEC's stated 10/s ceiling

# Periodic reports only. 8-K and S-1 carry XBRL facts too, but mixing filing
# types makes "the value as most recently reported" ambiguous.
PERIODIC_FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "40-F")

DEFAULT_CONCEPTS = [
    "Revenues",
    "NetIncomeLoss",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
    "OperatingIncomeLoss",
    "EarningsPerShareDiluted",
]

FACT_COLUMNS = [
    "cik", "ticker", "concept", "unit", "period_start", "period_end",
    "filed_date", "value", "form", "accession", "fiscal_year", "fiscal_period",
]


@dataclass
class EdgarClient:
    """Minimal EDGAR client with on-disk caching and polite rate limiting."""

    user_agent: str = DEFAULT_USER_AGENT
    cache_dir: Path = Path("data/edgar_cache")
    _last_request: float = 0.0

    def _get(self, url: str, timeout: int = 60) -> bytes:
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            payload = urllib.request.urlopen(request, timeout=timeout).read()
        finally:
            self._last_request = time.monotonic()
        return payload

    def ticker_to_cik(self) -> dict[str, int]:
        """Ticker -> CIK map. Cached; the SEC updates it roughly daily."""
        cache = Path(self.cache_dir) / "company_tickers.json"
        if cache.exists():
            raw = json.loads(cache.read_text())
        else:
            raw = json.loads(self._get(SEC_TICKERS_URL))
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw))
        return {entry["ticker"].upper(): int(entry["cik_str"]) for entry in raw.values()}

    def company_concept(self, cik: int, concept: str) -> dict | None:
        """Raw XBRL history for one concept. Returns None when the company has
        never reported it — common and not an error, since not every filer
        tags every us-gaap concept."""
        cache = Path(self.cache_dir) / f"CIK{cik:010d}_{concept}.json"
        if cache.exists():
            text = cache.read_text()
            return json.loads(text) if text else None

        try:
            payload = self._get(CONCEPT_URL.format(cik=cik, concept=concept))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text("")  # negative cache: don't re-ask every run
                return None
            raise

        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
        return json.loads(payload)


def parse_concept(payload: dict, ticker: str, concept: str) -> pd.DataFrame:
    """Flatten one companyconcept response into bitemporal fact rows.

    Each XBRL fact already carries `filed`, which is the transaction-time
    dimension the whole warehouse depends on. Facts for the same period from
    different filings are kept as SEPARATE ROWS rather than deduplicated --
    that redundancy is the restatement history, and collapsing it to "the
    latest value" is exactly the look-ahead this module exists to prevent.
    """
    if not payload or "units" not in payload:
        return pd.DataFrame(columns=FACT_COLUMNS)

    cik = int(payload.get("cik", 0))
    rows = []
    for unit, facts in payload["units"].items():
        for fact in facts:
            if fact.get("form") not in PERIODIC_FORMS or fact.get("filed") is None:
                continue
            rows.append(
                {
                    "cik": cik,
                    "ticker": ticker,
                    "concept": concept,
                    "unit": unit,
                    "period_start": pd.to_datetime(fact.get("start"), errors="coerce"),
                    "period_end": pd.to_datetime(fact.get("end"), errors="coerce"),
                    "filed_date": pd.to_datetime(fact["filed"]),
                    "value": float(fact["val"]),
                    "form": fact["form"],
                    "accession": fact.get("accn", ""),
                    "fiscal_year": fact.get("fy"),
                    "fiscal_period": fact.get("fp"),
                }
            )

    frame = pd.DataFrame(rows, columns=FACT_COLUMNS)
    if frame.empty:
        return frame
    return frame.dropna(subset=["period_end"]).reset_index(drop=True)


def fetch_facts(
    tickers: list[str],
    concepts: list[str] | None = None,
    client: EdgarClient | None = None,
) -> pd.DataFrame:
    """Bitemporal fact table for a set of tickers."""
    client = client or EdgarClient()
    concepts = concepts or DEFAULT_CONCEPTS
    mapping = client.ticker_to_cik()

    frames = []
    for ticker in tickers:
        cik = mapping.get(ticker.upper())
        if cik is None:
            continue
        for concept in concepts:
            payload = client.company_concept(cik, concept)
            if payload is None:
                continue
            frames.append(parse_concept(payload, ticker.upper(), concept))

    if not frames:
        return pd.DataFrame(columns=FACT_COLUMNS)
    return pd.concat(frames, ignore_index=True)
