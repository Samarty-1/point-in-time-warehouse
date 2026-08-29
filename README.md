# Point-in-Time Financial Data Warehouse

A bitemporal market-data warehouse: it stores not just what the numbers are,
but **what they were believed to be on any given date**. Built to make one
specific, invisible backtest bias measurable — and then to measure it.

The answer: using restated fundamentals instead of point-in-time ones inflated
a plain profitability strategy's Sharpe from **0.33 to 0.91**.

## The bias

Ask any fundamentals feed for Apple's FY2008 net income and it will tell you
**$6.119 billion**. That is the restated figure, filed in a 10-K/A on
2010-01-25.

What Apple actually reported, in its original FY2008 10-K on 2009-10-27, was
**$4.834 billion** — 26.6% lower. Anyone trading on fundamentals in November
2009 saw $4.834B, because the restatement did not exist yet.

A backtest using the restated number is trading on information nobody had. It
is a look-ahead bias, and it is **invisible**: nothing is missing, nothing is
malformed, the number is simply the wrong vintage. No amount of staring at the
data reveals it, and essentially no free dataset lets you avoid it.

So every fact here carries two time dimensions:

| dimension | column | meaning |
|---|---|---|
| valid time | `period_start`, `period_end` | the period the fact describes |
| transaction time | `filed_date` | when the fact became public |

which makes `as_of(date)` answerable: for each concept and period, the most
recently filed value *whose filing date is on or before* `date`.

## What the bias is worth

`python -m scripts.measure_lookahead` runs the same strategy twice — rank the
universe by return on equity, long the top quintile, short the bottom,
rebalance quarterly — changing only which vintage of fundamentals it sees.

120 S&P constituents, 37,169 bitemporal facts, 39 quarterly rebalances,
2016–2026.

| | point-in-time | restated | look-ahead |
|---|---|---|---|
| Total return | 33.5% | 122.4% | +88.9pp |
| Annualized return | 3.0% | 8.5% | +5.5pp |
| **Sharpe** | **0.33** | **0.91** | **+0.58** |
| Hit rate | 69.2% | 66.7% | −2.6pp |

Same code, same universe, same dates, same strategy. The only difference is
whether the fundamentals were knowable at the time. **Restated data made a
weak strategy look like a decent one** — a 2.8x Sharpe inflation, which is the
size of the effect sitting silently inside most fundamental backtests.

The restatement profile explains where it comes from: **6.6% of
company-concept-periods were revised after first publication**, and the median
lag before the correction appeared was **464 days**. For well over a year,
each of those figures was something other than what the restated feed reports.

## Two data-quality findings worth reading

Both were found by running the thing, not by reasoning about it.

**XBRL reporting-scale changes masquerade as restatements.** Atmos Energy's
FY2010 net income appears as 73,997 in one filing and 73,997,000 in a later
one — exactly 1000x. The company changed its XBRL scale attribute from
thousands to units. Nothing economic happened. Undetected, these dominate every
restatement statistic (they *are* the right tail), and worse, they corrupt any
ratio built from facts of different vintages: an ROE with net income in
thousands over equity in units is off by three orders of magnitude and reads as
a spectacular outlier rather than an error. `pit._is_scale_change` flags
same-sign changes whose ratio is within tolerance of a power of ten;
`restatement_summary` excludes them and reports the count separately. A
sign-flip — a loss becoming a profit, the most material restatement there is —
is explicitly never filtered.

**Balance-sheet facts were silently invisible.** Instantaneous concepts
(`Assets`, `StockholdersEquity`, `Cash`) describe a point rather than a span,
so XBRL gives them no `start` and they arrive with `period_start = NaT`. NaT
never equals NaT in a merge or a groupby key — so every balance-sheet fact
vanished from restatement detection, with no error raised and the code looking
like it worked. Normalising to a sentinel date fixes it, and it is the same
reason the DuckDB loader normalises before insert (a `NULL` is distinct in a
primary key, so reloads would duplicate forever).

## Warehouse design

**Idempotent loads.** Facts are keyed on
`(cik, concept, unit, period_start, period_end, accession)` — an accession
number identifies a specific filing, so the same fact from the same filing is
the same row however many times the pipeline sees it. Re-runs and overlapping
backfills converge instead of duplicating, which is what lets the loader be
retried after a partial failure with no cleanup.

**Append-only history.** A restatement is a *new row* with a later
`filed_date`, never an overwrite. Losing the superseded value would destroy
exactly the information the warehouse exists to preserve.

**Point-in-time in SQL, not just pandas.** `facts_as_of` pushes the window
function into DuckDB so it works on a warehouse larger than memory, and a test
asserts the SQL and pandas implementations never disagree — two
implementations of the same semantics that diverge is worse than one, because
whichever a caller happens to use decides whether their backtest leaks.

**Survivorship-free universe.** Index membership is reconstructed from the
Wikipedia page's own revision history via the MediaWiki API — free, no key,
and pinned to immutable revision ids, so a given historical date always
returns the same answer. Members are stored as sampled observations rather
than inferred add/drop dates, because quarterly samples do not support that
precision and pretending otherwise would be fabricating it.

**Quality gates that fail the pipeline.** Blocking gates raise; advisory ones
warn. A check that logs and continues is a check nobody reads. On the first
real run, `prices_positive` caught 98 malformed rows and stopped the load —
fixed by cleaning at the source rather than by relaxing the gate, because a
gate that tolerates its own failure mode is decoration.

The most important gate is `filed_after_period_end`: a fact cannot be filed
before the period it reports on has ended. A violation would make future
information visible to `as_of`, which is the one thing this project exists to
prevent.

## Usage

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

python pipeline.py --n-tickers 120 --period 10y
python -m scripts.measure_lookahead
pytest tests/ -q                  # 34 tests
```

```python
from src.warehouse import connect, facts_as_of
con = connect("data/warehouse.duckdb")

facts_as_of(con, "2009-11-15")    # only what was filed by then
```

```
[1] point-in-time index membership
    21,171 membership rows over 42 samples
[2] prices
    dropped 98 price rows with missing/non-positive close
    241,691 price rows
[3] fundamentals from EDGAR (bitemporal)
    37,169 facts across 96 tickers
[4] quality gates
  [PASS] no_future_filings            0 facts filed after 2026-08-29
  [PASS] filed_after_period_end       0 facts filed BEFORE the period they report ended
  [PASS] no_duplicate_facts           0 duplicate facts
  [PASS] membership_plausible_size    0 membership samples outside 425-575 names
  [PASS] prices_positive              0 rows with non-positive or missing close
  [PASS] restatement_rate             6.6% of periods restated (advisory threshold 25.0%)
  [PASS] price_coverage               85.0% of the universe has price data; 18 names missing
```

## Layout

```
src/edgar.py        SEC XBRL client, bitemporal fact parsing, on-disk cache
src/pit.py          as_of / latest / first_reported / restatements / lookahead_exposure
src/warehouse.py    DuckDB schema, idempotent loaders, SQL point-in-time queries
src/membership.py   point-in-time index membership from Wikipedia revisions
src/quality.py      blocking and advisory data quality gates
pipeline.py         membership -> prices -> fundamentals -> gates
scripts/measure_lookahead.py    the measurement above
```

## Limits, stated plainly

- **Survivorship bias is bounded, not eliminated.** Free price sources do not
  serve most delisted tickers; the run above reaches 85% coverage of the true
  point-in-time universe, and `gate_price_coverage` reports the shortfall
  rather than hiding it. The missing names are disproportionately the
  failures, so results here remain somewhat optimistic.
- **Membership is sampled quarterly.** A name added and removed between two
  samples is missed. That errs toward excluding a brief member rather than
  inventing membership that never existed.
- **`max_abs_pct_change` is not a useful statistic** on this data — it is
  driven by revisions off a near-zero base (a −34 net income becoming
  $112.6M). Read the median and p99 instead; both are reported.
- **EDGAR coverage starts around 2009**, when XBRL tagging became mandatory.
  Earlier fundamentals are not available through this route at all.
- The strategy in `measure_lookahead` is deliberately plain. It is a
  measuring instrument for the data vintage, not a proposed strategy, and its
  point-in-time Sharpe of 0.33 should not be read as an edge.
