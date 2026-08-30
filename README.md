# Point-in-Time Financial Data Warehouse

A bitemporal market-data warehouse: it stores not just what the numbers are,
but **what they were believed to be on any given date**. Built to make one
specific, invisible backtest bias measurable — and then to measure it.

The answer: using restated fundamentals instead of point-in-time ones inflated
a plain profitability strategy's Sharpe from **0.17 to 0.27**.

> **Correction (2026-08-30).** This README previously reported 0.33 → 0.91.
> That measurement was wrong, in the strategy's favour, for two reasons found
> by auditing it — a period-length confound and an unbounded forward reach.
> Both are described in [What the bias is worth](#what-the-bias-is-worth). The
> corrected effect is real but roughly **six times smaller** than the headline
> this repo used to lead with.

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
| Total return | 15.3% | 32.7% | +17.3pp |
| Annualized return | 1.5% | 2.9% | +1.5pp |
| **Sharpe** | **0.17** | **0.27** | **+0.09** |
| Hit rate | 64.1% | 66.7% | +2.6pp |

Same code, same universe, same dates, same strategy. The only difference is
which vintage of an **already-ended** period each arm sees. Restated data does
flatter the strategy, consistently and in the direction theory predicts — but
by about a tenth of a Sharpe point, not by the 0.58 this repo used to claim.

The restatement profile is where it comes from: **6.6% of
company-concept-periods were revised after first publication**, and the median
lag before the correction appeared was **464 days**. For well over a year, each
of those figures was something other than what the restated feed reports. That
this only buys a tenth of a Sharpe point is itself the finding: the revisions
are frequent but small (median 0.7%), and a quarterly-rebalanced quintile book
is not very sensitive to them.

### The two defects that produced 0.91

Both were in the measurement, not the warehouse, and both inflated the gap.

**1. The arms ranked on different period lengths.** XBRL tags
`NetIncomeLoss` over quarters, halves, nine-month spans and years, all in the
same filing. `roe_signal` took the fact with the latest `period_end` — which
picks a *different period length per company*, depending on the filer's tagging
habits. Measured on this warehouse:

| view | quarter | half | nine-month | annual |
|---|---|---|---|---|
| restated | 31 | 41 | 9 | 13 |
| point-in-time @2022-06-30 | 75 | 4 | 8 | 5 |

Names carrying an annual numerator showed a median ROE of **0.109** against
**0.039** for names carrying a quarterly one — a 2.8x gap that is pure period
length. Because tagging habits are stable per filer, the book was persistently
long "companies whose latest tagged figure is annual" and short "companies whose
latest tagged figure is quarterly". And because the two arms land on *different*
mixes, the gap between them was part look-ahead and part artifact.

**2. The restated arm read the future, not just the wrong vintage.** `latest()`
returns every period the warehouse ever learned about, including periods that
had not ended on the rebalance date. Taking "the latest period" out of that
handed a 2022 rebalance **Apple's FY2025 net income**. That is a multi-year
look-ahead — a far cruder thing than the restatement effect being measured, and
large enough to swamp it.

The fix is [`src/fundamentals.py`](src/fundamentals.py): one trailing-twelve-
month window per name, assembled identically for everyone (four consecutive
quarters, or a tagged annual figure — whichever ends later), capped at the
rebalance date, with a staleness floor. Names that cannot produce a clean
twelve-month window are dropped rather than approximated. The arms now differ
*only* in which vintage of an already-ended period they see, which is the one
difference the measurement is supposed to isolate.

`measure_lookahead` prints the naive period-length mix on every run, so a
regression is visible rather than silent.

## Three data-quality findings worth reading

All were found by running the thing, not by reasoning about it.

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

**...and the power-of-ten test was too narrow to catch them all.** The
paragraph above was written, the filter shipped, and
`max_abs_pct_change` still read **331,176,570%** — the exact statistic the
filter existed to protect. Amcor's Q1 2019 net income arrives as **−34 USD** in
the 10-Q and **112,600,000 USD** in the 10-K. The ratio is 3.3 million and the
sign flips, so it is neither a clean power of ten nor same-sign, and it walked
straight through. Only 4 of 973 value changes exceed 100x, and *all four* are
this kind of thing — a company does not revise its net income by two orders of
magnitude. `pit._classify_revision` now sorts every change into
`power_of_ten_rescale`, `implausible_ratio` (≥100x, either sign), or
`economic`, and the headline uses only the last. `max_abs_pct_change` is now
**1,621%**, and the excluded counts are reported rather than swallowed.

The general lesson, which cost this repo its headline number twice: a filter
that removes a known artifact is not evidence the artifact is gone. Check the
statistic it was protecting.

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
pytest tests/ -q                  # 49 tests
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
src/fundamentals.py consistent-period assembly: ttm_flow / instant_stock / return_on_equity
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
  point-in-time Sharpe of 0.17 should not be read as an edge.
