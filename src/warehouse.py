"""DuckDB storage for the bitemporal warehouse.

Design notes worth stating, because they are the parts that make this a
warehouse rather than a pile of CSVs:

**Idempotent loads.** ``load_facts`` is safe to re-run on overlapping data.
Rows are keyed on (cik, concept, unit, period_start, period_end, accession) --
an accession number is a specific filing, so the same fact from the same
filing is the same row no matter how many times the pipeline sees it. Re-runs
and backfills converge instead of duplicating, which is what lets the loader
be retried after a partial failure without cleanup.

**Append-only history.** Facts are never updated in place. A restatement is a
NEW row with a later `filed_date`, not an overwrite of the old value. Losing
the superseded value would destroy exactly the information the warehouse is
built to preserve.

**Membership as intervals.** Index membership is stored as sampled
observations rather than open-ended ranges, because that is what the source
actually supports, and inferring precise add/drop dates from quarterly
samples would be fabricating precision.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_DB = Path("data/warehouse.duckdb")

SCHEMA = """
CREATE TABLE IF NOT EXISTS fundamental_facts (
    cik            BIGINT     NOT NULL,
    ticker         VARCHAR    NOT NULL,
    concept        VARCHAR    NOT NULL,
    unit           VARCHAR    NOT NULL,
    period_start   DATE,
    period_end     DATE       NOT NULL,
    filed_date     DATE       NOT NULL,
    value          DOUBLE     NOT NULL,
    form           VARCHAR    NOT NULL,
    accession      VARCHAR    NOT NULL,
    fiscal_year    INTEGER,
    fiscal_period  VARCHAR,
    PRIMARY KEY (cik, concept, unit, period_start, period_end, accession)
);

CREATE TABLE IF NOT EXISTS index_membership (
    index_name   VARCHAR NOT NULL,
    sample_date  DATE    NOT NULL,
    ticker       VARCHAR NOT NULL,
    revision_id  BIGINT,
    PRIMARY KEY (index_name, sample_date, ticker)
);

CREATE TABLE IF NOT EXISTS prices (
    ticker  VARCHAR NOT NULL,
    date    DATE    NOT NULL,
    open    DOUBLE,
    high    DOUBLE,
    low     DOUBLE,
    close   DOUBLE,
    volume  DOUBLE,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS load_log (
    table_name   VARCHAR,
    loaded_at    TIMESTAMP,
    rows_offered BIGINT,
    rows_inserted BIGINT,
    note         VARCHAR
);
"""


def connect(db_path: Path | str = DEFAULT_DB) -> duckdb.DuckDBPyConnection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA)
    return con


def _insert_ignoring_duplicates(
    con: duckdb.DuckDBPyConnection, table: str, frame: pd.DataFrame, columns: list[str], note: str = ""
) -> int:
    """Insert with ON CONFLICT DO NOTHING, and report how many rows were new.

    Returning the *inserted* count rather than the offered count is what makes
    a re-run visibly a no-op in the load log -- useful when debugging whether
    a scheduled pipeline actually picked anything up.
    """
    if frame.empty:
        return 0

    staged = frame[columns].copy()
    # Normalise any datetime column to midnight before staging. The tables
    # store DATE, so a value carrying a time component (pd.Timestamp.today()
    # does) becomes a different Python object but the SAME database key -- two
    # such rows in one batch then collide. ON CONFLICT DO NOTHING does not save
    # you here: DuckDB raises on a conflict WITHIN the inserted batch rather
    # than skipping it, so the dedupe has to happen before the insert.
    for column in staged.columns:
        if pd.api.types.is_datetime64_any_dtype(staged[column]):
            staged[column] = staged[column].dt.normalize()
    staged = staged.drop_duplicates()

    before = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    con.register("staged", staged)
    con.execute(
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"SELECT {', '.join(columns)} FROM staged ON CONFLICT DO NOTHING"
    )
    con.unregister("staged")
    after = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    inserted = after - before
    con.execute(
        "INSERT INTO load_log VALUES (?, now(), ?, ?, ?)",
        [table, len(frame), inserted, note],
    )
    return inserted


def load_facts(con: duckdb.DuckDBPyConnection, facts: pd.DataFrame, note: str = "") -> int:
    columns = [
        "cik", "ticker", "concept", "unit", "period_start", "period_end",
        "filed_date", "value", "form", "accession", "fiscal_year", "fiscal_period",
    ]
    frame = facts.copy()
    # A NULL period_start (instantaneous facts like Assets) is part of the key.
    # DuckDB treats NULLs as distinct in a PK, which would let duplicates in,
    # so normalise to a sentinel date the domain can never produce.
    frame["period_start"] = frame["period_start"].fillna(pd.Timestamp("1900-01-01"))
    return _insert_ignoring_duplicates(con, "fundamental_facts", frame, columns, note)


def load_membership(
    con: duckdb.DuckDBPyConnection, membership: pd.DataFrame, index_name: str = "SP500", note: str = ""
) -> int:
    frame = membership.copy()
    frame["index_name"] = index_name
    if "revision_id" not in frame.columns:
        frame["revision_id"] = None
    return _insert_ignoring_duplicates(
        con, "index_membership", frame, ["index_name", "sample_date", "ticker", "revision_id"], note
    )


def load_prices(con: duckdb.DuckDBPyConnection, prices: pd.DataFrame, note: str = "") -> int:
    columns = ["ticker", "date", "open", "high", "low", "close", "volume"]
    return _insert_ignoring_duplicates(con, "prices", prices, columns, note)


def facts_as_of(con: duckdb.DuckDBPyConnection, date: str, tickers: list[str] | None = None) -> pd.DataFrame:
    """Point-in-time query pushed down to SQL.

    The window function is the whole trick: partition by the identity of the
    fact, order by filing date, and keep the first row -- after filtering to
    filings that had already happened. Doing this in SQL rather than pandas
    means it works on a warehouse larger than memory.
    """
    ticker_filter = ""
    params: list = [date]
    if tickers:
        placeholders = ", ".join("?" for _ in tickers)
        ticker_filter = f"AND ticker IN ({placeholders})"
        params.extend(tickers)

    return con.execute(
        f"""
        WITH visible AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY cik, concept, unit, period_start, period_end
                       ORDER BY filed_date DESC, accession DESC
                   ) AS recency
            FROM fundamental_facts
            WHERE filed_date <= ? {ticker_filter}
        )
        SELECT * EXCLUDE (recency) FROM visible WHERE recency = 1
        ORDER BY ticker, concept, period_end
        """,
        params,
    ).df()


def members_as_of(con: duckdb.DuckDBPyConnection, date: str, index_name: str = "SP500") -> list[str]:
    """Index constituents on `date`, from the most recent sample at or before
    it. Never carries membership backward from a later sample -- that would be
    look-ahead."""
    rows = con.execute(
        """
        SELECT ticker FROM index_membership
        WHERE index_name = ?
          AND sample_date = (
              SELECT MAX(sample_date) FROM index_membership
              WHERE index_name = ? AND sample_date <= ?
          )
        ORDER BY ticker
        """,
        [index_name, index_name, date],
    ).fetchall()
    return [r[0] for r in rows]


def restatement_report(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Every period whose reported value changed after first publication."""
    return con.execute(
        """
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY cik, concept, unit, period_start, period_end
                                      ORDER BY filed_date ASC, accession ASC) AS first_rn,
                   ROW_NUMBER() OVER (PARTITION BY cik, concept, unit, period_start, period_end
                                      ORDER BY filed_date DESC, accession DESC) AS last_rn
            FROM fundamental_facts
        ),
        firsts AS (SELECT * FROM ranked WHERE first_rn = 1),
        lasts  AS (SELECT * FROM ranked WHERE last_rn  = 1)
        SELECT f.ticker, f.concept, f.period_start, f.period_end,
               f.value AS original_value, l.value AS final_value,
               f.filed_date AS first_filed, l.filed_date AS last_filed,
               DATEDIFF('day', f.filed_date, l.filed_date) AS days_to_restate,
               100.0 * (l.value - f.value) / NULLIF(ABS(f.value), 0) AS pct_change
        FROM firsts f
        JOIN lasts l USING (cik, concept, unit, period_start, period_end)
        WHERE f.value <> l.value
        ORDER BY ABS(100.0 * (l.value - f.value) / NULLIF(ABS(f.value), 0)) DESC
        """
    ).df()


def table_counts(con: duckdb.DuckDBPyConnection) -> dict:
    return {
        table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("fundamental_facts", "index_membership", "prices", "load_log")
    }
