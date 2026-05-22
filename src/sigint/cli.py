"""Command-line interface for sigint.

Usage::

    sigint extract --tickers AAPL --tickers MSFT --lookback 3
    sigint serve --port 8080 --db sigint.duckdb
    sigint query --ticker AAPL --type risk_change
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from sigint._logging import configure_logging

console = Console()


@click.group()
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (-v for INFO, -vv for DEBUG).",
)
@click.option(
    "--json-logs",
    is_flag=True,
    default=False,
    help="Emit structured JSON logs.",
)
def main(verbose: int, json_logs: bool) -> None:
    """sigint -- Causal signal extraction from SEC filings."""
    configure_logging(verbosity=verbose, json=json_logs)


@main.command()
@click.option(
    "--tickers",
    "-t",
    required=True,
    multiple=True,
    help="Ticker symbols to analyse.",
)
@click.option(
    "--filing-types",
    "-f",
    multiple=True,
    default=["10-K", "10-Q"],
    help="SEC filing types to process.",
)
@click.option(
    "--lookback",
    "-l",
    default=3,
    type=int,
    help="Years of filings to fetch.",
)
@click.option(
    "--engines",
    "-e",
    multiple=True,
    default=["supply_chain", "risk_differ", "m_and_a", "tone"],
    help="Extraction engines to run.",
)
@click.option(
    "--model",
    "-m",
    default="claude-sonnet-4-6",
    help="LLM model to use.",
)
@click.option(
    "--user-agent",
    default="sigint research bot research@example.com",
    help="EDGAR User-Agent (name + email).",
)
@click.option(
    "--cache-dir",
    default="./edgar_cache",
    help="EDGAR filing cache directory.",
)
@click.option(
    "--db",
    default="sigint.duckdb",
    help="DuckDB database path.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Export signals to file (Parquet or CSV based on extension).",
)
def extract(
    tickers: tuple[str, ...],
    filing_types: tuple[str, ...],
    lookback: int,
    engines: tuple[str, ...],
    model: str,
    user_agent: str,
    cache_dir: str,
    db: str,
    output: str | None,
) -> None:
    """Extract causal signals from SEC filings."""
    from sigint.pipeline import Pipeline

    console.print(
        f"[bold blue]sigint[/] extracting signals for {', '.join(tickers)}",
    )

    pipeline = Pipeline(
        model=model,
        user_agent=user_agent,
        cache_dir=cache_dir,
        db_path=db,
    )

    collection = asyncio.run(
        pipeline.extract(
            tickers=list(tickers),
            filing_types=list(filing_types),
            lookback_years=lookback,
            engines=list(engines),
        )
    )

    # Display results
    _print_signal_table(collection)

    if output:
        if output.endswith(".parquet"):
            path = collection.to_parquet(output)
        elif output.endswith(".csv"):
            path = collection.to_csv(output)
        else:
            path = collection.to_parquet(output)
        console.print(f"\n[green]Exported to {path}[/]")


@main.command()
@click.option("--db", default="sigint.duckdb", help="DuckDB database path.")
@click.option("--ticker", default=None, help="Filter by ticker.")
@click.option("--type", "signal_type", default=None, help="Filter by signal type.")
@click.option(
    "--min-strength", default=None, type=float, help="Minimum signal strength."
)
@click.option("--limit", default=50, type=int, help="Maximum results.")
def query(
    db: str,
    ticker: str | None,
    signal_type: str | None,
    min_strength: float | None,
    limit: int,
) -> None:
    """Query stored signals from DuckDB."""
    from sigint.signals import SignalCollection
    from sigint.storage import SignalStore

    store = SignalStore(db)
    signals = store.query(
        ticker=ticker,
        signal_type=signal_type,
        min_strength=min_strength,
        limit=limit,
    )
    store.close()

    collection = SignalCollection(signals)
    _print_signal_table(collection)


@main.command()
@click.option("--db", default="sigint.duckdb", help="DuckDB database path.")
@click.option("--limit", default=25, type=int, help="Maximum tickers to display.")
@click.option(
    "--min-confidence",
    default=None,
    type=float,
    help="Only score signals at or above this confidence.",
)
@click.option(
    "--as-of",
    "as_of_raw",
    default=None,
    help="Evaluate decayed signal strength at this ISO timestamp.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "markdown"]),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.option("--output", "-o", default=None, help="Write report to a file.")
def rank(
    db: str,
    limit: int,
    min_confidence: float | None,
    as_of_raw: str | None,
    output_format: str,
    output: str | None,
) -> None:
    """Rank tickers by confidence-weighted signal exposure."""
    from sigint.reporting import rank_signals
    from sigint.storage import SignalStore

    as_of = _parse_cli_datetime(as_of_raw) if as_of_raw else None
    store = SignalStore(db)
    try:
        signals = store.query(
            min_confidence=min_confidence,
            limit=100_000,
        )
    finally:
        store.close()

    report = rank_signals(signals, as_of=as_of, limit=limit)
    if output_format == "json":
        rendered = report.to_json() + "\n"
    elif output_format == "markdown":
        rendered = report.to_markdown()
    else:
        rendered = ""
        _print_ranking_table(report.tickers)

    if output:
        if output_format == "table":
            rendered = report.to_markdown()
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        console.print(f"[green]Wrote ranking report to {output}[/]")
    elif rendered:
        console.print(rendered, end="")


@main.command()
@click.option("--db", default="sigint.duckdb", help="DuckDB database path.")
@click.option("--host", default="127.0.0.1", help="Bind address.")
@click.option("--port", default=8080, type=int, help="Bind port.")
def serve(db: str, host: str, port: int) -> None:
    """Launch REST API server for stored signals."""
    from sigint.output.api import serve_signals
    from sigint.storage import SignalStore

    store = SignalStore(db)
    signals = store.query(limit=100_000)
    store.close()

    console.print(
        f"[bold blue]sigint[/] serving {len(signals)} signals on {host}:{port}",
    )
    try:
        serve_signals(signals, host=host, port=port)
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc


def _parse_cli_datetime(value: str) -> datetime:
    """Parse an ISO datetime from CLI input."""
    raw = value.strip()
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise click.ClickException(f"Invalid --as-of timestamp: {value}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _print_signal_table(collection: Iterable[Any]) -> None:
    """Render a Rich table of signals."""
    table = Table(title="Extracted Signals", show_lines=True)
    table.add_column("Ticker", style="cyan", width=8)
    table.add_column("Type", style="magenta", width=14)
    table.add_column("Direction", width=10)
    table.add_column("Strength", justify="right", width=10)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Context", max_width=50)

    direction_style = {
        "bullish": "[green]BULLISH[/]",
        "bearish": "[red]BEARISH[/]",
        "neutral": "[dim]NEUTRAL[/]",
    }

    count = 0
    for signal in collection:
        table.add_row(
            signal.ticker,
            signal.signal_type.value,
            direction_style.get(signal.direction.value, signal.direction.value),
            f"{signal.strength:.2f}",
            f"{signal.confidence:.2f}",
            signal.context[:80],
        )
        count += 1

    console.print(table)
    console.print(f"\n[bold]Total signals:[/] {count}")


def _print_ranking_table(tickers: Iterable[Any]) -> None:
    """Render a Rich table of ranked ticker scores."""
    table = Table(title="Signal Ranking", show_lines=True)
    table.add_column("Rank", justify="right", width=5)
    table.add_column("Ticker", style="cyan", width=8)
    table.add_column("Direction", width=10)
    table.add_column("Score", justify="right", width=10)
    table.add_column("Gross", justify="right", width=10)
    table.add_column("Signals", justify="right", width=8)
    table.add_column("Avg Conf", justify="right", width=9)
    table.add_column("Latest", width=12)
    table.add_column("Top Context", max_width=48)

    direction_style = {
        "bullish": "[green]bullish[/]",
        "bearish": "[red]bearish[/]",
        "neutral": "[dim]neutral[/]",
    }

    count = 0
    for count, ticker in enumerate(tickers, start=1):
        table.add_row(
            str(count),
            ticker.ticker,
            direction_style.get(ticker.direction, ticker.direction),
            f"{ticker.score:.4f}",
            f"{ticker.gross_score:.4f}",
            str(ticker.signal_count),
            f"{ticker.avg_confidence:.2f}",
            ticker.latest_timestamp.date().isoformat(),
            ticker.top_contexts[0][:80] if ticker.top_contexts else "",
        )

    console.print(table)
    console.print(f"\n[bold]Ranked tickers:[/] {count}")
