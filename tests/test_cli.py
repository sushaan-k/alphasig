"""Tests for sigint.cli -- Command-line interface using Click's CliRunner."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from sigint.cli import _parse_cli_datetime, _print_signal_table, main
from sigint.models import Signal, SignalDirection, SignalType
from sigint.signals import SignalCollection


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mock_signals() -> list[Signal]:
    return [
        Signal(
            timestamp=datetime(2024, 11, 1, tzinfo=UTC),
            ticker="AAPL",
            signal_type=SignalType.SUPPLY_CHAIN,
            direction=SignalDirection.NEUTRAL,
            strength=0.90,
            confidence=0.85,
            context="AAPL depends_on TSMC (semiconductor manufacturing)",
            source_filing="https://sec.gov/test",
            related_tickers=["TSMC"],
            metadata={"target": "TSMC"},
        ),
    ]


class TestMainGroup:
    """Tests for the top-level CLI group."""

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "sigint" in result.output

    def test_verbose_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["-v", "--help"])
        assert result.exit_code == 0

    def test_json_logs_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--json-logs", "--help"])
        assert result.exit_code == 0


class TestExtractCommand:
    """Tests for the 'extract' CLI command."""

    def test_extract_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["extract", "--help"])
        assert result.exit_code == 0
        assert "--tickers" in result.output
        assert "--filing-types" in result.output
        assert "--lookback" in result.output
        assert "--engines" in result.output
        assert "--model" in result.output

    def test_extract_requires_tickers(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["extract"])
        assert result.exit_code != 0
        assert "tickers" in result.output.lower() or "required" in result.output.lower()

    @patch("sigint.pipeline.Pipeline")
    def test_extract_runs_pipeline(
        self,
        mock_pipeline_cls: MagicMock,
        runner: CliRunner,
        mock_signals: list[Signal],
    ) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_collection = SignalCollection(mock_signals)
        mock_pipeline.extract = AsyncMock(return_value=mock_collection)

        result = runner.invoke(
            main,
            ["extract", "--tickers", "AAPL", "--lookback", "1"],
        )
        assert result.exit_code == 0
        mock_pipeline_cls.assert_called_once()

    @patch("sigint.pipeline.Pipeline")
    def test_extract_with_output_parquet(
        self,
        mock_pipeline_cls: MagicMock,
        runner: CliRunner,
        mock_signals: list[Signal],
    ) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_collection = MagicMock(spec=SignalCollection)
        mock_collection.__iter__ = MagicMock(return_value=iter(mock_signals))
        mock_collection.__len__ = MagicMock(return_value=len(mock_signals))
        mock_collection.to_parquet = MagicMock(return_value="output.parquet")
        mock_pipeline.extract = AsyncMock(return_value=mock_collection)

        result = runner.invoke(
            main,
            [
                "extract",
                "--tickers",
                "AAPL",
                "--output",
                "output.parquet",
            ],
        )
        assert result.exit_code == 0
        mock_collection.to_parquet.assert_called_once_with("output.parquet")

    @patch("sigint.pipeline.Pipeline")
    def test_extract_with_output_csv(
        self,
        mock_pipeline_cls: MagicMock,
        runner: CliRunner,
        mock_signals: list[Signal],
    ) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_collection = MagicMock(spec=SignalCollection)
        mock_collection.__iter__ = MagicMock(return_value=iter(mock_signals))
        mock_collection.__len__ = MagicMock(return_value=len(mock_signals))
        mock_collection.to_csv = MagicMock(return_value="output.csv")
        mock_pipeline.extract = AsyncMock(return_value=mock_collection)

        result = runner.invoke(
            main,
            [
                "extract",
                "--tickers",
                "AAPL",
                "--output",
                "output.csv",
            ],
        )
        assert result.exit_code == 0
        mock_collection.to_csv.assert_called_once_with("output.csv")

    @patch("sigint.pipeline.Pipeline")
    def test_extract_with_unknown_extension_defaults_to_parquet(
        self,
        mock_pipeline_cls: MagicMock,
        runner: CliRunner,
        mock_signals: list[Signal],
    ) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_collection = MagicMock(spec=SignalCollection)
        mock_collection.__iter__ = MagicMock(return_value=iter(mock_signals))
        mock_collection.__len__ = MagicMock(return_value=len(mock_signals))
        mock_collection.to_parquet = MagicMock(return_value="output.dat")
        mock_pipeline.extract = AsyncMock(return_value=mock_collection)

        result = runner.invoke(
            main,
            [
                "extract",
                "--tickers",
                "AAPL",
                "--output",
                "output.dat",
            ],
        )
        assert result.exit_code == 0
        mock_collection.to_parquet.assert_called_once_with("output.dat")

    @patch("sigint.pipeline.Pipeline")
    def test_extract_passes_multiple_tickers(
        self,
        mock_pipeline_cls: MagicMock,
        runner: CliRunner,
        mock_signals: list[Signal],
    ) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_collection = SignalCollection(mock_signals)
        mock_pipeline.extract = AsyncMock(return_value=mock_collection)

        result = runner.invoke(
            main,
            [
                "extract",
                "--tickers",
                "AAPL",
                "--tickers",
                "MSFT",
            ],
        )
        assert result.exit_code == 0
        call_args = mock_pipeline.extract.call_args
        # extract() is called with keyword args
        tickers = call_args.kwargs.get("tickers", [])
        assert "AAPL" in tickers
        assert "MSFT" in tickers


class TestQueryCommand:
    """Tests for the 'query' CLI command."""

    def test_query_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["query", "--help"])
        assert result.exit_code == 0
        assert "--db" in result.output
        assert "--ticker" in result.output

    @patch("sigint.storage.SignalStore")
    def test_query_runs(
        self,
        mock_store_cls: MagicMock,
        runner: CliRunner,
        mock_signals: list[Signal],
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.query.return_value = mock_signals

        result = runner.invoke(main, ["query", "--db", ":memory:"])
        assert result.exit_code == 0
        mock_store.close.assert_called_once()

    @patch("sigint.storage.SignalStore")
    def test_query_with_filters(
        self,
        mock_store_cls: MagicMock,
        runner: CliRunner,
        mock_signals: list[Signal],
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.query.return_value = mock_signals

        result = runner.invoke(
            main,
            [
                "query",
                "--db",
                ":memory:",
                "--ticker",
                "AAPL",
                "--type",
                "supply_chain",
                "--min-strength",
                "0.5",
                "--limit",
                "10",
            ],
        )
        assert result.exit_code == 0
        mock_store.query.assert_called_once_with(
            ticker="AAPL",
            signal_type="supply_chain",
            min_strength=0.5,
            limit=10,
        )


class TestRankCommand:
    """Tests for the 'rank' CLI command."""

    def test_rank_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["rank", "--help"])
        assert result.exit_code == 0
        assert "--min-confidence" in result.output
        assert "--as-of" in result.output

    @patch("sigint.storage.SignalStore")
    def test_rank_outputs_json(
        self,
        mock_store_cls: MagicMock,
        runner: CliRunner,
        mock_signals: list[Signal],
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.query.return_value = mock_signals

        result = runner.invoke(
            main,
            [
                "rank",
                "--db",
                ":memory:",
                "--format",
                "json",
                "--min-confidence",
                "0.8",
            ],
        )

        assert result.exit_code == 0
        assert '"ticker_count"' in result.output
        assert '"AAPL"' in result.output
        mock_store.query.assert_called_once_with(
            min_confidence=0.8,
            limit=100_000,
        )
        mock_store.close.assert_called_once()

    @patch("sigint.storage.SignalStore")
    def test_rank_writes_markdown_report(
        self,
        mock_store_cls: MagicMock,
        runner: CliRunner,
        mock_signals: list[Signal],
        tmp_path,
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.query.return_value = mock_signals
        report_path = tmp_path / "reports" / "ranking.md"

        result = runner.invoke(
            main,
            [
                "rank",
                "--db",
                ":memory:",
                "--format",
                "markdown",
                "--output",
                str(report_path),
            ],
        )

        assert result.exit_code == 0
        assert "Wrote ranking report" in result.output
        assert "| Rank | Ticker | Direction |" in report_path.read_text()

    def test_parse_cli_datetime_accepts_z_suffix(self) -> None:
        parsed = _parse_cli_datetime("2024-01-01T12:30:00Z")
        assert parsed == datetime(2024, 1, 1, 12, 30, tzinfo=UTC)

    def test_parse_cli_datetime_rejects_invalid_value(self) -> None:
        with pytest.raises(Exception, match="Invalid --as-of"):
            _parse_cli_datetime("not-a-date")


class TestSectorsCommand:
    """Tests for the 'sectors' CLI command."""

    def test_sectors_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["sectors", "--help"])
        assert result.exit_code == 0
        assert "--exclude-unknown" in result.output
        assert "--min-confidence" in result.output

    @patch("sigint.storage.SignalStore")
    def test_sectors_outputs_json(
        self,
        mock_store_cls: MagicMock,
        runner: CliRunner,
        mock_signals: list[Signal],
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.query.return_value = mock_signals

        result = runner.invoke(
            main,
            [
                "sectors",
                "--db",
                ":memory:",
                "--format",
                "json",
                "--min-confidence",
                "0.8",
                "--exclude-unknown",
            ],
        )

        assert result.exit_code == 0
        assert '"sector_count"' in result.output
        assert '"technology"' in result.output
        mock_store.query.assert_called_once_with(
            min_confidence=0.8,
            limit=100_000,
        )
        mock_store.close.assert_called_once()

    @patch("sigint.storage.SignalStore")
    def test_sectors_writes_markdown_report(
        self,
        mock_store_cls: MagicMock,
        runner: CliRunner,
        mock_signals: list[Signal],
        tmp_path,
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.query.return_value = mock_signals
        report_path = tmp_path / "reports" / "sectors.md"

        result = runner.invoke(
            main,
            [
                "sectors",
                "--db",
                ":memory:",
                "--format",
                "markdown",
                "--output",
                str(report_path),
            ],
        )

        assert result.exit_code == 0
        assert "Wrote sector exposure report" in result.output
        assert "| Rank | Sector | Direction |" in report_path.read_text()


class TestServeCommand:
    """Tests for the 'serve' CLI command."""

    def test_serve_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--host" in result.output
        assert "--db" in result.output

    @patch("sigint.storage.SignalStore")
    @patch(
        "sigint.output.api.serve_signals",
        side_effect=ImportError("Install sigint[api]"),
    )
    def test_serve_reports_missing_api_dependency(
        self,
        _mock_serve_signals: MagicMock,
        mock_store_cls: MagicMock,
        runner: CliRunner,
    ) -> None:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.query.return_value = []

        result = runner.invoke(main, ["serve", "--db", ":memory:"])
        assert result.exit_code != 0
        assert "sigint[api]" in result.output or "Install sigint[api]" in result.output


class TestPrintSignalTable:
    """Tests for the _print_signal_table helper."""

    def test_prints_empty_collection(self) -> None:
        coll = SignalCollection()
        # Should not raise
        _print_signal_table(coll)

    def test_prints_signals(self, mock_signals: list[Signal]) -> None:
        coll = SignalCollection(mock_signals)
        # Should not raise
        _print_signal_table(coll)

    def test_prints_all_directions(self) -> None:
        signals = []
        for direction in SignalDirection:
            signals.append(
                Signal(
                    timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                    ticker="TEST",
                    signal_type=SignalType.SUPPLY_CHAIN,
                    direction=direction,
                    strength=0.5,
                    confidence=0.5,
                    context="Direction test for " + direction.value,
                    source_filing="",
                )
            )
        coll = SignalCollection(signals)
        # Should not raise on any direction
        _print_signal_table(coll)
