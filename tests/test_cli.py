"""Tests for alphasig.cli -- Command-line interface using Click's CliRunner."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from alphasig.cli import _parse_cli_datetime, _print_signal_table, main
from alphasig.models import Signal, SignalDirection, SignalType
from alphasig.signals import SignalCollection


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
        assert "alphasig" in result.output

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

    @patch("alphasig.pipeline.Pipeline")
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

    @patch("alphasig.pipeline.Pipeline")
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

    @patch("alphasig.pipeline.Pipeline")
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

    @patch("alphasig.pipeline.Pipeline")
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

    @patch("alphasig.pipeline.Pipeline")
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

    @patch("alphasig.storage.SignalStore")
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

    @patch("alphasig.storage.SignalStore")
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

    @patch("alphasig.storage.SignalStore")
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

    @patch("alphasig.storage.SignalStore")
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

    @patch("alphasig.storage.SignalStore")
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

    @patch("alphasig.storage.SignalStore")
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

    @patch("alphasig.storage.SignalStore")
    @patch(
        "alphasig.output.api.serve_signals",
        side_effect=ImportError("Install alphasig[api]"),
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
        assert (
            "alphasig[api]" in result.output or "Install alphasig[api]" in result.output
        )


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


class TestCliRegressions:
    @pytest.mark.parametrize(
        "args",
        [
            ["--tickers", "AAPL", "MSFT"],  # the README form
            ["--tickers", "aapl,msft"],
            ["AAPL", "-t", "MSFT"],
        ],
    )
    @patch("alphasig.pipeline.Pipeline")
    def test_ticker_forms(
        self,
        mock_pipeline_cls: MagicMock,
        runner: CliRunner,
        mock_signals: list[Signal],
        args: list[str],
    ) -> None:
        mock_pipeline_cls.return_value.extract = AsyncMock(
            return_value=SignalCollection(mock_signals)
        )
        result = runner.invoke(main, ["extract", *args])
        assert result.exit_code == 0, result.output
        tickers = mock_pipeline_cls.return_value.extract.call_args.kwargs["tickers"]
        assert sorted(tickers) == ["AAPL", "MSFT"]

    @patch("alphasig.pipeline.Pipeline")
    def test_incremental_and_llm_options_reach_the_pipeline(
        self,
        mock_pipeline_cls: MagicMock,
        runner: CliRunner,
        mock_signals: list[Signal],
        tmp_path: Path,
    ) -> None:
        mock_pipeline_cls.return_value.extract = AsyncMock(
            return_value=SignalCollection(mock_signals)
        )
        cache = str(tmp_path / "llm")
        result = runner.invoke(
            main,
            [
                "extract",
                "AAPL",
                "--incremental",
                "--llm-cache-dir",
                cache,
                "--llm-concurrency",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output
        init = mock_pipeline_cls.call_args.kwargs
        assert (init["llm_cache_dir"], init["llm_concurrency"]) == (cache, 3)
        call = mock_pipeline_cls.return_value.extract.call_args.kwargs
        assert call["incremental"] is True

    @patch("alphasig.pipeline.Pipeline")
    def test_llm_defaults(
        self,
        mock_pipeline_cls: MagicMock,
        runner: CliRunner,
        mock_signals: list[Signal],
    ) -> None:
        mock_pipeline_cls.return_value.extract = AsyncMock(
            return_value=SignalCollection(mock_signals)
        )
        result = runner.invoke(main, ["extract", "AAPL"])
        assert result.exit_code == 0, result.output
        init = mock_pipeline_cls.call_args.kwargs
        assert (init["llm_cache_dir"], init["llm_concurrency"]) == (None, 8)
        call = mock_pipeline_cls.return_value.extract.call_args.kwargs
        assert call["incremental"] is False

    def test_llm_concurrency_must_be_positive(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["extract", "AAPL", "--llm-concurrency", "0"])
        assert result.exit_code == 2

    def test_unknown_engine_is_a_usage_error(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["extract", "AAPL", "-e", "sentiment"])
        assert result.exit_code == 2
        assert "sentiment" in result.output

    def test_missing_user_agent_is_a_clean_error(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.delenv("ALPHASIG_USER_AGENT", raising=False)
        result = runner.invoke(main, ["extract", "AAPL", "--db", ":memory:"])
        assert result.exit_code == 1
        assert "User-Agent" in result.output
        assert "Traceback" not in result.output

    @patch("alphasig.storage.SignalStore")
    def test_rank_half_life(
        self, mock_store_cls: MagicMock, runner: CliRunner, mock_signals: list[Signal]
    ) -> None:
        mock_store_cls.return_value.query.return_value = mock_signals
        result = runner.invoke(
            main,
            [
                "rank",
                "--as-of",
                "2024-12-01T00:00:00Z",
                "--half-life",
                "30",
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0, result.output
        assert '"as_of": "2024-12-01T00:00:00+00:00"' in result.output
