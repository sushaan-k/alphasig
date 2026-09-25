"""Packaging regression guards: typed marker and console scripts."""

from __future__ import annotations

from importlib.metadata import entry_points
from importlib.resources import files

import alphasig
import alphasig.cli


def test_py_typed_ships() -> None:
    assert files("alphasig").joinpath("py.typed").is_file()


def test_version_is_exposed() -> None:
    assert alphasig.__version__ not in {"", "0+unknown"}


def test_console_scripts_resolve_to_cli() -> None:
    scripts = {ep.name: ep for ep in entry_points(group="console_scripts")}
    for name in ("alphasig", "sigint"):  # sigint: legacy pre-0.2 alias
        assert scripts[name].value == "alphasig.cli:main"
        assert scripts[name].load() is alphasig.cli.main
