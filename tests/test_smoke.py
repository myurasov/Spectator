# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests: import every public module and exercise `--help`.

Cheap, network-free coverage that catches import errors, missing
attributes, and broken Typer wiring before they hit a user.
"""

from __future__ import annotations

import importlib

import pytest
from typer.testing import CliRunner


_PUBLIC_MODULES = (
    "src",
    "src.cli",
    "src.config",
    "src.api",
    "src.audio",
    "src.deploy",
    "src.install",
    "src.preflight",
    "src.stack",
)


@pytest.mark.parametrize("modname", _PUBLIC_MODULES)
def test_public_module_imports(modname: str) -> None:
    importlib.import_module(modname)


def test_cli_help_runs() -> None:
    from src.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "spectator" in result.stdout.lower()


def test_audio_presets_subcommand_runs() -> None:
    from src.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["audio", "presets"])
    assert result.exit_code == 0
    assert "studio" in result.stdout
    assert "meeting" in result.stdout
    assert "phone" in result.stdout
    assert "extreme" in result.stdout


def test_stack_config_round_trips_overrides() -> None:
    from src.config import StackConfig

    cfg = StackConfig.from_env(workdir="/tmp/spectator", deploy_profile="alerts")
    assert cfg.workdir == "/tmp/spectator"
    assert cfg.deploy_profile == "alerts"


def test_version_attribute_present() -> None:
    import src

    assert isinstance(src.__version__, str)
    assert src.__version__.count(".") >= 1
