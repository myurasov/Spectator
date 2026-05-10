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


def test_default_remote_workdir_is_dotted() -> None:
    """v0.3.0 moved the default $workdir to a dot-prefixed hidden dir
    (originally `~/.spectator-workdir`, shortened to `~/.spectator` in
    v0.3.2) so the project's on-disk state stays out of the user's
    `ls ~` view by default. Pin the value so any future refactor that
    flips it back surfaces immediately."""
    from src import config

    assert config.DEFAULT_REMOTE_WORKDIR == "~/.spectator"


def test_tool_tree_relpath_is_capitalized() -> None:
    """The rsynced project tree on the target is `$workdir/Spectator/`
    (capitalized — represents the project name as humans see it browsing
    `ls $workdir/`). Pin to catch silent renames."""
    from src import config

    assert config.TOOL_TREE_RELPATH == "Spectator"


def test_remote_tool_dir_uses_capitalized_relpath() -> None:
    """Belt-and-suspenders: deploy._remote_tool_dir() must read from
    config.TOOL_TREE_RELPATH (single source of truth) and not hardcode."""
    from src import config, deploy

    cfg = config.StackConfig(workdir="/tmp/wd")
    assert deploy._remote_tool_dir(cfg) == "/tmp/wd/Spectator"


def test_whisper_command_emits_fp16_only_on_cuda() -> None:
    """`--fp16 True` is only safe on CUDA; MPS has known fp16 quality
    regressions in openai-whisper, and CPU doesn't support fp16 at all."""
    from src.audio import QUALITY_PRESETS, _whisper_command
    from src.config import StackConfig

    cfg = StackConfig.from_env(workdir="/tmp/spectator-test")
    preset = QUALITY_PRESETS["meeting"]

    common = dict(
        audio_basename="audio.mp3",
        cfg=cfg,
        preset=preset,
        model_override=None,
        language=None,
        task="transcribe",
        clip=None,
        initial_prompt=None,
        output_subdir="audio",
    )

    cmd_cuda = _whisper_command(device="cuda", **common)
    assert "--device cuda" in cmd_cuda
    assert "--fp16 True" in cmd_cuda

    cmd_mps = _whisper_command(device="mps", **common)
    assert "--device mps" in cmd_mps
    assert "--fp16 True" not in cmd_mps
    assert "--fp16 False" in cmd_mps

    cmd_cpu = _whisper_command(device="cpu", **common)
    assert "--device cpu" in cmd_cpu
    assert "--fp16 True" not in cmd_cpu
    assert "--fp16 False" in cmd_cpu


def test_detect_device_falls_back_to_cpu_when_probe_fails(monkeypatch) -> None:
    """If the audio-venv probe returns non-zero or empty stdout,
    `_detect_device` returns 'cpu' so the actual whisper call surfaces
    a clearer error than the probe."""
    from src import audio as audio_mod
    from src._run import RunResult
    from src.config import StackConfig

    cfg = StackConfig.from_env(workdir="/tmp/no-such-spectator-workdir")

    monkeypatch.setattr(audio_mod, "run", lambda *a, **kw: RunResult(rc=1, stdout="", stderr="venv missing"))
    assert audio_mod._detect_device(cfg, host=None) == "cpu"

    monkeypatch.setattr(audio_mod, "run", lambda *a, **kw: RunResult(rc=0, stdout="cpu\n", stderr=""))
    assert audio_mod._detect_device(cfg, host=None) == "cpu"

    monkeypatch.setattr(audio_mod, "run", lambda *a, **kw: RunResult(rc=0, stdout="mps\n", stderr=""))
    assert audio_mod._detect_device(cfg, host=None) == "mps"

    monkeypatch.setattr(audio_mod, "run", lambda *a, **kw: RunResult(rc=0, stdout="cuda\n", stderr=""))
    assert audio_mod._detect_device(cfg, host=None) == "cuda"

    monkeypatch.setattr(audio_mod, "run", lambda *a, **kw: RunResult(rc=0, stdout="something-weird\n", stderr=""))
    assert audio_mod._detect_device(cfg, host=None) == "cpu"


def test_audio_transcribe_help_advertises_device_flag() -> None:
    """The CLI's `audio transcribe --help` should document --device."""
    from src.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["audio", "transcribe", "--help"])
    assert result.exit_code == 0
    assert "--device" in result.stdout
    for kw in ("cuda", "mps", "cpu"):
        assert kw in result.stdout


def test_install_script_handles_existing_non_git_vss_dir() -> None:
    """v0.3.4: the user-install bash script must distinguish three states
    of `$workdir/video-search-and-summarization/`:

      1. dir + `.git/`   -> fetch path
      2. dir without .git, empty -> rmdir + clone fresh
      3. dir without .git, non-empty -> error with rm-rf hint

    Pre-v0.3.4 the script only handled (1) and an else `git clone` that
    crashed loudly on (2) and (3) ('destination path ... already exists
    and is not an empty directory'). We render the script and assert
    each branch is present so future edits can't silently regress."""
    from src.install import _user_install_script

    script = _user_install_script(
        workdir="~/.spectator",
        vss_checkout="video-search-and-summarization",
        ngc_api_key=None,
    )

    assert 'if [ -d "video-search-and-summarization/.git" ]; then' in script
    assert 'elif [ -d "video-search-and-summarization" ]; then' in script
    assert 'rmdir "video-search-and-summarization"' in script
    assert "exists but is not a git checkout" in script
    assert "rm -rf ~/.spectator/video-search-and-summarization" in script
