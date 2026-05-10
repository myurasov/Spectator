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


def test_down_script_covers_every_spectator_launched_thing() -> None:
    """v0.3.5: `spectator down` must cover everything Spectator launches
    on the target, not just the VSS docker stack and the bring-up tmux.

    Each branch of the rendered bash payload is pinned here so a future
    edit can't silently regress the cleanup. Branches:

      1. dev-profile.sh down (VSS docker stack)
      2. tmux kill-session -t spectator-up (bring-up watcher)
      3. tmux kill-session for any `audio-*` session (transcribe jobs —
         these can be holding the GPU long after VSS is down)
      4. cache cleaner status surfaced but NOT auto-stopped (sudo)
      5. final docker ps snapshot for confirmation
    """
    from src import stack
    from src.config import StackConfig

    captured: list[str] = []

    def fake_exec(host, script, env=None):
        captured.append(script)
        from src._run import RunResult

        return RunResult(rc=0, stdout="", stderr="")

    original = stack._exec
    stack._exec = fake_exec
    try:
        stack.down(StackConfig(workdir="~/.spectator"), host="myspark1-local")
    finally:
        stack._exec = original

    script = captured[0]

    assert "scripts/dev-profile.sh down" in script
    assert "tmux kill-session -t spectator-up" in script
    assert "grep '^audio-'" in script
    assert "tmux kill-session -t \"$s\"" in script
    assert "sys-cache-cleaner.sh" in script
    assert "spectator system cache-cleaner-stop" in script
    assert "post-down" in script
    assert "~/.spectator/bin/sys-cache-cleaner.sh" in script  # workdir interp


def test_wrapper_bootstrap_skips_uv_when_venv_is_healthy(tmp_path) -> None:
    """v0.3.8 regression test for the require-uv-even-when-cached bug.

    Pre-v0.3.8, `bootstrap()` in the `./spectator` wrapper called
    `require_uv` unconditionally, even when `.venv/` was already
    populated with all the dev deps and the install stamp was fresh.
    That broke the wrapper on hosts where uv isn't on PATH (typical
    non-interactive SSH session) but the venv had been pre-deployed
    via rsync, e.g. when running `./spectator audio presets` on the
    Spark after `./spectator deploy --target …` from the laptop.

    Fix: bootstrap takes a fast path that returns immediately if
    .venv looks healthy, only invoking `require_uv` when an actual
    sync would otherwise be needed.

    Test: simulate a healthy-venv directory tree, run the wrapper with
    a sabotaged PATH that has no uv, and confirm a no-op subcommand
    works end to end. We use the wrapper itself (read from the repo
    root) and rely on the fast-path triggering."""
    import os
    import shutil
    import stat
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    wrapper_src = repo_root / "spectator"
    pyproject_src = repo_root / "pyproject.toml"

    # Build a fake project tree under tmp_path mirroring the wrapper's
    # expectations: spectator wrapper, pyproject.toml, src/, .venv/ with
    # the deps_ready() probe targets.
    proj = tmp_path / "proj"
    proj.mkdir()
    shutil.copy(wrapper_src, proj / "spectator")
    (proj / "spectator").chmod(0o755)
    shutil.copy(pyproject_src, proj / "pyproject.toml")
    # Minimal src/ package so `python -m src --help` resolves something
    # — we don't actually invoke it, we just want the wrapper to NOT
    # complain about missing uv. We invoke `./spectator help` which
    # short-circuits before it would try to run python.
    (proj / "src").mkdir()
    (proj / "src" / "__init__.py").write_text("")
    venv_bin = proj / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    # The wrapper's `deps_ready` probes need executable shims at these
    # paths. We don't actually run them — `./spectator help` exits
    # before that. But `deps_ready` does exec the python with `-c
    # "import typer, rich, httpx, yaml"`. We make our shim succeed for
    # any args.
    py_shim = venv_bin / "python"
    py_shim.write_text("#!/bin/sh\nexit 0\n")
    py_shim.chmod(py_shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    for name in ("pytest", "ruff"):
        b = venv_bin / name
        b.write_text("#!/bin/sh\nexit 0\n")
        b.chmod(b.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    # Mark the install stamp older than... nothing (the stamp itself is
    # the freshness signal). Just creating it works because we set its
    # mtime to be newer than pyproject.toml's mtime explicitly.
    stamp = proj / ".venv" / ".spectator-installed"
    stamp.write_text("")
    pyproject_mtime = (proj / "pyproject.toml").stat().st_mtime
    os.utime(stamp, (pyproject_mtime + 60, pyproject_mtime + 60))

    # Sabotaged PATH: no uv anywhere.
    env = {
        **os.environ,
        "PATH": "/usr/bin:/bin",
    }
    # Confirm the sabotage actually denied uv.
    which = shutil.which("uv", path=env["PATH"])
    assert which is None, f"PATH sabotage failed; uv found at {which!r}"

    # Run `./spectator help` — should print help, exit 0, and never
    # complain about uv.
    r = subprocess.run(
        [str(proj / "spectator"), "help"],
        cwd=str(proj),
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert r.returncode == 0, (
        f"wrapper failed under sabotaged PATH: rc={r.returncode}\n"
        f"--stdout--\n{r.stdout}\n--stderr--\n{r.stderr}"
    )
    assert "uv is required" not in r.stderr, (
        f"wrapper still complained about uv:\n{r.stderr}"
    )
    # `help` outputs the docstring banner.
    assert "Spectator" in r.stdout
    assert "wrapper" in r.stdout.lower() or "subcommand" in r.stdout.lower()


def test_audio_transcribe_local_wrapper_emits_parseable_end_line(monkeypatch, tmp_path) -> None:
    """v0.3.7 regression test for the END-marker mismatch.

    The Web UI's WebSocket progress handler at `webui/routes/ws.py:85`
    falls back to "subprocess died without END marker" when its parser
    (`webui/progress.py:_END_RE = re.compile(r'==== END rc=(-?\\d+) ')`)
    can't find a matching END line in the job log. Pre-v0.3.7 the
    non-detach branch of `audio.transcribe()` emitted

        echo "==== END $(date) ===="

    — no `rc=` token, so the regex didn't match and successful local
    jobs got mislabeled as failed.

    This test captures the rendered bash payload and asserts that both
    branches now emit `rc=` in the END line, with the same shape the
    progress parser expects."""
    import re

    from src import audio as audio_mod
    from src._run import RunResult
    from src.config import StackConfig
    from src.webui.progress import _END_RE

    captured: list[tuple[list[str], dict | None]] = []

    def fake_run(args, env=None, **kw):
        captured.append((list(args), env))
        return RunResult(rc=0, stdout="", stderr="")

    monkeypatch.setattr(audio_mod, "run", fake_run)
    monkeypatch.setattr(audio_mod, "_detect_device", lambda *a, **kw: "cpu")

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    audio_local = tmp_path / "input.mp3"
    audio_local.write_bytes(b"fake mp3 bytes for test")

    cfg = StackConfig(workdir="~/wd")
    audio_mod.transcribe(audio_local, host=None, cfg=cfg, detach=False, follow=False)

    # The second `run()` call is the wrapper bash payload (the first is
    # the mkdir setup; we don't care about that one for this test).
    wrapper_calls = [c for c in captured if c[0][:2] == ["bash", "-c"]]
    assert len(wrapper_calls) >= 2, (
        f"expected setup + wrapper bash calls, got: {captured}"
    )
    wrapper_script = wrapper_calls[-1][0][2]

    # The END line must include `rc=` so the WS progress parser detects
    # completion. We don't pin the exact text (the date / rc value vary
    # at run time), just the regex match.
    rendered_end = re.search(r'echo "==== END rc=\$RC.*===="', wrapper_script)
    assert rendered_end is not None, (
        f"non-detach wrapper must emit '==== END rc=$RC <date> ====' "
        f"so the WS parser can match it. Got:\n{wrapper_script}"
    )

    # And: a synthetic log with the rendered shape (rc=0) parses as finished.
    sample_log = (
        "==== START Sat May  9 19:02:09 PM PDT 2026 ====\n"
        "[00:00.000 --> 00:01.000] hello\n"
        "==== END rc=0 Sat May  9 19:12:04 PM PDT 2026 ====\n"
    )
    assert _END_RE.search(sample_log) is not None
    m = _END_RE.search(sample_log)
    assert m is not None
    assert int(m.group(1)) == 0  # rc=0 → success


def test_audio_local_copy_resolves_workdir_correctly(tmp_path, monkeypatch) -> None:
    """v0.3.6 regression test for the literal-`$HOME`-directory bug.

    The internal helper `_in_dir(cfg)` returns a bash-friendly path
    form that prefixes a literal `$HOME/...` so it survives quoting in
    bash heredocs. Pre-v0.3.6, the local-copy block in `audio.transcribe()`
    fed that string through `os.path.expanduser`, which only expands a
    leading `~` — `$HOME` stayed as a literal token, and the resulting
    `Path("$HOME/...")` was treated as a relative path under the
    subprocess cwd. The audio file ended up at
    `<cwd>/$HOME/.spectator/audio-in/<basename>` (a real on-disk dir
    named `$HOME`), and whisper then failed with `Error opening input
    file ~/.spectator/audio-in/<basename>` because nothing actually
    landed there.

    Fix: walk `cfg.workdir` through `Path.expanduser()` directly. This
    test asserts that a local transcribe invocation copies the audio
    to `<workdir>/audio-in/<basename>` (real path, no `$HOME` token).
    """
    import shutil

    from src import audio as audio_mod
    from src.config import StackConfig

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    workdir = "~/wd"  # exercises the tilde-prefix branch of _expand_tilde
    cfg = StackConfig(workdir=workdir)

    # Stub out everything except the local-copy block we care about.
    # _detect_device + the bash setup + the actual whisper run all need
    # to be no-oped because we don't have an audio-venv in the test env.
    audio_local = tmp_path / "input.mp3"
    audio_local.write_bytes(b"fake mp3 bytes for test")

    from src._run import RunResult

    monkeypatch.setattr(audio_mod, "run", lambda *a, **kw: RunResult(rc=0, stdout="", stderr=""))
    monkeypatch.setattr(audio_mod, "ssh_run", lambda *a, **kw: RunResult(rc=0, stdout="", stderr=""))
    monkeypatch.setattr(audio_mod, "_detect_device", lambda *a, **kw: "cpu")

    audio_mod.transcribe(
        audio_local,
        host=None,
        cfg=cfg,
        detach=False,
        follow=False,
    )

    # Assert: the file landed at the real expanded workdir, NOT at a
    # literal `$HOME` directory anywhere under the cwd.
    expected = fake_home / "wd" / "audio-in" / "input.mp3"
    assert expected.is_file(), (
        f"file did not land at {expected}; tree under tmp_path:\n"
        + "\n".join(str(p) for p in tmp_path.rglob("*"))
    )
    assert expected.read_bytes() == b"fake mp3 bytes for test"

    # Belt-and-suspenders: the literal "$HOME" directory must NOT exist
    # anywhere we can see.
    bogus = list(tmp_path.rglob("$HOME"))
    assert not bogus, f"a literal $HOME dir leaked: {bogus}"


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
