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
    # Make sure the v0.4.1 MPS-skip default (which we test below) doesn't
    # interfere with these unrelated paths — explicitly clear the
    # opt-in env var first.
    monkeypatch.delenv("SPECTATOR_ALLOW_MPS_AUTO", raising=False)

    monkeypatch.setattr(audio_mod, "run", lambda *a, **kw: RunResult(rc=1, stdout="", stderr="venv missing"))
    assert audio_mod._detect_device(cfg, host=None) == "cpu"

    monkeypatch.setattr(audio_mod, "run", lambda *a, **kw: RunResult(rc=0, stdout="cpu\n", stderr=""))
    assert audio_mod._detect_device(cfg, host=None) == "cpu"

    monkeypatch.setattr(audio_mod, "run", lambda *a, **kw: RunResult(rc=0, stdout="cuda\n", stderr=""))
    assert audio_mod._detect_device(cfg, host=None) == "cuda"

    monkeypatch.setattr(audio_mod, "run", lambda *a, **kw: RunResult(rc=0, stdout="something-weird\n", stderr=""))
    assert audio_mod._detect_device(cfg, host=None) == "cpu"


def test_creds_source_block_renders_bash_with_workdir() -> None:
    """v0.4.4: source_block produces a bash one-liner that sources
    `$workdir/.creds` if it exists, with set -a/+a so unexported
    `VAR=VALUE` lines still get exported (defensive for hand-edited
    .creds files)."""
    from src import _creds

    block = _creds.source_block("$HOME/.spectator")

    assert "$HOME/.spectator/.creds" in block
    assert "set -a" in block
    assert "set +a" in block
    # ". <path>" is the POSIX `source` (more portable than `source ...`).
    assert '. "$HOME/.spectator/.creds"' in block
    # The whole block should be a single line so it can sit at the top
    # of any heredoc without disrupting indentation.
    assert "\n" not in block

    # Sanity: actually parses as bash (no `bash -n` available cross-
    # platform here, so we just check it doesn't have unbalanced quotes).
    assert block.count('"') % 2 == 0


def test_creds_save_block_writes_file_only_when_absent(tmp_path) -> None:
    """v0.4.4: save_block produces bash that writes $workdir/.creds
    only if the file doesn't exist, captures NGC_CLI_API_KEY /
    NVIDIA_API_KEY / LLM_ENDPOINT_URL via printf %q (shell-safe), and
    chmod 600's the result.

    Run the rendered bash against a tmp fake-workdir and confirm: file
    is created, has 0600 perms, and contains export lines for any vars
    that were set in env."""
    import os
    import subprocess

    from src import _creds

    fake_wd = tmp_path / "wd"
    fake_wd.mkdir()
    block = _creds.save_block(str(fake_wd))

    # Run the block with a minimal env containing test creds.
    env = {
        **os.environ,
        "NGC_CLI_API_KEY": "nvapi-test-ngc",
        "NVIDIA_API_KEY": "nvapi-test-nvidia",
        # LLM_ENDPOINT_URL deliberately unset to exercise the
        # "skip-empty-vars" branch.
    }
    env.pop("LLM_ENDPOINT_URL", None)

    r = subprocess.run(
        ["bash", "-c", block],
        env=env, capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, f"save_block bash failed: {r.stderr}"

    creds_file = fake_wd / ".creds"
    assert creds_file.is_file()
    # 0600 permissions
    mode = creds_file.stat().st_mode & 0o777
    assert mode == 0o600, f"expected mode 0600, got {oct(mode)}"

    contents = creds_file.read_text()
    assert "export NGC_CLI_API_KEY=nvapi-test-ngc" in contents
    assert "export NVIDIA_API_KEY=nvapi-test-nvidia" in contents
    # Unset var should NOT show up.
    assert "LLM_ENDPOINT_URL" not in contents

    # Re-running the block must NOT clobber the file (idempotent).
    creds_file.write_text("# user-edited; should be preserved\n")
    r = subprocess.run(
        ["bash", "-c", block],
        env=env, capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0
    assert creds_file.read_text() == "# user-edited; should be preserved\n"


def test_creds_round_trip_source_overrides_env() -> None:
    """End-to-end priority: source_block overrides whatever was in env
    before sourcing, because `set -a` + `.` re-exports the .creds
    values. Spec is "creds is the source of truth once it exists" —
    pin that the override actually happens.
    """
    import os
    import subprocess
    import tempfile

    from src import _creds

    with tempfile.TemporaryDirectory() as tmp:
        creds_file = os.path.join(tmp, ".creds")
        with open(creds_file, "w") as fh:
            fh.write("export NGC_CLI_API_KEY=from-creds\n")
            fh.write("export NVIDIA_API_KEY=from-creds\n")

        block = _creds.source_block(tmp)
        # Bash that runs source_block and then echoes the resulting
        # env vars. Pre-set the env vars to "from-shell" so we can
        # confirm the source overrides them.
        bash = (
            'export NGC_CLI_API_KEY=from-shell\n'
            'export NVIDIA_API_KEY=from-shell\n'
            f'{block}\n'
            'echo "NGC=$NGC_CLI_API_KEY"\n'
            'echo "NVIDIA=$NVIDIA_API_KEY"\n'
        )

        r = subprocess.run(["bash", "-c", bash], capture_output=True, text=True, timeout=5)
        assert r.returncode == 0, f"bash failed: {r.stderr}"
        assert "NGC=from-creds" in r.stdout, (
            f".creds should override env-set NGC_CLI_API_KEY; got:\n{r.stdout}"
        )
        assert "NVIDIA=from-creds" in r.stdout


def test_install_script_includes_creds_source_and_save_blocks() -> None:
    """v0.4.4 plumbing pin: the rendered install bash payload sources
    .creds at the top and writes it at the end (after the NGC docker
    login step). Both pieces must be present so first-install creds
    capture works AND subsequent installs respect existing .creds."""
    from src.install import _user_install_script

    script = _user_install_script(
        workdir="~/.spectator",
        vss_checkout="video-search-and-summarization",
        ngc_api_key=None,
    )

    # source_block: bash that sources .creds before any other work.
    assert '$HOME/.spectator/.creds' in script
    assert "set -a" in script
    # save_block: writes .creds when it doesn't exist.
    assert "wrote $CREDS_FILE" in script
    assert "chmod 600" in script
    # Both blocks reference the bash-friendly $HOME/.spectator path,
    # NOT the literal "~/.spectator" (which doesn't tilde-expand
    # inside double quotes).
    assert "~/.spectator/.creds" not in script


def test_preflight_finds_creds_in_workdir(monkeypatch) -> None:
    """v0.4.5: preflight's NGC / NVIDIA key checks should report
    ``in $workdir/.creds`` when the file is present and contains the
    keys, with priority over the driving-shell-env fallback. Mock
    `_exec` to return what `set -a; . .creds; echo ${VAR:+SET}` would
    produce on a target that has both keys persisted."""
    from src import preflight as pm
    from src._run import RunResult

    captured_scripts: list[str] = []

    def fake_exec(host, script):
        captured_scripts.append(script)
        # Match the bash patterns preflight uses. The .creds probe is
        # the only one we care about for this test; every other probe
        # we stub to "MISSING\n" so its check transitions to a non-OK
        # state without blowing up on splitlines()[0].
        if ".creds" in script and "set -a" in script:
            return RunResult(rc=0, stdout="NGC=SET\nNVIDIA=SET\n", stderr="")
        return RunResult(rc=0, stdout="MISSING\n", stderr="")

    monkeypatch.setattr(pm, "_exec", fake_exec)
    # Clear env-var fallback so we can see .creds win unambiguously.
    monkeypatch.delenv("NGC_CLI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    checks = pm.collect_checks(host="myspark1-local", workdir="~/.spectator")
    by_name = {c.name: c for c in checks}

    assert "NGC API key" in by_name
    assert by_name["NGC API key"].ok is True
    assert "in ~/.spectator/.creds on target" in by_name["NGC API key"].detail

    assert "NVIDIA API key" in by_name
    assert by_name["NVIDIA API key"].ok is True
    assert "in ~/.spectator/.creds on target" in by_name["NVIDIA API key"].detail

    # Sanity: the .creds probe was actually invoked, with the right path.
    creds_probes = [s for s in captured_scripts if "/.creds" in s]
    assert creds_probes, "preflight should ssh-probe $workdir/.creds"
    assert any("~/.spectator/.creds" in s for s in creds_probes)


def test_preflight_falls_back_to_driving_shell_env_when_creds_absent(monkeypatch) -> None:
    """If .creds doesn't exist on the target, preflight reports the
    driving-shell env var. Pre-v0.4.4 behavior preserved as the
    fallback."""
    from src import preflight as pm
    from src._run import RunResult

    def fake_exec(host, script):
        if ".creds" in script and "set -a" in script:
            # No file → both empty. (Bash `${VAR:+SET}` is empty when
            # the var is empty/unset.)
            return RunResult(rc=0, stdout="NGC=\nNVIDIA=\n", stderr="")
        if "test -f ~/.ngc/api_key" in script:
            return RunResult(rc=0, stdout="absent\n", stderr="")
        return RunResult(rc=0, stdout="MISSING\n", stderr="")  # short-circuit other probes

    monkeypatch.setattr(pm, "_exec", fake_exec)
    monkeypatch.setenv("NGC_CLI_API_KEY", "nvapi-from-shell")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    checks = pm.collect_checks(host="myspark1-local", workdir="~/.spectator")
    by_name = {c.name: c for c in checks}

    # NGC: env wins (creds empty).
    assert by_name["NGC API key"].ok is True
    assert "set in driving shell" in by_name["NGC API key"].detail
    assert "first install will persist to .creds" in by_name["NGC API key"].detail

    # NVIDIA: nothing anywhere → fail.
    assert by_name["NVIDIA API key"].ok is False
    assert "$workdir/.creds" in by_name["NVIDIA API key"].detail


def test_stack_up_sources_creds_block() -> None:
    """spectator up needs NVIDIA_API_KEY for remote LLM auth and
    NGC_CLI_API_KEY for docker pull. After v0.4.4, the up bash payload
    sources .creds at the top so both come from the canonical file
    (or fall through to the SSH-propagated env if .creds is absent)."""
    from src import config, stack

    captured: list[str] = []

    def fake_exec(host, script, env=None):
        captured.append(script)
        from src._run import RunResult
        return RunResult(rc=0, stdout="", stderr="")

    original_exec = stack._exec
    stack._exec = fake_exec
    try:
        cfg = config.StackConfig(workdir="~/.spectator")
        stack.up(cfg, host="myspark1-local")
    finally:
        stack._exec = original_exec

    assert len(captured) == 1
    script = captured[0]
    assert '$HOME/.spectator/.creds' in script
    assert "set -a" in script


def test_audio_fetch_quotes_remote_path_with_spaces(monkeypatch, tmp_path) -> None:
    """v0.4.2 regression test for bug 3 in the 2026-05-09 report.

    `spectator audio fetch --target HOST --only 'stem with spaces'`
    used to fail because the remote rsync path wasn't shell-escaped
    before going through SSH. The remote shell would word-split
    ``host:/path/audio-out/stem with spaces/`` into three pieces, each
    treated as a separate rsync source, and rsync would die with
    ``link_stat /path/audio-out/stem ... No such file or directory``.

    Fix: ``audio.fetch()`` now wraps the remote path with
    ``shlex.quote()``. We render the rsync argv via a fake `run` and
    assert: (a) only one positional path argument before dest, (b) it
    starts with ``host:`` followed by the single-quoted full path
    (literal apostrophes around the entire ``cfg.workdir/audio-out/...``
    suffix), (c) the inner spaces / parens / dollar signs are not
    escaped — single-quoting handles them whole.
    """
    from src import audio as audio_mod
    from src._run import RunResult
    from src.config import StackConfig

    captured: list[list[str]] = []

    def fake_run(args, env=None, **kw):
        captured.append(list(args))
        return RunResult(rc=0, stdout="", stderr="")

    monkeypatch.setattr(audio_mod, "run", fake_run)

    cfg = StackConfig(workdir="~/.spectator")
    dest = tmp_path / "out"

    # Stem with all the real-world shell-active troublemakers — spaces,
    # parens, brackets, dollar, backticks (on a Mac where meeting
    # recordings tend to have these).
    nasty_stem = "MY-NIMA 1_1 0507 (v2) [final] $TEST `cmd`"

    audio_mod.fetch(host="myspark1-local", cfg=cfg, dest=dest, only=nasty_stem)

    assert len(captured) == 1
    args = captured[0]
    # Layout: ["rsync", "-avh", "--progress", "host:'<quoted-path>'", "<dest>/"]
    assert args[0] == "rsync"
    assert "-avh" in args
    assert "--progress" in args

    # Find the host:... arg
    remote_args = [a for a in args if a.startswith("myspark1-local:")]
    assert len(remote_args) == 1, (
        f"expected exactly one host:path arg, got: {args}"
    )
    remote_arg = remote_args[0]

    # The path part must be single-quoted (shlex.quote default), and
    # the nasty stem must appear inside the quotes verbatim.
    _, _, remote_path_quoted = remote_arg.partition(":")
    assert remote_path_quoted.startswith("'")
    assert remote_path_quoted.endswith("/'")  # trailing slash is rsync convention
    assert nasty_stem in remote_path_quoted

    # Sanity: verify the rendered shell sees ONE token. We don't do
    # this on a real remote — just shell-parse the local rsync argv
    # element to confirm it would round-trip.
    import shlex as _shlex
    parsed = _shlex.split(remote_arg)
    assert len(parsed) == 1, (
        f"remote arg word-splits into multiple tokens — quoting failed.\n"
        f"  arg = {remote_arg!r}\n"
        f"  split = {parsed}"
    )
    # Single token == "host:/full/path/with everything intact/"
    expected_path = (
        f"~/.spectator/audio-out/{nasty_stem}/"
    )
    assert parsed[0] == f"myspark1-local:{expected_path}"


def test_audio_fetch_local_no_host_no_quoting_needed(monkeypatch, tmp_path) -> None:
    """Sanity: the local-fetch branch (no SSH) doesn't need shell-quoting
    because the path crosses no shell boundary — Python-side argv stays
    one element. Pin that we're NOT accidentally quoting it (which
    would create a directory literally named "'<path>'")."""
    from src import audio as audio_mod
    from src._run import RunResult
    from src.config import StackConfig

    captured: list[list[str]] = []

    def fake_run(args, env=None, **kw):
        captured.append(list(args))
        return RunResult(rc=0, stdout="", stderr="")

    monkeypatch.setattr(audio_mod, "run", fake_run)

    cfg = StackConfig(workdir=str(tmp_path / "wd"))
    dest = tmp_path / "out"

    audio_mod.fetch(host=None, cfg=cfg, dest=dest, only="stem with spaces")

    assert len(captured) == 1
    args = captured[0]
    src = args[-2]  # rsync ... <src> <dest>/
    # No surrounding quotes (Python-level argv).
    assert not src.startswith("'")
    assert "stem with spaces" in src


def test_detect_device_skips_mps_by_default(monkeypatch, capsys) -> None:
    """v0.4.1: auto-detected MPS is downgraded to CPU because openai-whisper
    crashes on Apple Silicon GPU with the large-v3 model family used by
    every Spectator quality preset. See bug 2 in the 2026-05-09 report
    + upstream openai/whisper#2151.

    Probe returns 'mps' but `_detect_device` returns 'cpu' and prints
    a one-line warning explaining the override. Operator can opt back
    in with SPECTATOR_ALLOW_MPS_AUTO=1.
    """
    from src import audio as audio_mod
    from src._run import RunResult
    from src.config import StackConfig

    cfg = StackConfig.from_env(workdir="/tmp/no-such-spectator-workdir")
    monkeypatch.delenv("SPECTATOR_ALLOW_MPS_AUTO", raising=False)
    monkeypatch.setattr(audio_mod, "run", lambda *a, **kw: RunResult(rc=0, stdout="mps\n", stderr=""))

    # Default: probe says mps, _detect_device returns cpu.
    detected = audio_mod._detect_device(cfg, host=None)
    assert detected == "cpu", (
        "auto-detected mps must downgrade to cpu by default — see "
        "openai/whisper#2151 for the upstream crash this avoids."
    )

    # The downgrade should be visible to the operator (rich Console
    # writes to stdout by default, which capsys catches).
    captured = capsys.readouterr()
    out_plus_err = captured.out + captured.err
    assert "MPS detected but auto-skipped" in out_plus_err, (
        f"warning missing from stdout/stderr; got:\n{out_plus_err}"
    )
    assert "openai/whisper" in out_plus_err  # the issue link/citation
    assert "SPECTATOR_ALLOW_MPS_AUTO" in out_plus_err  # the override hint


def test_detect_device_respects_mps_opt_in_env_var(monkeypatch, capsys) -> None:
    """Setting SPECTATOR_ALLOW_MPS_AUTO=1 re-enables auto-detect MPS.
    For users who've patched whisper or are testing a smaller model
    known not to hit the upstream bug."""
    from src import audio as audio_mod
    from src._run import RunResult
    from src.config import StackConfig

    cfg = StackConfig.from_env(workdir="/tmp/no-such-spectator-workdir")
    monkeypatch.setenv("SPECTATOR_ALLOW_MPS_AUTO", "1")
    monkeypatch.setattr(audio_mod, "run", lambda *a, **kw: RunResult(rc=0, stdout="mps\n", stderr=""))

    assert audio_mod._detect_device(cfg, host=None) == "mps"

    captured = capsys.readouterr()
    out_plus_err = captured.out + captured.err
    assert "auto-skipped" not in out_plus_err  # no warning when opted in


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


def test_uninstall_script_removes_workdir_and_lists_untouched_things() -> None:
    """v0.4.0: the uninstall bash payload must rm -rf the configured
    $workdir AND list (in plain text, for the operator) every category
    of state Spectator deliberately doesn't auto-clean: docker images,
    NGC docker login, --apply-system mutations, the uv binary.

    Pin every branch so a future edit can't silently regress to
    "uninstall removes ~/.docker/config.json out from under unrelated
    NVIDIA tooling" or "uninstall forgot the workdir".
    """
    from src.install import _uninstall_script

    script = _uninstall_script(
        workdir="~/.spectator",
        vss_checkout="video-search-and-summarization",
    )

    assert 'WORKDIR="~/.spectator"' in script
    assert 'rm -rf "$WORKDIR"' in script

    # Safety guards against rm-rf-ing the wrong thing
    assert '[ -z "$WORKDIR" ]' in script
    assert '[ "$WORKDIR" = "/" ]' in script
    assert '[ "$WORKDIR" = "$HOME" ]' in script

    # Each category of left-behind state is mentioned by name so the
    # operator gets a checklist of what they may want to clean
    # manually.
    assert "docker images" in script.lower()
    assert "nvcr.io" in script
    assert "docker logout nvcr.io" in script
    assert "apply-system" in script
    assert "uv" in script  # ~/.local/bin/uv mention


def test_uninstall_function_calls_stack_down_first(monkeypatch) -> None:
    """uninstall() must run stack.down() before the rm -rf so VSS
    containers / spectator-up tmux / audio-* tmux sessions exit
    cleanly. A naive `rm -rf` against a still-running compose project
    would leave dangling docker volumes and confuse the next install.
    """
    from src import install as install_mod
    from src import stack as stack_mod
    from src._run import RunResult
    from src.config import StackConfig

    call_order: list[str] = []

    def fake_down(cfg, host=None):
        call_order.append("stack.down")
        return RunResult(rc=0, stdout="(fake down output)", stderr="")

    def fake_run(args, env=None, **kw):
        call_order.append("rm -rf bash")
        return RunResult(rc=0, stdout="", stderr="")

    def fake_ssh_run(host, script, env=None):
        call_order.append(f"rm -rf ssh ({host})")
        return RunResult(rc=0, stdout="", stderr="")

    monkeypatch.setattr(stack_mod, "down", fake_down)
    monkeypatch.setattr(install_mod, "run", fake_run)
    monkeypatch.setattr(install_mod, "ssh_run", fake_ssh_run)

    cfg = StackConfig(workdir="~/.spectator")

    # Local
    install_mod.uninstall(cfg, host=None)
    assert call_order == ["stack.down", "rm -rf bash"]

    # Remote
    call_order.clear()
    install_mod.uninstall(cfg, host="myspark1-local")
    assert call_order == ["stack.down", "rm -rf ssh (myspark1-local)"]


def test_uninstall_cli_help_advertises_safety_flags() -> None:
    """`spectator uninstall --help` documents both --target and --force."""
    from typer.testing import CliRunner

    from src.cli import app

    runner = CliRunner()
    r = runner.invoke(app, ["uninstall", "--help"])
    assert r.exit_code == 0
    assert "--force" in r.stdout
    assert "--target" in r.stdout
    assert "--workdir" in r.stdout
    # The docstring should make clear this is destructive (mention
    # remove or rm in some form).
    assert "remove" in r.stdout.lower() or "rm " in r.stdout.lower()


def test_deploy_remote_uv_sync_includes_dev_and_writes_install_stamp() -> None:
    """v0.3.9 + v0.4.3: deploy.py's remote uv-sync over SSH must
    (a) install dev deps so the wrapper's deps_ready() probe passes on
    the target, (b) write a content-hash of pyproject.toml into the
    install stamp so the wrapper's fast-path skips require_uv on
    subsequent invocations even after a no-op `./spectator rsync`
    (which bumps pyproject.toml's mtime but leaves content unchanged),
    and (c) fall back to `touch` if neither shasum nor sha256sum is
    available — the wrapper's bootstrap() treats an empty stamp as a
    legacy v0.3.9-v0.4.2 stamp and self-heals.

    Without (a), `./spectator <verb>` on a deployed target fails the
    pytest/ruff existence checks. Without (b), every rsync invalidates
    the fast-path because mtime is newer than the stamp. Without (c),
    a host that lacks both hashing tools would have no stamp at all
    and the wrapper would never take the fast path."""
    import inspect

    from src import deploy

    src_text = inspect.getsource(deploy.deploy)
    assert "uv sync --extra dev" in src_text, (
        "deploy.py's remote uv-sync must use --extra dev so the wrapper's "
        "deps_ready() check passes on the target."
    )
    # Both hashing tools must be tried (shasum first since it's in
    # macOS's default Perl distribution; sha256sum as Linux fallback)
    # before falling through to a bare `touch`.
    assert "shasum -a 256 pyproject.toml" in src_text
    assert "sha256sum pyproject.toml" in src_text
    # cut -d' ' -f1 extracts just the hash (drops the filename column).
    # We use cut, not awk '{print $1}', because the latter's braces
    # collide with f-string interpolation in the heredoc.
    assert "cut -d' ' -f1" in src_text
    assert "touch .venv/.spectator-installed" in src_text  # final fallback
    assert ".spectator-installed" in src_text  # the stamp path itself


def _build_fake_project_for_wrapper_tests(tmp_path):
    """Helper: build a minimal project tree (wrapper + pyproject + fake
    .venv with the executables deps_ready probes for) so the wrapper
    will take its fast path under a sabotaged PATH (no uv).

    Returns (proj_root, install_stamp). Both wrapper-fast-path tests
    use this — keeps the boilerplate together."""
    import os
    import shutil
    import stat
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    wrapper_src = repo_root / "spectator"
    pyproject_src = repo_root / "pyproject.toml"

    proj = tmp_path / "proj"
    proj.mkdir()
    shutil.copy(wrapper_src, proj / "spectator")
    (proj / "spectator").chmod(0o755)
    shutil.copy(pyproject_src, proj / "pyproject.toml")
    (proj / "src").mkdir()
    (proj / "src" / "__init__.py").write_text("")

    venv_bin = proj / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    py_shim = venv_bin / "python"
    py_shim.write_text("#!/bin/sh\nexit 0\n")
    py_shim.chmod(py_shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    for name in ("pytest", "ruff"):
        b = venv_bin / name
        b.write_text("#!/bin/sh\nexit 0\n")
        b.chmod(b.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    stamp = proj / ".venv" / ".spectator-installed"
    return proj, stamp


def _hash_file(path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_wrapper_fast_path_uses_content_hash_not_mtime(tmp_path) -> None:
    """v0.4.3 regression test for the rsync-bumps-mtime trap.

    Pre-v0.4.3, the wrapper's fast-path check used `pyproject.toml -nt
    $INSTALL_STAMP` (mtime comparison). rsync bumps the receiving
    file's mtime even when content is byte-identical, so every
    `./spectator rsync --target HOST` followed by an SSH'd
    `./spectator <verb>` on the remote re-entered bootstrap → fired
    require_uv. On hosts where uv was deleted post-deploy, that
    exploded with 'uv is required' even though .venv was fully
    populated.

    Fix: stamp file now stores the SHA256 of pyproject.toml's content;
    the wrapper compares that hash, not mtime. So an mtime bump from
    rsync that doesn't change content stays in the fast path.

    Test: write the correct hash into the stamp, bump pyproject.toml's
    mtime to be NEWER than the stamp (simulating post-rsync state),
    sabotage PATH so uv is gone, and confirm `./spectator help`
    succeeds without require_uv firing.
    """
    import os
    import shutil
    import subprocess

    proj, stamp = _build_fake_project_for_wrapper_tests(tmp_path)
    pyproject = proj / "pyproject.toml"

    # Record the correct content hash in the stamp.
    expected_hash = _hash_file(pyproject)
    stamp.write_text(expected_hash)

    # Bump pyproject's mtime to simulate post-rsync (newer than stamp,
    # but content unchanged so hash still matches).
    stamp_mtime = stamp.stat().st_mtime
    os.utime(pyproject, (stamp_mtime + 60, stamp_mtime + 60))
    assert pyproject.stat().st_mtime > stamp.stat().st_mtime, (
        "test setup: pyproject must be newer than stamp"
    )

    # Sabotage PATH so uv isn't findable.
    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    assert shutil.which("uv", path=env["PATH"]) is None

    # Use `install` (which calls bootstrap), not `help` (which doesn't).
    r = subprocess.run(
        [str(proj / "spectator"), "install"],
        cwd=str(proj), env=env, capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, (
        f"wrapper failed under sabotaged PATH despite content-hash match: "
        f"rc={r.returncode}\n--stdout--\n{r.stdout}\n--stderr--\n{r.stderr}"
    )
    assert "uv is required" not in r.stderr
    # `install` logs 'ready: ...' after bootstrap returns.
    assert "ready" in r.stdout


def test_wrapper_fast_path_re_syncs_when_pyproject_content_changes(tmp_path) -> None:
    """If pyproject.toml's content actually changed (a real new dep was
    added on the laptop and rsync'd over), the recorded hash no longer
    matches and the wrapper falls through to require_uv → sync. That's
    correct behavior — we'd want the new dep installed before
    forwarding to the CLI.

    Test: write a hash that DOESN'T match the current pyproject's
    content, sabotage PATH, confirm the wrapper now DOES complain
    about uv (proving the fast path got skipped)."""
    import os
    import shutil
    import subprocess

    proj, stamp = _build_fake_project_for_wrapper_tests(tmp_path)

    # Write a bogus hash — simulates "stamp was for an older pyproject
    # content; the current pyproject differs".
    stamp.write_text("0" * 64)  # valid sha256 hex shape, just wrong value

    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    # `install` calls bootstrap directly, so the hash mismatch here
    # forces require_uv → die under the sabotaged PATH.
    r = subprocess.run(
        [str(proj / "spectator"), "install"],
        cwd=str(proj), env=env, capture_output=True, text=True, timeout=10,
    )
    assert "uv is required" in r.stderr, (
        f"wrapper should have demanded uv (hash mismatch → re-sync needed),"
        f" but it succeeded silently:\n--stdout--\n{r.stdout}\n--stderr--\n{r.stderr}"
    )
    assert r.returncode != 0


def test_wrapper_self_heals_legacy_empty_stamp(tmp_path) -> None:
    """A stamp from v0.3.9-v0.4.2 was empty (just `touch`'d). v0.4.3's
    bootstrap() treats that as legacy and takes the fast path while
    writing the new hash for next time. Sabotage PATH so we can also
    verify uv is NOT required for the legacy-stamp path.

    Note: we use `./spectator install` to trigger bootstrap (the verb
    `help` short-circuits before bootstrap runs in the wrapper's case
    statement). `install` calls bootstrap then logs 'ready: ...' — fast
    path with our healthy fake .venv."""
    import os
    import shutil
    import subprocess

    proj, stamp = _build_fake_project_for_wrapper_tests(tmp_path)
    stamp.write_text("")  # legacy empty stamp

    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    r = subprocess.run(
        [str(proj / "spectator"), "install"],
        cwd=str(proj), env=env, capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, (
        f"legacy empty stamp should self-heal under sabotaged PATH, "
        f"but the wrapper failed:\n--stdout--\n{r.stdout}\n--stderr--\n{r.stderr}"
    )
    assert "uv is required" not in r.stderr
    assert "ready" in r.stdout  # confirms the install verb ran past bootstrap

    # The stamp should now have the current pyproject's hash.
    expected = _hash_file(proj / "pyproject.toml")
    assert stamp.read_text().strip() == expected, (
        f"legacy empty stamp wasn't self-healed; "
        f"expected hash {expected!r}, got {stamp.read_text()!r}"
    )


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
