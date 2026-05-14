# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Tests for the diarize module + CLI surface.

Network-free: every test that touches the diarize pipeline mocks the
underlying SSH / subprocess primitives. The actual pyannote run is
exercised manually on a GPU host (see REFERENCE.md § "audio diarize").
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from src._run import RunResult

# ---------------------------------------------------------------------------
# imports + module surface
# ---------------------------------------------------------------------------

def test_diarize_module_imports() -> None:
    """Smoke check that the module is loadable from a clean interpreter."""
    import src.diarize  # noqa: F401


def test_diarize_module_exports_public_api() -> None:
    """The four public entry points the CLI + audio.transcribe rely on."""
    from src import diarize

    for name in (
        "DEFAULT_DIARIZE_MODEL",
        "HF_TOKEN_ENV_VARS",
        "PYANNOTE_PIN",
        "build_diarize_command",
        "build_merge_command",
        "diarize",
        "install_diarize_deps",
        "merge_whisper_with_diarization",
        "merged_segments_to_txt",
    ):
        assert hasattr(diarize, name), f"diarize.{name} missing"


def test_default_model_is_pyannote_3_1() -> None:
    """Pin the default away from accidental bumps; 3.1 is the validated baseline."""
    from src import diarize

    assert diarize.DEFAULT_DIARIZE_MODEL == "pyannote/speaker-diarization-3.1"


def test_pyannote_pin_targets_4x_line() -> None:
    """4.x is the current import-clean line against torchaudio >= 2.2;
    the 3.x line broke on ``torchaudio.AudioMetaData`` removal."""
    from src import diarize

    assert "pyannote.audio>=4.0" in diarize.PYANNOTE_PIN
    assert "<5" in diarize.PYANNOTE_PIN


def test_hf_token_env_vars_priority() -> None:
    """Canonical name first; HF_TOKEN as alias."""
    from src import diarize

    assert diarize.HF_TOKEN_ENV_VARS[0] == "HUGGING_FACE_HUB_TOKEN"
    assert "HF_TOKEN" in diarize.HF_TOKEN_ENV_VARS


def test_creds_vars_includes_hf_token() -> None:
    """The HF token must round-trip through $workdir/.creds via _creds.CREDS_VARS."""
    from src._creds import CREDS_VARS

    assert "HUGGING_FACE_HUB_TOKEN" in CREDS_VARS


def test_ensure_var_block_rejects_unknown_var() -> None:
    """ensure_var_block must refuse vars not declared in CREDS_VARS;
    that's the single source of truth for what round-trips through the
    creds file. Catching the typo here beats silently writing a var
    that source_block / save_block don't know about."""
    from src import _creds

    with pytest.raises(ValueError, match="CREDS_VARS"):
        _creds.ensure_var_block("/tmp/wd", "NOT_A_REAL_VAR")


def test_ensure_var_block_renders_idempotent_append() -> None:
    """The rendered bash must guard with `if [ -n ... ]`, create the
    file when missing, and grep before appending — three properties
    that together make the block safe to run on every diarize call."""
    from src import _creds

    block = _creds.ensure_var_block("/home/u/.spectator", "HUGGING_FACE_HUB_TOKEN")
    # Guard: skip when var is unset.
    assert 'if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then' in block
    # Create-if-missing path.
    assert 'if [ ! -f "$CREDS_FILE" ]; then' in block
    assert 'chmod 600 "$CREDS_FILE"' in block
    # Idempotent append: grep first, then printf %q the value.
    assert 'grep -q "^export HUGGING_FACE_HUB_TOKEN=" "$CREDS_FILE"' in block
    assert 'printf "export HUGGING_FACE_HUB_TOKEN=%s' in block
    # Path is interpolated correctly (no naked literal `~`).
    assert "/home/u/.spectator/.creds" in block


def test_build_diarize_command_includes_creds_source_and_ensure_blocks() -> None:
    """The diarize bash payload must source `.creds` AND ensure
    HUGGING_FACE_HUB_TOKEN is persisted to it — that's the
    "passes --hf-token once, future calls don't need it" contract."""
    from src.config import StackConfig
    from src.diarize import build_diarize_command

    cfg = StackConfig(workdir="~/.spectator")
    cmd = build_diarize_command(
        audio_basename="x.mp3", cfg=cfg,
        model="pyannote/speaker-diarization-3.1",
        num_speakers=None, min_speakers=None, max_speakers=None,
        device="cuda", stem="x",
    )
    # source_block runs first (token may already be in .creds).
    assert '. "$HOME/.spectator/.creds"' in cmd
    # ensure_var_block runs next (token from --hf-token gets persisted).
    assert 'grep -q "^export HUGGING_FACE_HUB_TOKEN=" "$CREDS_FILE"' in cmd


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def test_audio_diarize_subcommand_runs() -> None:
    """`spectator audio diarize --help` advertises the expected flags."""
    from src.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["audio", "diarize", "--help"])
    assert result.exit_code == 0, result.stdout
    for flag in (
        "--target",
        "--workdir",
        "--model",
        "--num-speakers",
        "--min-speakers",
        "--max-speakers",
        "--hf-token",
        "--device",
        "--session",
        "--detach",
        "--follow",
        "--skip-upload",
        "--auto-merge",
    ):
        assert flag in result.stdout, f"missing flag {flag} in `audio diarize --help`"


def test_audio_transcribe_help_advertises_diarize_flags() -> None:
    """The diarize family must appear in `audio transcribe --help`."""
    from src.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["audio", "transcribe", "--help"])
    assert result.exit_code == 0, result.stdout
    for flag in (
        "--diarize",
        "--diarize-model",
        "--num-speakers",
        "--min-speakers",
        "--max-speakers",
        "--hf-token",
    ):
        assert flag in result.stdout, f"missing flag {flag} in `audio transcribe --help`"


def test_audio_install_help_advertises_with_diarize_flag() -> None:
    """`audio install --help` must surface the new --with-diarize toggle."""
    from src.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["audio", "install", "--help"])
    assert result.exit_code == 0, result.stdout
    assert "--with-diarize" in result.stdout
    assert "--no-with-diarize" in result.stdout


def test_audio_transcribe_rejects_num_speakers_with_range() -> None:
    """--num-speakers is mutually exclusive with --min/--max-speakers."""
    from src.cli import app

    runner = CliRunner()
    # Use a path that exists so we don't trip Typer's `exists=True` check
    # before reaching our own validation.
    import sys
    result = runner.invoke(app, [
        "audio", "transcribe", sys.executable,
        "--diarize",
        "--num-speakers", "3",
        "--min-speakers", "2",
    ])
    assert result.exit_code == 2
    assert "mutually exclusive" in (result.stdout + result.stderr)


def test_audio_diarize_rejects_inverted_speaker_range() -> None:
    """--min-speakers must be <= --max-speakers."""
    from src.cli import app

    runner = CliRunner()
    import sys
    result = runner.invoke(app, [
        "audio", "diarize", sys.executable,
        "--min-speakers", "5",
        "--max-speakers", "2",
    ])
    assert result.exit_code == 2
    assert "must be <=" in (result.stdout + result.stderr)


# ---------------------------------------------------------------------------
# merge logic — the algorithm the user cares most about, exercised
# against handcrafted whisper/diarization pairs
# ---------------------------------------------------------------------------

def test_merge_assigns_speaker_by_max_overlap() -> None:
    """Each whisper segment picks the speaker with the largest overlap."""
    from src.diarize import merge_whisper_with_diarization

    whisper = [
        {"id": 0, "start": 0.0, "end": 5.0, "text": "alice's line"},
        {"id": 1, "start": 5.0, "end": 10.0, "text": "bob's line"},
    ]
    diar = [
        {"start": 0.0, "end": 4.5, "speaker": "SPEAKER_00"},
        {"start": 4.5, "end": 10.5, "speaker": "SPEAKER_01"},
    ]
    merged, stats = merge_whisper_with_diarization(whisper, diar)
    assert merged[0]["speaker"] == "SPEAKER_00"
    assert merged[1]["speaker"] == "SPEAKER_01"
    assert stats.segments_total == 2
    assert stats.segments_with_speaker == 2
    assert stats.segments_without_overlap == 0
    assert stats.speakers_seen == ("SPEAKER_00", "SPEAKER_01")


def test_merge_returns_none_for_segments_with_no_overlap() -> None:
    """Segments outside every diarization turn get speaker=None."""
    from src.diarize import merge_whisper_with_diarization

    whisper = [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "before any diarized speech"},
        {"id": 1, "start": 5.0, "end": 6.0, "text": "covered"},
    ]
    diar = [
        {"start": 4.0, "end": 7.0, "speaker": "SPEAKER_00"},
    ]
    merged, stats = merge_whisper_with_diarization(whisper, diar)
    assert merged[0]["speaker"] is None
    assert merged[1]["speaker"] == "SPEAKER_00"
    assert stats.segments_with_speaker == 1
    assert stats.segments_without_overlap == 1


def test_merge_breaks_ties_alphabetically() -> None:
    """Equal overlap → pick the alphabetically-first speaker label."""
    from src.diarize import merge_whisper_with_diarization

    # 10-second segment bisected exactly: 5s of SPEAKER_01, 5s of SPEAKER_00.
    # Both contribute equal overlap; the lexicographically smaller wins.
    whisper = [{"id": 0, "start": 0.0, "end": 10.0, "text": "split"}]
    diar = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_01"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_00"},
    ]
    merged, _ = merge_whisper_with_diarization(whisper, diar)
    assert merged[0]["speaker"] == "SPEAKER_00"


def test_merge_handles_partial_overlap_with_multiple_speakers() -> None:
    """Segment spans three turns; pick the one with the largest cumulative overlap."""
    from src.diarize import merge_whisper_with_diarization

    whisper = [{"id": 0, "start": 0.0, "end": 10.0, "text": "spans three"}]
    diar = [
        # SPEAKER_00 contributes 2s
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        # SPEAKER_01 contributes 7s (the winner)
        {"start": 2.0, "end": 9.0, "speaker": "SPEAKER_01"},
        # SPEAKER_00 contributes another 1s (cumulative SPEAKER_00 = 3s)
        {"start": 9.0, "end": 10.0, "speaker": "SPEAKER_00"},
    ]
    merged, _ = merge_whisper_with_diarization(whisper, diar)
    assert merged[0]["speaker"] == "SPEAKER_01"


def test_merge_does_not_mutate_input_segments() -> None:
    """The merge returns shallow copies; the caller's list stays clean."""
    from src.diarize import merge_whisper_with_diarization

    whisper = [{"id": 0, "start": 0.0, "end": 5.0, "text": "x"}]
    diar = [{"start": 0.0, "end": 5.0, "speaker": "S"}]
    merged, _ = merge_whisper_with_diarization(whisper, diar)
    assert "speaker" not in whisper[0]
    assert merged[0]["speaker"] == "S"
    assert merged[0] is not whisper[0]


def test_merged_segments_to_txt_groups_consecutive_same_speaker() -> None:
    """Consecutive same-speaker segments collapse into one labeled block."""
    from src.diarize import merged_segments_to_txt

    merged = [
        {"start": 0.0, "end": 2.0, "speaker": "A", "text": "hello"},
        {"start": 2.0, "end": 4.0, "speaker": "A", "text": "world"},
        {"start": 4.0, "end": 6.0, "speaker": "B", "text": "goodbye"},
    ]
    text = merged_segments_to_txt(merged)
    assert text.count("] A:") == 1
    assert text.count("] B:") == 1
    assert "hello" in text
    assert "world" in text
    assert "goodbye" in text


def test_merged_segments_to_txt_labels_unknown_speaker() -> None:
    """speaker=None blocks are rendered as `(unknown)`."""
    from src.diarize import merged_segments_to_txt

    merged = [
        {"start": 0.0, "end": 2.0, "speaker": None, "text": "anon"},
    ]
    text = merged_segments_to_txt(merged)
    assert "(unknown)" in text


# ---------------------------------------------------------------------------
# bash payload construction — pin the shape so a regression doesn't sneak
# a quoting bug back in
# ---------------------------------------------------------------------------

def test_build_diarize_command_quotes_path_with_spaces() -> None:
    """A stem with spaces in the basename must survive the bash payload."""
    from src.config import StackConfig
    from src.diarize import build_diarize_command

    cfg = StackConfig(workdir="~/.spectator")
    cmd = build_diarize_command(
        audio_basename="My Recording.mp3",
        cfg=cfg,
        model="pyannote/speaker-diarization-3.1",
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
        device="cuda",
        stem="My Recording",
    )
    # Path body interpolated under bash double-quotes; the basename has
    # to land verbatim so pyannote opens the right file.
    assert "My Recording.mp3" in cmd
    assert "My Recording" in cmd
    # Token sourcing happens at the head so .creds is read before Python runs.
    assert "/.creds" in cmd


def test_build_diarize_command_pipeline_call_has_no_trailing_comma() -> None:
    """No constraint args ⇒ ``pipeline(AUDIO_PATH)`` (no stray comma)."""
    from src.config import StackConfig
    from src.diarize import build_diarize_command

    cfg = StackConfig(workdir="~/.spectator")
    cmd = build_diarize_command(
        audio_basename="x.mp3", cfg=cfg,
        model="pyannote/speaker-diarization-3.1",
        num_speakers=None, min_speakers=None, max_speakers=None,
        device="cpu", stem="x",
    )
    assert "pipeline(AUDIO_PATH)" in cmd
    assert "pipeline(AUDIO_PATH, )" not in cmd


def test_build_diarize_command_emits_num_speakers_constraint() -> None:
    """--num-speakers maps to ``pipeline(AUDIO_PATH, num_speakers=N)``."""
    from src.config import StackConfig
    from src.diarize import build_diarize_command

    cfg = StackConfig(workdir="~/.spectator")
    cmd = build_diarize_command(
        audio_basename="x.mp3", cfg=cfg,
        model="pyannote/speaker-diarization-3.1",
        num_speakers=7, min_speakers=None, max_speakers=None,
        device="cuda", stem="x",
    )
    assert "pipeline(AUDIO_PATH, num_speakers=7)" in cmd


def test_build_diarize_command_emits_min_max_constraints() -> None:
    """--min/--max-speakers map to keyword args; can coexist."""
    from src.config import StackConfig
    from src.diarize import build_diarize_command

    cfg = StackConfig(workdir="~/.spectator")
    cmd = build_diarize_command(
        audio_basename="x.mp3", cfg=cfg,
        model="pyannote/speaker-diarization-3.1",
        num_speakers=None, min_speakers=2, max_speakers=5,
        device="cuda", stem="x",
    )
    assert "min_speakers=2" in cmd
    assert "max_speakers=5" in cmd


def test_install_audio_venv_with_diarize_includes_pyannote_install_block() -> None:
    """`audio install` (default `with_diarize=True`) installs pyannote idempotently."""
    from src import audio
    from src.config import StackConfig

    captured_script: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        captured_script.append(cmd[-1] if isinstance(cmd, list) else cmd)
        return RunResult(rc=0, stdout="", stderr="")

    orig = audio.run
    audio.run = fake_run
    try:
        audio.install_audio_venv(None, StackConfig(workdir="~/.spectator"), with_diarize=True)
    finally:
        audio.run = orig

    script = captured_script[-1]
    assert "import pyannote.audio" in script
    assert "pyannote.audio>=4.0,<5" in script


def test_install_audio_venv_without_diarize_omits_pyannote_block() -> None:
    """`audio install --no-with-diarize` skips the pyannote install entirely."""
    from src import audio
    from src.config import StackConfig

    captured_script: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        captured_script.append(cmd[-1] if isinstance(cmd, list) else cmd)
        return RunResult(rc=0, stdout="", stderr="")

    orig = audio.run
    audio.run = fake_run
    try:
        audio.install_audio_venv(None, StackConfig(workdir="~/.spectator"), with_diarize=False)
    finally:
        audio.run = orig

    script = captured_script[-1]
    # Neither the install nor the verify block should reference pyannote
    # when with_diarize=False — that's the whole point of opting out.
    assert "pyannote.audio>=4.0,<5" not in script
    assert "import pyannote.audio" not in script


def test_transcribe_with_diarize_chains_pyannote_into_runner(monkeypatch) -> None:
    """`audio transcribe --diarize` appends the diarize + merge commands
    to the whisper runner so a single tmux session handles the whole
    pipeline. Pin the subshell wrapping so a future edit can't silently
    regress to ``exit $WHISPER_RC`` (which terminates the wrapper and
    skips the ``==== END rc=... ====`` marker)."""
    from pathlib import Path

    from src import audio
    from src.config import StackConfig

    captured: list[str] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        captured.append(cmd[-1] if isinstance(cmd, list) else cmd)
        return RunResult(rc=0, stdout="", stderr="")

    monkeypatch.setattr(audio, "run", fake_run)
    monkeypatch.setattr(audio, "_detect_device", lambda *a, **kw: "cpu")

    fake_audio = Path("/tmp/fake-audio.mp3")
    fake_audio.touch()
    try:
        audio.transcribe(
            fake_audio,
            host=None,
            cfg=StackConfig(workdir="~/.spectator"),
            diarize=True,
            skip_upload=True,
            detach=False,
        )
    finally:
        fake_audio.unlink(missing_ok=True)

    # The synchronous (non-detach) wrapper is the last captured script.
    wrapper = captured[-1]
    assert "==== whisper done, starting diarization ====" in wrapper
    assert "==== diarization done, merging ====" in wrapper
    # Subshell wrapping keeps `exit $WHISPER_RC` local to the chain;
    # the outer wrapper's `RC=$?` + END line still fires.
    assert "(\n" in wrapper or wrapper.count("(") >= 1
    assert "WHISPER_RC=$?" in wrapper
    assert "DIARIZE_RC=$?" in wrapper
    # Pin: the `pipeline(AUDIO_PATH)` snippet (from build_diarize_command)
    # was actually inlined — proves the chain reaches the diarize step.
    assert "pipeline(AUDIO_PATH)" in wrapper
