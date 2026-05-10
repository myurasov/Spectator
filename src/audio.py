# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Audio-only transcription via Whisper.

Pure-audio sibling to the VSS video pipeline. Same containment policy as
the rest of Spectator: never writes outside `$workdir` on the target.

Layered surface, per command:

    audio install   — bootstrap a whisper+torch venv at $workdir/audio-venv/
    audio transcribe — rsync local audio to $workdir/audio-in/, run whisper
                       in a tmux session, write results to $workdir/audio-out/
    audio status    — list running jobs + completed transcripts
    audio fetch     — rsync $workdir/audio-out/ back to a local dir
    audio presets   — show the quality presets

Quality presets pick the right Whisper flags for a given recording. The
defaults are tuned by the Whisper run we did against MSFT-NV HQ meeting
audio (large-v3 for the bad source, large-v3-turbo for the cleaner one).
"""

from __future__ import annotations

import os
import shlex
import textwrap
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import _creds, config
from ._run import RunResult, run, rsync_to, ssh_run, ssh_stream

console = Console()


# ---------------------------------------------------------------------------
# layout (paths under $workdir)
# ---------------------------------------------------------------------------

AUDIO_VENV_RELPATH = "audio-venv"
AUDIO_IN_RELPATH = "audio-in"
AUDIO_OUT_RELPATH = "audio-out"
AUDIO_LOG_RELPATH = "audio-logs"


def _expand_tilde(p: str) -> str:
    """Replace a leading ~ with $HOME so the resulting string is a bash
    double-quote-friendly path. Tilde expansion does NOT happen inside
    `'...'` single quotes (which is what shlex.quote produces) but
    `$HOME` expansion DOES happen inside `"..."` double quotes. We need
    the latter for paths with spaces in the basename."""
    if p.startswith("~/"):
        return "$HOME/" + p[2:]
    if p == "~":
        return "$HOME"
    return p


def _bash_dq(s: str) -> str:
    """Wrap s in bash double quotes, escaping `\\`, `"`, and backticks.
    `$VAR` expansion is preserved so `$HOME/...` still works."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`") + '"'


def _venv_dir(cfg: config.StackConfig) -> str:
    return f"{_expand_tilde(cfg.workdir)}/{AUDIO_VENV_RELPATH}"


def _in_dir(cfg: config.StackConfig) -> str:
    return f"{_expand_tilde(cfg.workdir)}/{AUDIO_IN_RELPATH}"


def _out_dir(cfg: config.StackConfig) -> str:
    return f"{_expand_tilde(cfg.workdir)}/{AUDIO_OUT_RELPATH}"


def _log_dir(cfg: config.StackConfig) -> str:
    return f"{_expand_tilde(cfg.workdir)}/{AUDIO_LOG_RELPATH}"


# ---------------------------------------------------------------------------
# quality presets
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Preset:
    name: str
    description: str
    model: str
    flags: dict[str, object]


QUALITY_PRESETS: dict[str, Preset] = {
    "studio": Preset(
        name="studio",
        description="Clean studio-mic recording. Greedy decoding, fastest.",
        model="large-v3-turbo",
        flags={
            "beam_size": 1,
            "best_of": 1,
            "temperature": 0,
            "condition_on_previous_text": True,
            "word_timestamps": True,
        },
    ),
    "meeting": Preset(
        name="meeting",
        description="Default. Standard video-conferencing recording, mixed quality.",
        model="large-v3-turbo",
        flags={
            "beam_size": 5,
            "best_of": 5,
            "patience": 1.0,
            "temperature": 0,
            "temperature_increment_on_fallback": 0.2,
            "condition_on_previous_text": False,
            "word_timestamps": True,
        },
    ),
    "phone": Preset(
        name="phone",
        description="Voice-coded / low-bitrate / phone-call audio (≤ 32 kbps).",
        model="large-v3",
        flags={
            "beam_size": 5,
            "best_of": 5,
            "patience": 1.0,
            "temperature": 0,
            "temperature_increment_on_fallback": 0.2,
            "condition_on_previous_text": False,
            "word_timestamps": True,
            "no_speech_threshold": 0.6,
        },
    ),
    "extreme": Preset(
        name="extreme",
        description="Very poor: distant mic, lots of noise, heavy crosstalk.",
        model="large-v3",
        flags={
            "beam_size": 10,
            "best_of": 10,
            "patience": 2.0,
            "temperature": 0,
            "temperature_increment_on_fallback": 0.2,
            "condition_on_previous_text": False,
            "word_timestamps": True,
            "no_speech_threshold": 0.7,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.5,
        },
    ),
}

DEFAULT_QUALITY = "meeting"


def render_presets() -> None:
    table = Table(title="Audio quality presets")
    table.add_column("preset")
    table.add_column("model")
    table.add_column("decode")
    table.add_column("description")
    for p in QUALITY_PRESETS.values():
        decode = f"beam={p.flags['beam_size']} best_of={p.flags['best_of']}"
        if "patience" in p.flags:
            decode += f" patience={p.flags['patience']}"
        table.add_row(p.name, p.model, decode, p.description)
    console.print(table)


# ---------------------------------------------------------------------------
# install (bootstrap whisper venv on target)
# ---------------------------------------------------------------------------

def install_audio_venv(host: str | None, cfg: config.StackConfig) -> RunResult:
    """Idempotent install: venv + torch (cu128 if GPU, else CPU) + openai-whisper."""
    workdir_bash = _expand_tilde(cfg.workdir)
    script = textwrap.dedent(f'''
        set -e
        export PATH="$HOME/.local/bin:$PATH"

        # Source $workdir/.creds if it exists. v0.4.4: .creds is the
        # source of truth for any creds Spectator needs; we source
        # before anything else so the rest of the script sees the
        # right values.
        {_creds.source_block(workdir_bash)}

        if ! command -v uv >/dev/null 2>&1; then
          echo "==== installing uv ===="
          curl -LsSf https://astral.sh/uv/install.sh | sh
          export PATH="$HOME/.local/bin:$PATH"
        fi

        WORKDIR={_bash_dq(_expand_tilde(cfg.workdir))}
        VENV="$WORKDIR/{AUDIO_VENV_RELPATH}"
        mkdir -p "$WORKDIR/{AUDIO_IN_RELPATH}" "$WORKDIR/{AUDIO_OUT_RELPATH}" "$WORKDIR/{AUDIO_LOG_RELPATH}"

        if [ ! -d "$VENV" ]; then
          echo "==== creating venv: $VENV ===="
          uv venv --python 3.12 "$VENV"
        else
          echo "==== reusing existing venv: $VENV ===="
        fi
        # `uv pip install` doesn't honor UV_PROJECT_ENVIRONMENT (that flag only
        # applies to `uv run` / `uv sync`). For pip operations we must point at
        # the python explicitly via --python.
        PY="$VENV/bin/python"

        # Probe for GPU. If nvidia-smi is present, use the cu128 wheels (works
        # on Blackwell / GB10). Otherwise CPU-only torch. Skipped entirely if
        # torch is already installed.
        if "$PY" -c "import torch" 2>/dev/null; then
          echo "==== torch already installed ===="
        else
          if command -v nvidia-smi >/dev/null 2>&1; then
            echo "==== installing torch (cu128 / GPU) ===="
            uv pip install --python "$PY" torch torchaudio --index-url https://download.pytorch.org/whl/cu128
          else
            echo "==== installing torch (CPU-only — no GPU detected) ===="
            uv pip install --python "$PY" torch torchaudio
          fi
        fi

        if "$PY" -c "import whisper" 2>/dev/null; then
          echo "==== openai-whisper already installed ===="
        else
          echo "==== installing openai-whisper ===="
          uv pip install --python "$PY" -U openai-whisper
        fi

        echo
        echo "==== verify ===="
        "$PY" - <<'PY_EOF'
import torch, whisper
print("torch:", torch.__version__, "  cuda available:", torch.cuda.is_available())
print("whisper:", whisper.__version__)
print("models available (large-v3, large-v3-turbo, etc.):", "ok")
mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
if mps_ok and not torch.cuda.is_available():
    # Apple Silicon path. openai-whisper × torch >= 2.x crashes on MPS for
    # the large-v3 family with "Cannot convert MPS Tensor to float64", so
    # Spectator's auto-detect deliberately skips MPS in favor of CPU. Tell
    # the operator up front so they don't expect MPS perf without opting in.
    print()
    print("Heads-up: MPS detected. Auto-detect will skip it (openai-whisper")
    print("crashes on Apple Silicon GPU with the large-v3 family used by")
    print("every Spectator preset). Override with --device mps per call,")
    print("or set SPECTATOR_ALLOW_MPS_AUTO=1 to re-enable in auto-detect.")
    print("Tracking: https://github.com/openai/whisper/issues/2151")
PY_EOF
        echo
        echo "==== audio-venv ready: $VENV ===="
    ''').strip()
    if host:
        return ssh_run(host, script, env=cfg.env_block())
    return run(["bash", "-c", script])


# ---------------------------------------------------------------------------
# device auto-detection (cuda > mps > cpu, with an MPS-skip default —
# see _detect_device docstring for why)
# ---------------------------------------------------------------------------

VALID_DEVICES = ("cuda", "mps", "cpu")


def _detect_device(cfg: config.StackConfig, host: str | None) -> str:
    """Probe the audio-venv's torch for the best available device.

    Preference order is **cuda > cpu** by default — note that **mps is
    deliberately skipped** unless the operator explicitly opts in via
    ``SPECTATOR_ALLOW_MPS_AUTO=1``. The reason is upstream: the
    ``openai-whisper`` × ``torch >= 2.x`` combination has a known
    crash on Apple Silicon GPUs for the entire ``large-v3`` model
    family — and all four of Spectator's quality presets (``studio``,
    ``meeting``, ``phone``, ``extreme``) use ``large-v3`` or
    ``large-v3-turbo``. Auto-selecting MPS would make a fresh Apple
    Silicon Mac transcribe run crash with::

        TypeError: Cannot convert a MPS Tensor to float64 dtype as the
        MPS framework doesn't support float64. Please use float32 instead.

    See https://github.com/openai/whisper/issues/2151 for upstream
    status. CPU on Apple Silicon (~real-time to 2× slower than
    real-time on M-series for ``meeting`` preset) is the most reliable
    auto-detected fallback until upstream lands a fix.

    Explicit ``--device mps`` (forced via the CLI flag) still works —
    if the user knows what they're asking for (e.g. testing with a
    smaller model that's known to work, or post-fix upstream),
    Spectator will honor it. The CPU-skip is only on the auto-detect
    path.

    If the probe fails (audio-venv missing, torch import broken, ...)
    we return "cpu" so the actual whisper call surfaces a clearer
    error than the probe would.

    Called from `transcribe()` after `install_audio_venv()` has run.
    """
    venv_py = _bash_dq(f"{_venv_dir(cfg)}/bin/python")
    script = textwrap.dedent(f'''
        if [ ! -x {venv_py} ]; then
            echo cpu
            exit 0
        fi
        {venv_py} - <<'PY_EOF'
import torch

if torch.cuda.is_available():
    print("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY_EOF
    ''').strip()
    if host:
        r = ssh_run(host, script)
    else:
        r = run(["bash", "-c", script])
    if not r.ok or not r.stdout.strip():
        return "cpu"
    detected = r.stdout.strip().splitlines()[-1].strip()
    if detected not in VALID_DEVICES:
        return "cpu"
    if detected == "mps" and not os.environ.get("SPECTATOR_ALLOW_MPS_AUTO"):
        # Downgrade auto-detected MPS to CPU because openai-whisper
        # crashes on MPS with the large-v3 family (which all four
        # Spectator presets use). Override with SPECTATOR_ALLOW_MPS_AUTO=1
        # if you've patched whisper locally or are testing a smaller
        # model that's known to work.
        console.print(
            "[yellow]MPS detected but auto-skipped[/yellow]: openai-whisper "
            "currently crashes on Apple Silicon GPU with the large-v3 family "
            "(\"Cannot convert MPS Tensor to float64\"). Falling back to "
            "[cyan]cpu[/cyan]. Override with [italic]"
            "SPECTATOR_ALLOW_MPS_AUTO=1[/italic] or [italic]--device mps[/italic] "
            "(both honored). Tracking: "
            "[link=https://github.com/openai/whisper/issues/2151]"
            "openai/whisper#2151[/link]."
        )
        return "cpu"
    return detected


# ---------------------------------------------------------------------------
# build the whisper command for a transcribe run
# ---------------------------------------------------------------------------

def _whisper_command(audio_basename: str, cfg: config.StackConfig,
                     preset: Preset,
                     model_override: str | None,
                     language: str | None,
                     task: str,
                     device: str,
                     clip: str | None,
                     initial_prompt: str | None,
                     output_subdir: str) -> str:
    """Return the bash one-liner that runs whisper on the target.

    `language=None` (or "auto") drops the `--language` flag entirely so
    Whisper auto-detects per 30-second window — the right setting for
    bilingual / code-switched recordings. Pass an ISO-639-1 code (en,
    es, ru, hi, ja, zh, fr, de, pt, …) to lock the language.

    `task` is "transcribe" (default — same language as audio) or
    "translate" (Whisper translates non-English speech into English
    while keeping timestamps).
    """
    flags = []
    flags.append(f"--model {model_override or preset.model}")
    if language and language.lower() != "auto":
        flags.append(f"--language {language}")
    flags.append(f"--task {task}")
    flags.append(f"--device {device}")
    # fp16 only on CUDA. MPS has long-standing fp16 quality issues with
    # openai-whisper (garbage segments at boundaries); CPU doesn't support
    # fp16 at all in whisper. fp32 is the safe default for non-CUDA paths.
    flags.append("--fp16 True" if device == "cuda" else "--fp16 False")
    for k, v in preset.flags.items():
        if isinstance(v, bool):
            flags.append(f"--{k} {v}")
        else:
            flags.append(f"--{k} {v}")
    flags.append("--output_format all")
    flags.append(f"--output_dir {_bash_dq(f'{_out_dir(cfg)}/{output_subdir}')}")
    flags.append("--verbose True")
    if clip:
        flags.append(f"--clip_timestamps {shlex.quote(clip)}")
    if initial_prompt:
        flags.append(f"--initial_prompt {shlex.quote(initial_prompt)}")

    audio_path = f"{_in_dir(cfg)}/{audio_basename}"
    whisper_bin = f"{_venv_dir(cfg)}/bin/whisper"
    return f"{_bash_dq(whisper_bin)} {_bash_dq(audio_path)} " + " ".join(flags)


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------

def transcribe(
    audio_local: Path,
    *,
    host: str | None,
    cfg: config.StackConfig,
    quality: str = DEFAULT_QUALITY,
    model: str | None = None,
    language: str | None = None,
    task: str = "transcribe",
    clip: str | None = None,
    initial_prompt: str | None = None,
    session_name: str | None = None,
    detach: bool | None = None,
    follow: bool | None = None,
    skip_upload: bool = False,
    device_override: str | None = None,
) -> RunResult:
    """Upload audio (if not skip_upload) and run whisper.

    `detach=None` → auto-detach when host is set, foreground when local.
    `follow=None` → auto-follow when detach is set (so user sees live
    progress in the console). Ctrl-C exits the tail; the underlying tmux
    job keeps running. With `follow=False` the call returns immediately
    after starting the tmux session — laptop-close-safe.

    `device_override=None` (the default) auto-detects the best torch
    device available in the audio-venv (cuda > mps > cpu). Pass an
    explicit string to force a specific device — useful when a host has
    a GPU but you want to test the CPU path, or when MPS is detected
    but a model is known not to work well on it.
    """
    if quality not in QUALITY_PRESETS:
        raise ValueError(f"unknown quality preset: {quality!r}; "
                         f"choose from {sorted(QUALITY_PRESETS)}")
    preset = QUALITY_PRESETS[quality]

    basename = audio_local.name
    output_subdir = audio_local.stem  # transcripts land in audio-out/<stem>/
    if session_name is None:
        # tmux dislikes spaces and dots in session names
        safe = "".join(c if c.isalnum() else "-" for c in audio_local.stem)[:40]
        session_name = f"audio-{safe}"
    if detach is None:
        detach = host is not None
    if follow is None:
        # By default, follow when we detach (typical interactive use:
        # user wants to see progress without keeping a synchronous run
        # in the foreground). They can Ctrl-C the tail at any time
        # without stopping the underlying tmux job.
        follow = detach and host is not None

    # 1. ensure target dirs exist
    setup = textwrap.dedent(f'''
        set -e
        mkdir -p {_in_dir(cfg)} {_out_dir(cfg)}/{shlex.quote(output_subdir)} {_log_dir(cfg)}
    ''').strip()
    if host:
        r = ssh_run(host, setup)
    else:
        r = run(["bash", "-c", setup])
    if not r.ok:
        return r

    # 2. upload (rsync) if remote and not skip_upload
    if host and not skip_upload:
        size = audio_local.stat().st_size
        console.print(
            f"[bold]→[/bold] uploading {audio_local.name} "
            f"({size / 1e6:.1f} MB) to {host}:{_in_dir(cfg)}/"
        )
        # Stream rsync's --progress output to the terminal in real time
        # (the default `_run.run` captures, which silences progress on big files).
        import subprocess
        proc = subprocess.run(
            ["rsync", "-avh", "--progress",
             str(audio_local),
             f"{host}:{_in_dir(cfg)}/"],
        )
        if proc.returncode != 0:
            console.print(f"[red]upload failed (rc={proc.returncode})[/red]")
            return RunResult(rc=proc.returncode, stdout="", stderr="")

    if not host and not skip_upload:
        # local mode: copy the file into audio-in/ so output paths stay consistent.
        # (Skip when --skip-upload — caller has already placed it.)
        # NB: we go through `Path.expanduser` on cfg.workdir directly here,
        # NOT through `_in_dir(cfg)`. The latter returns a bash-friendly
        # form that prefixes a literal `$HOME/...` (so it round-trips
        # safely through bash heredocs that quote-escape `~`); but
        # `os.path.expanduser` only handles a leading `~`, leaving any
        # `$HOME` token intact — which silently created a literal "$HOME"
        # directory under the subprocess cwd before this fix.
        import shutil
        dest = Path(cfg.workdir).expanduser() / AUDIO_IN_RELPATH / basename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.resolve() != audio_local.resolve():
            shutil.copy2(audio_local, dest)

    # 3. detect device + build the whisper command
    if device_override is not None:
        if device_override not in VALID_DEVICES:
            raise ValueError(f"unknown device {device_override!r}; "
                             f"choose from {VALID_DEVICES}")
        device = device_override
    else:
        device = _detect_device(cfg, host)
    console.print(f"  device: [cyan]{device}[/cyan]"
                  + (" (auto-detected)" if device_override is None else " (forced via --device)"))
    cmd = _whisper_command(
        audio_basename=basename,
        cfg=cfg,
        preset=preset,
        model_override=model,
        language=language,
        task=task,
        device=device,
        clip=clip,
        initial_prompt=initial_prompt,
        output_subdir=output_subdir,
    )

    log_path = f"{_log_dir(cfg)}/{session_name}.log"

    # 4. dispatch
    if detach:
        # Tmux session — survives ssh disconnect. We write a runner script
        # to disk first (rather than passing the command inline) so $(date),
        # $?, and ${{PIPESTATUS[0]}} expand at *run* time inside tmux's
        # shell, not at *build* time in the outer ssh-bash that creates the
        # tmux session. Inline-quoted forms expand once too early and bake
        # the start timestamp into the END line.
        runner_path = f"{_log_dir(cfg)}/{session_name}.sh"
        wrapper = textwrap.dedent(f'''
            set -e
            : > {log_path}
            if tmux has-session -t {shlex.quote(session_name)} 2>/dev/null; then
              echo "✗ tmux session {session_name} already running" >&2
              exit 1
            fi
            cat > {runner_path} <<'RUNNER_EOF'
#!/bin/bash
# PYTHONUNBUFFERED so whisper's per-segment lines flush per-write
# (Python defaults to 4KB block buffering when stdout is a file/pipe,
# which makes `tail -f` of this log feel stalled for many seconds).
export PYTHONUNBUFFERED=1
exec >> {log_path} 2>&1
echo "==== START $(date) ===="
set -o pipefail
{cmd}
RC=$?
echo "==== END rc=$RC $(date) ===="
RUNNER_EOF
            chmod +x {runner_path}
            tmux new-session -d -s {shlex.quote(session_name)} {runner_path}
            sleep 2
            echo "tmux: $(tmux list-sessions | grep {shlex.quote(session_name)})"
            echo "log : {log_path}"
            echo "out : {_out_dir(cfg)}/{output_subdir}"
            tail -10 {log_path} 2>/dev/null
        ''').strip()
    else:
        # Non-detach branch — synchronous run. END line must match the
        # detach branch's `==== END rc=$RC <date> ====` shape so the
        # WebUI's progress parser (`webui/progress.py:_END_RE`) detects
        # completion uniformly. Pre-v0.3.7 this branch emitted just
        # `==== END <date> ====` (no `rc=`), and the parser silently
        # missed the marker → the WS handler fell through to its
        # "subprocess died without END marker" sentinel and mislabeled
        # successful local jobs as failed.
        #
        # We capture $? rather than relying on `set -e` so a non-zero
        # exit is reported in the END line (rc=N) instead of skipping
        # it entirely — same shape as the runner.sh script in the
        # detach branch.
        wrapper = textwrap.dedent(f'''
            echo "==== START $(date) ===="
            set -o pipefail
            {cmd}
            RC=$?
            echo "==== END rc=$RC $(date) ===="
            exit $RC
        ''').strip()

    if host:
        r = ssh_run(host, wrapper, env=cfg.env_block())
    else:
        r = run(["bash", "-c", wrapper])

    if not r.ok:
        return r
    console.print(r.stdout)

    if detach and follow and host is not None:
        console.print(
            f"\n[bold]→[/bold] live progress from [italic]{log_path}[/italic]\n"
            f"  (Ctrl-C to detach; the tmux session [cyan]{session_name}[/cyan] "
            f"will keep running on {host}.)\n"
        )
        try:
            # `tail -F` follows by name (handles log rotation, file swaps).
            # `awk` exits cleanly the moment the END line appears, so the
            # user gets back to a prompt automatically when the job ends.
            ssh_stream(
                host,
                f"tail -n +1 -F {log_path} | "
                f"awk '{{print}} /==== END rc=/{{exit 0}}'",
            )
        except KeyboardInterrupt:
            console.print(
                f"\n[yellow]detached.[/yellow] tmux session [cyan]{session_name}[/cyan] "
                f"is still running on {host}. Re-attach with:\n"
                f"  spectator audio status --target {host}\n"
                f"  ssh {host} 'tail -f {log_path}'"
            )
    return r


# ---------------------------------------------------------------------------
# status / fetch
# ---------------------------------------------------------------------------

def status(host: str | None, cfg: config.StackConfig) -> RunResult:
    script = textwrap.dedent(f'''
        echo "==== running whisper jobs (audio-* tmux sessions) ===="
        tmux list-sessions 2>/dev/null | grep -E "^audio-" || echo "(none)"
        echo
        echo "==== recent log lines (per session) ===="
        for L in {_log_dir(cfg)}/audio-*.log; do
          [ -f "$L" ] || continue
          echo "--- $(basename $L) ---"
          tail -3 "$L"
        done 2>/dev/null
        echo
        echo "==== completed transcripts in {_out_dir(cfg)} ===="
        find {_out_dir(cfg)} -mindepth 1 -maxdepth 2 -name "*.txt" 2>/dev/null | head -20
    ''').strip()
    if host:
        return ssh_run(host, script)
    return run(["bash", "-c", script])


def fetch(host: str | None, cfg: config.StackConfig,
          dest: Path, only: str | None = None) -> RunResult:
    """rsync `$workdir/audio-out/[<only>/]` to `dest/`.

    Shell-quotes the remote path so stems with spaces / parens /
    brackets / ``$`` / backticks survive the remote-shell parsing.
    Without quoting, ``rsync host:/path/with spaces/`` is split by the
    remote shell into multiple paths, and each piece is treated as a
    separate source — invariably failing with ``link_stat ... No such
    file or directory`` on every fragment. Real-world trip-ups are
    most often macOS meeting recordings with spaces in the basename
    (e.g. ``"My Recording.mp3"`` → stem ``My Recording`` → fetch
    fails). v0.4.2 fix.
    """
    dest.mkdir(parents=True, exist_ok=True)
    src = f"{cfg.workdir}/{AUDIO_OUT_RELPATH}/"
    if only:
        src += only.rstrip("/") + "/"
        dest = dest / only.rstrip("/")
        dest.mkdir(parents=True, exist_ok=True)
    if host:
        # shlex.quote single-quotes the path; the remote shell sees
        # one argument and rsync forwards it intact. No further
        # escaping needed because we don't expand $VAR / `...` in
        # paths. Local rsync (else branch below) doesn't need this
        # because Python-level argv stays a single argument.
        remote_arg = f"{host}:{shlex.quote(src)}"
        return run(["rsync", "-avh", "--progress",
                    remote_arg, str(dest) + "/"])
    return run(["rsync", "-avh", str(src), str(dest) + "/"])
