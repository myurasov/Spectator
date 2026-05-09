# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Subprocess wrapper for spawning audio / video jobs from the Web UI.

The Web UI doesn't reimplement the audio or video pipelines — it shells out
to the existing ``./spectator audio transcribe`` / ``./spectator process``
CLI commands and tracks the resulting subprocess's lifecycle. This keeps
the orchestration layer thin and ensures the Web UI stays in sync with any
future improvements to the CLI without dual-implementation drift.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from .. import config


def _python_m_src() -> list[str]:
    """The canonical "invoke the Spectator CLI" prefix.

    Resolves to ``<sys.executable> -m src``; the caller must arrange for
    ``PYTHONPATH`` to contain the project root so ``src`` is importable.
    """
    return [sys.executable, "-m", "src"]


def project_root() -> Path:
    """Return the project root (parent of ``src/``) for PYTHONPATH purposes."""
    return Path(__file__).resolve().parents[2]


def _env_with_pythonpath() -> dict[str, str]:
    env = dict(os.environ)
    root = str(project_root())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}:{existing}" if existing else root
    return env


def spawn_audio_transcribe(
    *,
    audio_path: str,
    log_path: str,
    target: str | None,
    workdir: str,
    quality: str | None,
    language: str | None,
    task: str,
    model: str | None,
    device: str | None,
    skip_upload: bool = False,
) -> int:
    """Spawn ``spectator audio transcribe ...`` in the background.

    Returns the child PID. Stdout + stderr go to ``log_path`` (appended).
    The caller is responsible for tracking the PID and waiting on / killing
    the child via :func:`kill_pid`.
    """
    cmd = [*_python_m_src(), "audio", "transcribe", audio_path,
           "--workdir", workdir,
           "--task", task,
           "--no-detach", "--no-follow"]
    if target:
        cmd += ["--target", target]
        # When running against a remote target, Spectator's transcribe()
        # auto-detaches into tmux on the remote and follows the log over
        # SSH. Forcing --no-detach there breaks the workflow, so flip back.
        cmd[cmd.index("--no-detach")] = "--detach"
        cmd[cmd.index("--no-follow")] = "--follow"
    if quality:
        cmd += ["--quality", quality]
    if language:
        cmd += ["--language", language]
    if model:
        cmd += ["--model", model]
    if device:
        cmd += ["--device", device]
    if skip_upload:
        cmd += ["--skip-upload"]

    log = open(log_path, "ab")
    proc = subprocess.Popen(
        cmd,
        stdout=log, stderr=subprocess.STDOUT,
        cwd=str(project_root()),
        env=_env_with_pythonpath(),
        # Detach from the server's controlling terminal so the child
        # survives a uvicorn reload (jobs outlive the WebUI process if
        # anyone restarts it mid-run).
        start_new_session=True,
    )
    log.close()
    return proc.pid


def spawn_video_process(
    *,
    video_path: str,
    log_path: str,
    target: str | None,
    prompt: str | None,
    output_path: str,
) -> int:
    """Spawn ``spectator process VIDEO ...`` in the background.

    Same conventions as :func:`spawn_audio_transcribe`.
    """
    cmd = [*_python_m_src(), "process", video_path, "--output", output_path]
    if target:
        cmd += ["--target", target]
    if prompt:
        cmd += ["--prompt", prompt]

    log = open(log_path, "ab")
    proc = subprocess.Popen(
        cmd,
        stdout=log, stderr=subprocess.STDOUT,
        cwd=str(project_root()),
        env=_env_with_pythonpath(),
        start_new_session=True,
    )
    log.close()
    return proc.pid


def is_pid_alive(pid: int) -> bool:
    """Cheap check: send signal 0 and look for ``ProcessLookupError``."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Pid exists but we can't signal it — still alive from our POV.
        return True


def kill_pid(pid: int, *, sig: signal.Signals = signal.SIGTERM) -> bool:
    """Send a signal to ``pid``. Returns True if the signal was delivered."""
    if not is_pid_alive(pid):
        return False
    try:
        # Kill the entire process group so subprocesses (whisper, ssh, etc.)
        # also exit. Spawn used start_new_session=True so the child is its
        # own process-group leader.
        os.killpg(os.getpgid(pid), sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def kill_remote_tmux(host: str, session: str) -> bool:
    """SSH to ``host`` and kill the tmux session. Returns True on success."""
    if not shutil.which("ssh"):
        return False
    r = subprocess.run(
        ["ssh", host, f"tmux kill-session -t {session}"],
        capture_output=True, text=True, timeout=15,
    )
    return r.returncode == 0


def vss_default_workdir() -> str:
    """The default $workdir for both VSS + audio outputs."""
    return config.DEFAULT_REMOTE_WORKDIR


def fetch_audio_outputs_from_remote(
    *, host: str, workdir: str, stem: str, dest_dir: str,
) -> tuple[bool, str]:
    """Pull a remote run's audio-out/<stem>/ back to ``dest_dir``.

    Returns ``(ok, message)``. Used after a remote audio job completes so
    the Web UI can serve the transcript files for download.
    """
    if not shutil.which("rsync"):
        return False, "rsync not found on PATH"
    src = f"{host}:{workdir.rstrip('/')}/audio-out/{stem}/"
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["rsync", "-avh", "--progress", src, dest_dir.rstrip("/") + "/"],
        capture_output=True, text=True, timeout=600,
    )
    return r.returncode == 0, r.stderr or r.stdout
