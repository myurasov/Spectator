# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Process / SSH primitives shared by preflight, install, deploy, stack, api."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunResult:
    rc: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.rc == 0


def run(cmd: list[str] | str, *, cwd: str | Path | None = None,
        check: bool = False, env: dict[str, str] | None = None) -> RunResult:
    """Local subprocess. Always captures both streams."""
    if isinstance(cmd, str):
        shell_cmd: list[str] | str = cmd
        shell = True
    else:
        shell_cmd = cmd
        shell = False
    proc = subprocess.run(
        shell_cmd,
        cwd=str(cwd) if cwd else None,
        shell=shell,
        capture_output=True,
        text=True,
        env=env,
    )
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, shell_cmd, output=proc.stdout, stderr=proc.stderr,
        )
    return RunResult(proc.returncode, proc.stdout, proc.stderr)


def ssh_run(host: str, script: str, *, env: dict[str, str] | None = None,
            check: bool = False) -> RunResult:
    """Run a multi-line bash script on `host` over SSH via stdin heredoc.

    Inline single-quoted SSH commands are fragile when the body contains
    nested quotes or `$( )`. Passing the script over stdin is the robust
    form we settled on while bringing up the Whisper run.
    """
    env_prelude = ""
    if env:
        for k, v in env.items():
            env_prelude += f"export {k}={shlex.quote(v)}\n"
    full = "set -e\n" + env_prelude + script
    proc = subprocess.run(
        ["ssh", host, "bash", "-s"],
        input=full,
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, ["ssh", host, "bash", "-s"],
            output=proc.stdout, stderr=proc.stderr,
        )
    return RunResult(proc.returncode, proc.stdout, proc.stderr)


def ssh_one(host: str, cmd: str) -> RunResult:
    """Run a single oneshot command on `host` via SSH (no heredoc)."""
    return run(["ssh", host, cmd])


def ssh_stream(host: str, cmd: str) -> int:
    """Run a single command on `host` over SSH and stream its output to
    the user's terminal in real time. Returns the exit code.

    Use this for `tail -f` and other long-running monitoring commands
    where the user wants live output and is happy to Ctrl-C the tail.
    """
    proc = subprocess.run(["ssh", host, cmd])
    return proc.returncode


def rsync_to(host: str, local_dir: str | Path, remote_dir: str,
             *, exclude: list[str] | None = None) -> RunResult:
    """Rsync a directory to a remote host (creates remote dir as needed)."""
    args = ["rsync", "-avh", "--delete"]
    for pattern in (exclude or []):
        args += ["--exclude", pattern]
    args += [str(local_dir).rstrip("/") + "/", f"{host}:{remote_dir}/"]
    return run(args)
