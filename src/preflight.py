# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Preflight: verify the target host meets VSS Blueprint v3.1 prerequisites.

Per https://build.nvidia.com/spark/vss/instructions:
    - Driver >= 580.95.05 (DGX-SPARK)
    - CUDA 13.0
    - Docker + Docker Compose v2 + NVIDIA Container Toolkit
    - User in docker group
    - NGC API key reachable (env var on the driving shell, or `~/.ngc/api_key`)

The same checks run locally (`spectator preflight`) or remotely
(`spectator preflight --target HOST`) by routing through `_run.run` or
`_run.ssh_run`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from . import config
from ._run import RunResult, run, ssh_run

console = Console()


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    blocking: bool = True


def _exec(host: str | None, script: str) -> RunResult:
    if host:
        return ssh_run(host, script)
    return run(["bash", "-c", script])


def _version_ge(have: str, want: str) -> bool:
    """Compare dotted versions. Missing tail components treated as 0."""
    def parts(v: str) -> list[int]:
        out: list[int] = []
        for p in re.split(r"[.\-]", v):
            try:
                out.append(int(p))
            except ValueError:
                out.append(0)
        return out
    a, b = parts(have), parts(want)
    pad = max(len(a), len(b))
    a += [0] * (pad - len(a))
    b += [0] * (pad - len(b))
    return a >= b


def collect_checks(host: str | None = None,
                   workdir: str | None = None) -> list[Check]:
    """Run all preflight checks against `host` (None = local).

    `workdir` is used to probe ``$workdir/.creds`` for NGC / NVIDIA
    keys (v0.4.4+; persistent creds store). Defaults to
    ``config.DEFAULT_REMOTE_WORKDIR`` if None — the right answer for
    99% of users.
    """
    from . import config as _config

    if workdir is None:
        workdir = _config.DEFAULT_REMOTE_WORKDIR

    out: list[Check] = []

    # nvidia-smi + driver version
    r = _exec(host, "nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null || echo MISSING")
    if "MISSING" in r.stdout or r.rc != 0:
        out.append(Check("nvidia-smi", False, "not found — install NVIDIA drivers"))
    else:
        line = r.stdout.strip().splitlines()[0]
        ver, name = [p.strip() for p in line.split(",", 1)]
        out.append(Check(
            f"driver / GPU",
            _version_ge(ver, config.REQUIRED_DRIVER),
            f"{ver} on {name} (need ≥ {config.REQUIRED_DRIVER})",
        ))

    # CUDA via libcudart fallback (nvcc isn't always installed on Spark)
    r = _exec(host, (
        "if command -v nvcc >/dev/null 2>&1; then "
        "  nvcc --version | grep -oE 'release [0-9]+\\.[0-9]+' | head -1 | awk '{print $2}'; "
        "elif ls /usr/local/cuda/version.json 2>/dev/null; then "
        "  python3 -c \"import json; print(json.load(open('/usr/local/cuda/version.json'))['cuda']['version'])\" 2>/dev/null; "
        "elif ls -1 /usr/local/cuda/targets/*/lib/libcudart.so.* 2>/dev/null | head -1; then "
        "  ls -1 /usr/local/cuda/targets/*/lib/libcudart.so.* 2>/dev/null | head -1 | grep -oE '[0-9]+\\.[0-9]+' | head -1; "
        "else echo MISSING; fi"
    ))
    cuda = (r.stdout.strip().splitlines() or ["MISSING"])[-1]
    if cuda == "MISSING":
        out.append(Check("CUDA", False, f"not detected (need {config.REQUIRED_CUDA_MAJOR_MINOR})"))
    else:
        out.append(Check(
            "CUDA",
            _version_ge(cuda, config.REQUIRED_CUDA_MAJOR_MINOR),
            f"{cuda} (need ≥ {config.REQUIRED_CUDA_MAJOR_MINOR})",
        ))

    # docker + version
    r = _exec(host, "docker --version 2>/dev/null || echo MISSING")
    if "MISSING" in r.stdout:
        out.append(Check("docker", False, "not installed"))
    else:
        m = re.search(r"Docker version ([0-9.]+)", r.stdout)
        ver = m.group(1) if m else "unknown"
        out.append(Check(
            "docker",
            _version_ge(ver, config.REQUIRED_DOCKER) if m else False,
            f"{ver} (need ≥ {config.REQUIRED_DOCKER})",
        ))

    # docker compose v2
    r = _exec(host, "docker compose version 2>/dev/null || echo MISSING")
    if "MISSING" in r.stdout:
        out.append(Check("docker compose", False, "not installed"))
    else:
        m = re.search(r"v?([0-9.]+)", r.stdout)
        ver = m.group(1) if m else "unknown"
        out.append(Check(
            "docker compose",
            _version_ge(ver, config.REQUIRED_COMPOSE.lstrip("v")) if m else False,
            f"v{ver} (need ≥ {config.REQUIRED_COMPOSE})",
        ))

    # docker group membership (running user)
    r = _exec(host, "id -nG 2>/dev/null")
    in_grp = "docker" in r.stdout.split() if r.ok else False
    out.append(Check(
        "docker group",
        in_grp,
        "user is in docker group" if in_grp else "user is NOT in docker group — `sudo usermod -aG docker $USER`",
        blocking=False,  # spectator install fixes this
    ))

    # NVIDIA Container Toolkit (nvidia-ctk) — TWO checks: installed AND
    # registered with the docker daemon. Both are required; an installed
    # nvidia-ctk that hasn't been `nvidia-ctk runtime configure`'d will
    # let bring-up pull all images then fail at `docker run` time with
    # "unknown or invalid runtime name: nvidia". Costs ~30 GB of pulls
    # before the failure surfaces.
    r = _exec(host, "command -v nvidia-ctk 2>/dev/null && nvidia-ctk --version 2>/dev/null | head -1 || echo MISSING")
    if "MISSING" in r.stdout:
        out.append(Check("nvidia-ctk", False, "not installed (NVIDIA Container Toolkit)"))
    else:
        out.append(Check("nvidia-ctk", True, r.stdout.strip().splitlines()[-1]))

    r = _exec(host, "docker info 2>/dev/null | grep -E '^ Runtimes:' | head -1")
    runtimes = r.stdout.strip()
    has_nvidia_runtime = "nvidia" in runtimes
    out.append(Check(
        "nvidia docker runtime",
        has_nvidia_runtime,
        runtimes if has_nvidia_runtime
        else "not registered with docker — run `spectator install --apply-system --target ...`",
        blocking=True,
    ))

    # ffmpeg (used for video → frames before upload)
    r = _exec(host, "command -v ffmpeg >/dev/null && ffmpeg -version 2>&1 | head -1 || echo MISSING")
    if "MISSING" in r.stdout:
        out.append(Check("ffmpeg", False, "not installed (recommended)", blocking=False))
    else:
        out.append(Check("ffmpeg", True, r.stdout.strip().splitlines()[-1][:60]))

    # NGC + NVIDIA API keys. Three sources (highest priority first):
    #   1. $workdir/.creds on the target (v0.4.4+; the canonical store).
    #   2. Env var on the driving shell — we forward $NGC_CLI_API_KEY /
    #      $NVIDIA_API_KEY over SSH, so a key set in the local shell
    #      reaches every bash payload Spectator ssh-execs.
    #   3. (NGC only) ~/.ngc/api_key on the target — the upstream NGC
    #      playbook convention; docker login picks this up too.
    creds_check = _exec(
        host,
        f'if [ -f "{workdir}/.creds" ]; then '
        '  set -a; '
        f'  . "{workdir}/.creds"; '
        '  set +a; '
        '  echo "NGC=${NGC_CLI_API_KEY:+SET}"; '
        '  echo "NVIDIA=${NVIDIA_API_KEY:+SET}"; '
        'else '
        '  echo "NGC="; echo "NVIDIA="; '
        'fi',
    )
    creds_lines = creds_check.stdout.splitlines() if creds_check.ok else []
    creds_has_ngc = any(line.strip() == "NGC=SET" for line in creds_lines)
    creds_has_nvidia = any(line.strip() == "NVIDIA=SET" for line in creds_lines)

    if creds_has_ngc:
        out.append(Check(
            "NGC API key",
            True,
            f"in {workdir}/.creds on target (v0.4.4+; sourced by every bash payload)",
            blocking=False,
        ))
    elif os.environ.get("NGC_CLI_API_KEY"):
        out.append(Check(
            "NGC API key",
            True,
            "$NGC_CLI_API_KEY set in driving shell (forwarded over SSH; "
            "first install will persist to .creds)",
            blocking=False,
        ))
    else:
        r = _exec(host, "test -f ~/.ngc/api_key && echo present || echo absent")
        has_key = "present" in r.stdout
        out.append(Check(
            "NGC API key",
            has_key,
            "~/.ngc/api_key present on target" if has_key
            else "no key found anywhere — set $NGC_CLI_API_KEY locally (forwarded "
                 "via SSH; first install persists it to $workdir/.creds), or place "
                 "the key at ~/.ngc/api_key on target. "
                 "https://org.ngc.nvidia.com/setup/api-keys",
            blocking=False,
        ))

    if creds_has_nvidia:
        out.append(Check(
            "NVIDIA API key",
            True,
            f"in {workdir}/.creds on target (v0.4.4+; used by remote LLM auth)",
            blocking=False,
        ))
    elif os.environ.get("NVIDIA_API_KEY"):
        out.append(Check(
            "NVIDIA API key",
            True,
            "$NVIDIA_API_KEY set in driving shell (used by remote LLM auth; "
            "first install will persist to .creds)",
            blocking=False,
        ))
    else:
        out.append(Check(
            "NVIDIA API key",
            False,
            "$NVIDIA_API_KEY unset and not in $workdir/.creds — "
            "required for `spectator up` (remote LLM auth)",
            blocking=False,
        ))

    # Disk free
    r = _exec(host, "df -BG --output=avail $HOME 2>/dev/null | tail -1 | tr -d ' G' || echo 0")
    try:
        free_gb = int(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        free_gb = 0
    out.append(Check(
        "disk free (home)",
        free_gb >= 50,
        f"{free_gb}G (need ≥ 50G for images + sample data)",
        blocking=False,
    ))

    return out


def render(checks: list[Check], host: str | None) -> bool:
    """Pretty-print a checks table; return True if no blocking failures."""
    title = f"VSS preflight on {host}" if host else "VSS preflight (local)"
    table = Table(title=title, show_lines=False)
    table.add_column("Check")
    table.add_column("OK")
    table.add_column("Detail")
    blocking_failed = False
    for c in checks:
        mark = "[green]✓[/green]" if c.ok else (
            "[red]✗[/red]" if c.blocking else "[yellow]![/yellow]"
        )
        table.add_row(c.name, mark, c.detail)
        if not c.ok and c.blocking:
            blocking_failed = True
    console.print(table)
    if blocking_failed:
        console.print("[red]Blocking checks failed.[/red] Fix and rerun.")
    return not blocking_failed
