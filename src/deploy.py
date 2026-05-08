# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Deploy: rsync this tool to a remote host, then run install over SSH.

Used as `spectator deploy --target <gpu-machine>`. The remote copy of
Spectator (in $workdir/spectator-tool/) becomes self-sufficient: from
that point the user can ssh in and run `spectator …` directly on the
Spark.

`rsync_only` is exposed for iterative development — a `spectator rsync
--target …` after editing source files pushes the code without
re-running `uv sync` or the install step.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from . import config
from ._run import RunResult, rsync_to, ssh_run

console = Console()


_RSYNC_EXCLUDES = [
    ".venv", "__pycache__", "*.pyc",
    ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "*.egg-info", "dist", "build",
    "uv.lock",  # remote resolves its own
    ".DS_Store", ".git",
    "creds.txt", ".creds", "*.creds",  # never ship secrets through rsync
]


def _remote_tool_dir(cfg: config.StackConfig) -> str:
    return f"{cfg.workdir}/spectator-tool"


def rsync_only(host: str, cfg: config.StackConfig, *,
               tool_root: Path | None = None) -> RunResult:
    """Just rsync the source tree. No uv sync, no install. Idempotent.

    Use during iterative development — push edits to the remote tool
    without paying the uv-sync round-trip. The remote venv at
    $workdir/spectator-tool/.venv/ is untouched, so subsequent
    `./spectator …` calls on the remote still work as long as no new
    deps were added (otherwise run a full `spectator deploy`).
    """
    if tool_root is None:
        # __file__ → src/deploy.py; tool root is one level up from src/.
        tool_root = Path(__file__).resolve().parents[1]  # …/spectator/
    remote_dir = _remote_tool_dir(cfg)

    console.print(f"[bold]→[/bold] mkdir {cfg.workdir}, {remote_dir} on {host}")
    r = ssh_run(host, f"mkdir -p {cfg.workdir} {remote_dir}")
    if not r.ok:
        return r

    console.print(f"[bold]→[/bold] rsync {tool_root.name}/ → {host}:{remote_dir}/")
    r = rsync_to(host, tool_root, remote_dir, exclude=list(_RSYNC_EXCLUDES))
    if r.ok:
        console.print("[green]✓[/green] rsync complete (no install run; "
                      "use `spectator deploy --target …` for a full sync)")
    else:
        console.print(f"[red]rsync failed[/red]\n{r.stderr}")
    return r


def deploy(host: str, cfg: config.StackConfig, *, do_install: bool = True,
           tool_root: Path | None = None) -> RunResult:
    """Full deploy: rsync + remote uv sync + (optional) install."""

    if tool_root is None:
        tool_root = Path(__file__).resolve().parents[1]
    remote_tool_dir = _remote_tool_dir(cfg)

    r = rsync_only(host, cfg, tool_root=tool_root)
    if not r.ok:
        return r

    console.print(f"[bold]→[/bold] sync the remote tool venv")
    r = ssh_run(host, f"""
        export PATH="$HOME/.local/bin:$PATH"
        if ! command -v uv >/dev/null; then
          echo "==== Installing uv on remote ===="
          curl -LsSf https://astral.sh/uv/install.sh | sh
          export PATH="$HOME/.local/bin:$PATH"
        fi
        cd {remote_tool_dir}
        if [ ! -d .venv ]; then
          uv venv --python 3.12 .venv
        fi
        uv sync
        echo "==== uv install complete (invoke as 'uv run python -m src') ===="
        ls -la .venv/bin/python
    """)
    if not r.ok:
        console.print(f"[red]uv sync failed[/red]\n{r.stdout}\n{r.stderr}")
        return r
    console.print(r.stdout)

    if do_install:
        console.print(f"[bold]→[/bold] running `spectator install` on {host}")
        r = ssh_run(host, f"""
            export PATH="$HOME/.local/bin:$PATH"
            cd {remote_tool_dir}
            uv run python -m src install \\
                --workdir {cfg.workdir} \\
                {'--ngc-key "$NGC_CLI_API_KEY"' if cfg.ngc_api_key else ''}
        """, env=cfg.env_block())
        console.print(r.stdout)
        if not r.ok:
            console.print(f"[red]install step failed[/red]\n{r.stderr}")
        return r

    return r
