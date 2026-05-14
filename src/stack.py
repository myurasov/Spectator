# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle (up/down/status/logs) wrappers around the VSS dev-profile.sh.

Per the v3.1 Spark playbook the canonical bring-up is:

    cd $workdir/video-search-and-summarization
    export NGC_CLI_API_KEY=...
    export LLM_ENDPOINT_URL=...
    export NVIDIA_API_KEY=...
    scripts/dev-profile.sh up -p base -H DGX-SPARK \
        --use-remote-llm --llm nvidia/nvidia-nemotron-nano-9b-v2

`up` runs in tmux (the bring-up pulls multi-GB images). The other
verbs are short and run in the foreground.
"""

from __future__ import annotations

import textwrap

from . import _creds, config
from ._run import RunResult, run, ssh_run


def _bash_workdir(workdir: str) -> str:
    """Same convention as audio._expand_tilde / install._bash_workdir —
    bash-friendly form so the rendered ``[ -f "$path" ]`` test
    actually expands the leading tilde."""
    if workdir.startswith("~/"):
        return "$HOME/" + workdir[2:]
    if workdir == "~":
        return "$HOME"
    return workdir


def _vss_dir(cfg: config.StackConfig) -> str:
    return f"{cfg.workdir}/{cfg.vss_checkout}"


def _logs_dir(cfg: config.StackConfig) -> str:
    return f"{cfg.workdir}/logs"


def _exec(host: str | None, script: str,
          env: dict[str, str] | None = None) -> RunResult:
    if host:
        return ssh_run(host, script, env=env or {})
    return run(["bash", "-c", script], env=env)


def up(cfg: config.StackConfig, host: str | None = None,
       tmux_session: str = "spectator-up") -> RunResult:
    """Bring the stack up inside a tmux session so it survives ssh disconnect."""
    flags = (
        f"-p {cfg.deploy_profile} "
        f"-H {cfg.hardware_profile} "
        f"--use-remote-llm "
        f"--llm {cfg.remote_llm}"
    )
    workdir_bash = _bash_workdir(cfg.workdir)
    script = textwrap.dedent(f'''
        # Source $workdir/.creds if present (v0.4.4 — creds are
        # authoritative once the file exists; SSH-propagated env vars
        # become a no-op here when .creds wins).
        {_creds.source_block(workdir_bash)}
        cd {_vss_dir(cfg)}
        mkdir -p {_logs_dir(cfg)}
        > {_logs_dir(cfg)}/up.log
        if tmux has-session -t {tmux_session} 2>/dev/null; then
          tmux kill-session -t {tmux_session}
        fi
        tmux new-session -d -s {tmux_session} \\
          "cd {_vss_dir(cfg)} && \\
           export NGC_CLI_API_KEY='${{NGC_CLI_API_KEY:-}}' && \\
           export LLM_ENDPOINT_URL='${{LLM_ENDPOINT_URL:-{config.DEFAULT_LLM_ENDPOINT}}}' && \\
           export NVIDIA_API_KEY='${{NVIDIA_API_KEY:-}}' && \\
           scripts/dev-profile.sh up {flags} 2>&1 | tee {_logs_dir(cfg)}/up.log"
        sleep 2
        echo "==== tmux ===="
        tmux ls
        echo
        echo "==== first lines of up.log ===="
        head -20 {_logs_dir(cfg)}/up.log 2>/dev/null
        echo
        echo "Tail with:  spectator logs --follow"
    ''').strip()
    return _exec(host, script, env=cfg.env_block())


def down(cfg: config.StackConfig, host: str | None = None) -> RunResult:
    """Stop everything Spectator launches on the target.

    Covers (in order):
      1. VSS docker compose stack (`scripts/dev-profile.sh down`).
      2. The `spectator-up` tmux session (the bring-up watcher).
      3. Any `audio-*` and `diar-*` tmux sessions (one per
         `audio transcribe` and `audio diarize` job respectively).
         These can be holding the GPU long after VSS is down — silently
         competing with whatever the user brings back up next.
      4. The user-local cache cleaner is *surfaced but not stopped* —
         it runs under sudo, so auto-stopping would require an
         interactive password prompt during `down`. Run
         `spectator system cache-cleaner-stop` separately to retire it.
      5. Final post-down container snapshot, so the user can confirm
         nothing's lingering.
    """
    cleaner_path = f"{cfg.workdir}/{config.CACHE_CLEANER_RELPATH}"
    script = textwrap.dedent(f'''
        cd {_vss_dir(cfg)}
        scripts/dev-profile.sh down || true
        tmux kill-session -t spectator-up 2>/dev/null || true

        # Kill any audio transcribe / diarize tmux sessions
        # (named `audio-<stem>` and `diar-<stem>` respectively).
        # Listing first + iterating lets us print one line per killed
        # session — `tmux list-sessions | xargs kill-session` would be
        # silent.
        audio_sessions="$(tmux list-sessions -F '#{{session_name}}' \\
          2>/dev/null | grep -E '^(audio|diar)-' || true)"
        if [ -n "$audio_sessions" ]; then
          echo "==== killing audio transcribe / diarize tmux sessions ===="
          echo "$audio_sessions" | while read -r s; do
            tmux kill-session -t "$s" 2>/dev/null && echo "  ✓ $s killed"
          done
        fi

        # Cache cleaner status: surface but don't auto-stop. It needs
        # sudo, so killing it would require an interactive password
        # prompt — out of scope for `down`. Caller can retire it with
        # the dedicated verb.
        if pgrep -f sys-cache-cleaner.sh >/dev/null 2>&1; then
          echo "==== cache cleaner is still running (left running) ===="
          echo "    script: {cleaner_path}"
          echo "    stop with: spectator system cache-cleaner-stop"
        fi

        echo "==== docker containers (post-down) ===="
        docker ps --format 'table {{{{.Names}}}}\t{{{{.Status}}}}' | head -20
    ''').strip()
    return _exec(host, script)


def status(cfg: config.StackConfig, host: str | None = None) -> RunResult:
    script = textwrap.dedent(f'''
        echo "==== tmux ===="
        tmux ls 2>&1 || true
        echo
        echo "==== docker compose ps (in {_vss_dir(cfg)}) ===="
        cd {_vss_dir(cfg)} 2>/dev/null && docker compose ps 2>/dev/null || \\
          docker ps --format 'table {{{{.Names}}}}\t{{{{.Status}}}}\t{{{{.Ports}}}}' | head -30
        echo
        echo "==== UI port {config.UI_PORT} ===="
        (curl -sI -m 3 http://localhost:{config.UI_PORT} | head -1) || echo "(unreachable on localhost)"
        echo
        echo "==== nvidia-smi (1 line) ===="
        nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader || true
    ''').strip()
    return _exec(host, script)


def logs(cfg: config.StackConfig, host: str | None = None,
         service: str | None = None, follow: bool = False,
         lines: int = 200) -> RunResult:
    if service:
        body = f"docker logs {'--follow ' if follow else ''}--tail {lines} {service}"
    else:
        body = f"tail {'-f ' if follow else ''}-n {lines} {_logs_dir(cfg)}/up.log 2>/dev/null"
    script = textwrap.dedent(f'''
        cd {_vss_dir(cfg)}
        {body}
    ''').strip()
    return _exec(host, script)
