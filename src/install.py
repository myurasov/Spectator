# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Install / setup pipeline for the VSS Blueprint v3.1 stack.

Mirrors the playbook at https://build.nvidia.com/spark/vss/instructions
but with a strict containment rule: the **default install only writes
inside `$workdir` and `~/.docker/config.json`**. Anything that mutates
the system (nvidia-ctk runtime config, docker group membership, files
under `/usr/local/bin/`, restarting `dockerd`) is gated behind the
`--apply-system` flag.

Layered surface, default → opt-in:

  default                         user-space only
    - clone VSS repo into ${workdir}/${vss_checkout}
    - NGC docker login (writes ~/.docker/config.json — user-level)
    - install user-local cache cleaner script at ${workdir}/bin/

  --apply-system                  global mutations (require sudo)
    - sudo nvidia-ctk runtime configure --runtime=docker
    - sudo systemctl restart docker
    - sudo usermod -aG docker $USER

  spectator system cache-cleaner --start    (separate verb)
    - sudo -b ${workdir}/bin/sys-cache-cleaner.sh

The split lets you ssh into a Spark you don't own (or one with strict
config rules) and bring up the stack without ever touching system state
the host owner didn't already authorize.
"""

from __future__ import annotations

import textwrap

from rich.console import Console

from . import config
from ._run import RunResult, run, ssh_run

console = Console()


# ---------------------------------------------------------------------------
# script fragments
# ---------------------------------------------------------------------------

def _user_install_script(workdir: str, vss_checkout: str,
                         ngc_api_key: str | None) -> str:
    """Default install — user-space only, no sudo, no global writes."""
    return textwrap.dedent(f'''
        # ---- 1. workdir + repo clone (idempotent) ----
        mkdir -p "{workdir}"
        cd "{workdir}"

        if [ -d "{vss_checkout}/.git" ]; then
          echo "==== VSS repo already cloned, fetching latest ===="
          cd "{vss_checkout}"
          # Revert just the files we patch so `git pull --ff-only` has a clean
          # working tree. Sed-patches re-apply below, so this is a no-op for
          # behaviour — just gives us a fresh canvas.
          git checkout -- \\
            deployments/agents/agent_ui/compose.yml \\
            deployments/proxy/nginx.conf.template \\
            deployments/proxy/compose.yml \\
            2>/dev/null || true
          git fetch --tags
          git checkout main
          git pull --ff-only
          cd ..
        elif [ -d "{vss_checkout}" ]; then
          # Dir exists but is not a git checkout. Two sub-cases:
          #   1. empty dir   — leftover from a failed clone attempt. Rmdir
          #      and clone fresh; no data loss possible.
          #   2. non-empty   — something else lives there. Refuse and ask
          #      the user to inspect / remove it before retrying. Better
          #      than a confusing `git clone` "already exists" error.
          if [ -z "$(ls -A "{vss_checkout}" 2>/dev/null)" ]; then
            echo "==== {vss_checkout}/ exists but is empty; removing and cloning fresh ===="
            rmdir "{vss_checkout}"
            git clone "{config.VSS_REPO_URL}" "{vss_checkout}"
          else
            echo "==== ERROR: {workdir}/{vss_checkout}/ exists but is not a git checkout. ===="
            echo "    contents (first few entries):"
            ls -A "{vss_checkout}" | head -10 | sed 's/^/      /'
            echo
            echo "    If it's leftover from a failed install or an interrupted migration,"
            echo "    remove it and re-run deploy:"
            echo "      ssh <gpu-machine> 'rm -rf {workdir}/{vss_checkout}'"
            echo "      ./spectator deploy --target <gpu-machine>"
            echo
            echo "    If you put real data there, move it elsewhere first."
            exit 1
          fi
        else
          echo "==== Cloning {config.VSS_REPO_URL} ===="
          git clone "{config.VSS_REPO_URL}" "{vss_checkout}"
        fi

        # ---- 1b. Free-up-port-3000 patch ----
        # Upstream agent_ui hardcodes host port 3000; we make it ${{VSS_UI_PORT:-{config.UI_PORT}}}
        # so the user's other tooling (whatsapp bridges, dev servers, etc) can keep 3000.
        # Also patches the proxy nginx upstream and envsubst list to follow.
        AGENT_UI_COMPOSE="{vss_checkout}/deployments/agents/agent_ui/compose.yml"
        PROXY_TEMPLATE="{vss_checkout}/deployments/proxy/nginx.conf.template"
        PROXY_COMPOSE="{vss_checkout}/deployments/proxy/compose.yml"

        if [ -f "$AGENT_UI_COMPOSE" ] && grep -q "^      - 3000:3000$" "$AGENT_UI_COMPOSE"; then
          echo "==== patching agent_ui host port 3000 -> \\${{VSS_UI_PORT:-{config.UI_PORT}}} ===="
          sed -i.bak 's|- 3000:3000|- ${{VSS_UI_PORT:-{config.UI_PORT}}}:3000|' "$AGENT_UI_COMPOSE"
        fi
        if [ -f "$PROXY_TEMPLATE" ] && grep -q "proxy_pass http://127.0.0.1:3000;" "$PROXY_TEMPLATE"; then
          echo "==== patching proxy upstream 127.0.0.1:3000 -> 127.0.0.1:\\$VSS_UI_PORT ===="
          sed -i.bak 's|proxy_pass http://127.0.0.1:3000;|proxy_pass http://127.0.0.1:${{VSS_UI_PORT}};|' "$PROXY_TEMPLATE"
        fi
        if [ -f "$PROXY_COMPOSE" ]; then
          if ! grep -q "VSS_UI_PORT:" "$PROXY_COMPOSE"; then
            echo "==== adding VSS_UI_PORT to proxy environment ===="
            sed -i.bak '/PROXY_PORT: \\${{PROXY_PORT:-7777}}/a\\      VSS_UI_PORT: \\${{VSS_UI_PORT:-{config.UI_PORT}}}' "$PROXY_COMPOSE"
          fi
          if ! grep -q '\\$\\$VSS_UI_PORT' "$PROXY_COMPOSE"; then
            echo "==== teaching proxy envsubst about VSS_UI_PORT ===="
            sed -i 's|envsubst .\\$\\$PROXY_PORT \\$\\$VST_SUB_FILTER_HOST \\$\\$VST_SUB_FILTER_EXTERNAL.|envsubst '"'"'\\$\\$PROXY_PORT \\$\\$VST_SUB_FILTER_HOST \\$\\$VST_SUB_FILTER_EXTERNAL \\$\\$VSS_UI_PORT'"'"'|' "$PROXY_COMPOSE"
          fi
        fi

        # ---- 2. user-local cache cleaner script (NO global write) ----
        mkdir -p "{workdir}/bin"
        cat > "{workdir}/{config.CACHE_CLEANER_RELPATH}" <<'CC_EOF'
#!/bin/bash
# sys-cache-cleaner.sh — Spark performance helper.
# Writes to /proc/sys/vm/{{nr_hugepages,drop_caches}}, so MUST run as root.
# Lives in user space so the script itself doesn't pollute /usr/local/bin/.
# Start it with:    sudo -b {workdir}/{config.CACHE_CLEANER_RELPATH}
# Stop it with:     sudo pkill -f sys-cache-cleaner.sh
set -e
echo "disable vm/nr_hugepage"
echo 0 | tee /proc/sys/vm/nr_hugepages > /dev/null
echo "Starting cache cleaner — Ctrl-C to stop"
while true; do
  sync && echo 3 | tee /proc/sys/vm/drop_caches > /dev/null
  sleep 3
done
CC_EOF
        chmod +x "{workdir}/{config.CACHE_CLEANER_RELPATH}"
        echo "==== user-local cache cleaner: {workdir}/{config.CACHE_CLEANER_RELPATH} ===="

        # ---- 3. NGC docker login (writes ~/.docker/config.json — user-level) ----
        if [ -n "${{NGC_CLI_API_KEY:-}}" ]; then
          echo "==== Logging into nvcr.io ===="
          if ! command -v docker >/dev/null 2>&1; then
            echo "    docker not installed; skipping login. Install docker, then re-run."
          elif ! docker info >/dev/null 2>&1; then
            echo "    docker daemon unreachable from current user."
            echo "    fix: add yourself to the docker group (\\`spectator install --apply-system\\` or"
            echo "         \\`sudo usermod -aG docker \\$USER && newgrp docker\\`), then re-run."
          else
            echo "$NGC_CLI_API_KEY" | docker login --username '$oauthtoken' --password-stdin nvcr.io
          fi
        else
          echo "==== Skipping nvcr.io login — NGC_CLI_API_KEY not set ===="
          echo "    pass --ngc-key or export NGC_CLI_API_KEY before 'spectator up'"
        fi

        # ---- 4. summary ----
        echo
        echo "==== User-space install complete ===="
        echo "  VSS checkout : {workdir}/{vss_checkout}"
        echo "  Cache cleaner: {workdir}/{config.CACHE_CLEANER_RELPATH}"
        echo
        echo "Next:"
        echo "  spectator preflight  --target …  # confirm system bits are OK"
        echo "  spectator up         --target …  # bring stack up (in tmux)"
        echo
        echo "If preflight flags 'docker group' or 'nvidia-ctk', run:"
        echo "  spectator install --apply-system --target …"
    ''').strip()


def _system_install_script() -> str:
    """Opt-in system mutations. All sudo, all detected-and-skipped if already done."""
    return textwrap.dedent('''
        echo "==== --apply-system: running global mutations (sudo required) ===="

        # nvidia-ctk runtime configure
        if docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
          echo "  ✓ docker NVIDIA runtime already configured"
        else
          if command -v nvidia-ctk >/dev/null 2>&1; then
            echo "  → sudo nvidia-ctk runtime configure --runtime=docker"
            sudo nvidia-ctk runtime configure --runtime=docker
            echo "  → sudo systemctl restart docker"
            sudo systemctl restart docker
          else
            echo "  ! nvidia-ctk not installed — install NVIDIA Container Toolkit first"
            echo "    (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)"
          fi
        fi

        # docker group
        if id -nG | tr ' ' '\\n' | grep -qx docker; then
          echo "  ✓ user already in docker group"
        else
          echo "  → sudo usermod -aG docker $USER"
          sudo usermod -aG docker "$USER" || true
          echo "  ! you'll need to 'newgrp docker' or re-login for the group change to take effect"
        fi

        echo "==== --apply-system done ===="
    ''').strip()


# ---------------------------------------------------------------------------
# entrypoints (used by cli.py)
# ---------------------------------------------------------------------------

def install_local(cfg: config.StackConfig, *, apply_system: bool = False) -> int:
    """Stream the install bash payload locally so the user sees prompts in real time."""
    import os
    import subprocess

    pieces = [_user_install_script(cfg.workdir, cfg.vss_checkout, cfg.ngc_api_key)]
    if apply_system:
        pieces.append(_system_install_script())
    script = "\n\n".join(pieces)
    env = os.environ.copy()
    env.update(cfg.env_block())
    proc = subprocess.Popen(["bash", "-s"], stdin=subprocess.PIPE, env=env)
    assert proc.stdin is not None
    proc.stdin.write(("set -e\n" + script).encode())
    proc.stdin.close()
    return proc.wait()


def install_remote(host: str, cfg: config.StackConfig, *,
                   apply_system: bool = False) -> RunResult:
    pieces = [_user_install_script(cfg.workdir, cfg.vss_checkout, cfg.ngc_api_key)]
    if apply_system:
        pieces.append(_system_install_script())
    script = "\n\n".join(pieces)
    return ssh_run(host, script, env=cfg.env_block())


def start_cache_cleaner(host: str | None, cfg: config.StackConfig) -> RunResult:
    """Launch the user-local cache cleaner in the background (sudo -b)."""
    target_script = f"{cfg.workdir}/{config.CACHE_CLEANER_RELPATH}"
    body = textwrap.dedent(f'''
        if pgrep -f sys-cache-cleaner.sh >/dev/null; then
          echo "✓ already running (pid: $(pgrep -f sys-cache-cleaner.sh | head -1))"
          exit 0
        fi
        if [ ! -x "{target_script}" ]; then
          echo "✗ {target_script} not found — run \\`spectator install\\` first"
          exit 1
        fi
        sudo -b {target_script} >/dev/null 2>&1
        sleep 1
        if pgrep -f sys-cache-cleaner.sh >/dev/null; then
          echo "✓ started (pid: $(pgrep -f sys-cache-cleaner.sh | head -1))"
        else
          echo "✗ failed to start"
          exit 1
        fi
    ''').strip()
    if host:
        return ssh_run(host, body)
    return run(["bash", "-c", body])


def stop_cache_cleaner(host: str | None) -> RunResult:
    body = "sudo pkill -f sys-cache-cleaner.sh && echo stopped || echo 'not running'"
    if host:
        return ssh_run(host, body)
    return run(["bash", "-c", body])
