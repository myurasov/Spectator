# AGENTS.md — Spectator

This file is the **universal entry-point for AI-enabled IDEs** (Cursor,
Claude Code, OpenAI Codex, GitHub Copilot, and any other tool that
respects the [`AGENTS.md`](https://agents.md/) convention). Read it on
every turn; it is intentionally short.

## Table of Contents

- [What Spectator is](#what-spectator-is)
- [Bootstrap and run](#bootstrap-and-run)
- [Project layout](#project-layout)
- [Conventions](#conventions)
- [IDE-specific notes](#ide-specific-notes)

## What Spectator is

A thin (~1.5 kLoC) Python CLI wrapper around two stacks:

1. NVIDIA's [Video Search & Summarization (VSS) Blueprint v3.1](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)
   — `process`, `query`, `up`/`down`/`status`/`logs`/`ui`. Designed for
   [DGX Spark (GB10)](https://build.nvidia.com/spark/vss); works on any
   v3.1-supported host (`H100`, `L40S`, `RTXPRO6000BW`, Jetson `THOR`).
2. A sibling pure-audio (Whisper) pipeline — `audio install`, `audio
   transcribe`, `audio status`, `audio fetch`, `audio presets`.

Apache 2.0 licensed.

User-facing surface (pipes mean "any one of"):

```bash
# bring-up
./spectator preflight | install | rsync | deploy

# lifecycle
./spectator up | down | status | logs | ui

# video pipeline
./spectator process VIDEO | query "..."

# audio pipeline (Whisper)
./spectator audio install | transcribe | status | fetch | presets

# system helpers (sudo, opt-in)
./spectator system cache-cleaner-start | cache-cleaner-stop

# dev workflow (handled by the wrapper itself, not forwarded)
./spectator install | test | lint | fmt | shell | clean | help
```

## Bootstrap and run

Use the `./spectator` script for everything. Never bootstrap the venv
manually:

```bash
# ensure venv + deps (idempotent)
./spectator install

# pytest
./spectator test

# ruff check
./spectator lint

# ruff check --fix (+ ruff format)
./spectator fmt

# anything outside the reserved names → forwarded to the Spectator Python CLI
./spectator <anything>
```

Reserved dev-workflow names: `install / test / lint / fmt / shell /
clean / help`. Anything else is forwarded to the Spectator Python CLI
as-is. The wrapper sets `PYTHONPATH=$HERE` (the project root) and
invokes `python -m src`, sidestepping editable installs entirely —
cloud-synced filesystems (iCloud / OneDrive) sometimes mark
setuptools' `.pth` shim as hidden, and we want zero exposure to that.

## Project layout

```
spectator/
├── pyproject.toml            # [tool.uv] package = false; not pip-installable by design
├── README.md                 # beginner-friendly setup walk-through (account-manager-grade)
├── REFERENCE.md              # full reference: SSH config, hardware profiles, all subcommands, gotchas
├── LICENSE                   # Apache-2.0
├── CONTRIBUTING.md           # PR workflow + DCO sign-off requirement
├── SECURITY.md               # private-disclosure process for security issues
├── THIRD_PARTY_NOTICES.md    # runtime / dev / external-service dep licenses
├── AGENTS.md                 # ← this file
├── spectator                 # thin shell wrapper (install/test/lint/fmt + CLI forwarder)
├── src/                      # invoked as `python -m src` with PYTHONPATH=<project-root>
│   ├── __init__.py           # __version__
│   ├── __main__.py           # `python -m src` entrypoint
│   ├── cli.py                # typer entrypoint + main() function
│   ├── config.py             # StackConfig + constants
│   ├── _run.py               # local + ssh subprocess primitives
│   ├── preflight.py          # driver / CUDA / docker / nvidia-ctk / NGC
│   ├── install.py            # idempotent install bash payload
│   ├── deploy.py             # rsync_only + full deploy
│   ├── stack.py              # up/down/status/logs (wraps dev-profile.sh)
│   ├── api.py                # upload / summarize / query (REST + OpenAI-compat)
│   └── audio.py              # whisper install + transcribe + status + fetch
└── tests/
    ├── __init__.py
    └── test_smoke.py         # import + --help round-trip
```

## Conventions

- Internal imports use **relative** form (`from . import config`,
  `from ._run import ssh_run`). Don't sprinkle absolute
  `from spectator.X` imports inside the package — relative imports
  keep the package portable if the import name ever changes.
- Subprocess calls go through `_run.run` / `_run.ssh_run` /
  `_run.ssh_stream`. Don't shell out via raw `subprocess.run` from new
  call-sites — adding a primitive to `_run.py` is the right factoring.
- All commands accept `--target HOST` (an SSH alias). Without
  `--target`, the command runs on the local machine.
- The default install path **never** writes outside `$workdir` and
  `~/.docker/config.json`. System-level mutations (nvidia-ctk runtime
  configure, docker group, restart docker) live behind
  `--apply-system`. Don't add new system-level writes outside that
  flag without flagging the change in the PR.
- **Every new source file** (`.py`, `.sh`, the wrapper, etc.) must
  start with the SPDX header — match the format of any existing file
  in the same directory:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) <year> <Your Name>
  # SPDX-License-Identifier: Apache-2.0
  ```
- **Commits must be DCO-signed** (`git commit -s`). See
  [`CONTRIBUTING.md`](CONTRIBUTING.md). Unsigned commits are blocked
  by policy, not by tooling — be deliberate.
- **Third-party dependencies**: any addition / removal of a runtime
  or dev dep (`pyproject.toml`) must come with a matching update to
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) in the same PR.

## IDE-specific notes

- **Cursor** picks up `AGENTS.md` automatically (and any
  `.cursor/rules/*.mdc` files, none of which are present here).
- **Claude Code** picks up `CLAUDE.md` if present; absent that, it
  reads `AGENTS.md`. Spectator ships only this file — both work.
- **OpenAI Codex / Codex CLI** reads `AGENTS.md` per the published
  spec.
- **GitHub Copilot** reads `.github/copilot-instructions.md` if
  present; for Spectator the canonical instructions live here, so link
  or import this file when configuring Copilot for the repo.

If you add another IDE-specific shim later, keep it as a thin
forwarder to `AGENTS.md` rather than duplicating content.
