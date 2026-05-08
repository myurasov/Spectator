# AGENTS.md — Spectator

This file is the **universal entry-point for AI-enabled IDEs** (Cursor,
Claude Code, OpenAI Codex, GitHub Copilot, and any other tool that
respects the [`AGENTS.md`](https://agents.md/) convention). Read it on
every turn; it is intentionally short and points at the canonical
sources.

## Table of Contents

- [What Spectator is](#what-spectator-is)
- [Read in order, on every turn](#read-in-order-on-every-turn)
- [Bootstrap and run](#bootstrap-and-run)
- [IDE-specific notes](#ide-specific-notes)

## What Spectator is

A thin (~1.5 kLoC) Python CLI wrapper around two stacks:

1. NVIDIA's [Video Search & Summarization (VSS) Blueprint v3.1](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)
   — `process`, `query`, `up` / `down` / `status` / `logs` / `ui`.
   Designed for [DGX Spark (GB10)](https://build.nvidia.com/spark/vss);
   works on any v3.1-supported host (`H100`, `L40S`, `RTXPRO6000BW`,
   Jetson `THOR`).
2. A sibling pure-audio (Whisper) pipeline — `audio install`,
   `audio transcribe`, `audio status`, `audio fetch`, `audio presets`.

Apache 2.0 licensed. User-facing surface (pipes mean "any one of"):

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

## Read in order, on every turn

1. **[`ai/dev.agent.md`](ai/dev.agent.md)** — the actual rules: who
   you are, how the maintainer wants the project built, the commit and
   test discipline, when to ask vs. just-do-it. **This is your
   primary instruction file.**
2. **[`ai/dev.memory.md`](ai/dev.memory.md)** — accumulated maintainer
   preferences (workflow shortcuts, gotchas, conventions). Treat each
   entry as a hard rule unless overridden in the current turn.
3. **[`ai/spec.txt`](ai/spec.txt)** — canonical specification of what
   Spectator does (architecture, on-disk layout, CLI surface, edge
   cases). Consult before adding, changing, or removing behavior.

`ai/dev.agent.md` itself opens with the "read in this exact order"
list, so the chain is self-reinforcing — start there.

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

For documentation aimed at humans (not agents):

- **[`README.md`](README.md)** — beginner-friendly setup walk-through
  (account-manager-grade).
- **[`REFERENCE.md`](REFERENCE.md)** — full reference: SSH config,
  hardware profiles, all subcommands, gotchas.
- **[`CONTRIBUTING.md`](CONTRIBUTING.md)** — PR workflow + DCO
  sign-off requirement.
- **[`SECURITY.md`](SECURITY.md)** — private-disclosure process.
- **[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)** — runtime /
  dev / external-service dep licenses.

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
forwarder to `ai/dev.agent.md` rather than duplicating content.
