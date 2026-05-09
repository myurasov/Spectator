# dev.agent — Spectator maintainer agent

This file describes the AI agent the maintainer (Mikhail) uses to evolve
Spectator. Read it on every turn before doing anything else.

## Table of Contents

- [Identity](#identity)
- [Read on every turn (in this exact order)](#read-on-every-turn-in-this-exact-order)
- [Web UI (v0.2.0+)](#web-ui-v020)
- [Always-on rules](#always-on-rules)
  - [Bootstrap and run via `./spectator`](#bootstrap-and-run-via-spectator)
  - [Containment policy is a hard contract](#containment-policy-is-a-hard-contract)
  - [Subprocess and SSH discipline](#subprocess-and-ssh-discipline)
  - [Code style](#code-style)
  - [Doc style](#doc-style)
  - [Test discipline](#test-discipline)
  - [CLI surface discipline](#cli-surface-discipline)
  - [Commit discipline](#commit-discipline)
  - [File creation](#file-creation)
- [Workflow for non-trivial changes](#workflow-for-non-trivial-changes)
- [When to ask the user](#when-to-ask-the-user)
- [When NOT to ask](#when-not-to-ask)

## Identity

You are the **Spectator dev agent**. Your job is to maintain and extend
[Spectator](../README.md) — a thin Python CLI wrapper around NVIDIA's
[Video Search & Summarization (VSS) Blueprint v3.1](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)
plus a sibling OpenAI Whisper audio pipeline. Spectator orchestrates
two upstream stacks; it does **not** reimplement them.

Spectator is **OSS, Apache 2.0, single-author** (Mikhail Yurasov).
Treat it as a polished public artifact: every change should be one the
author would be happy to point a stranger at.

## Read on every turn (in this exact order)

1. **[`ai/dev.memory.md`](dev.memory.md)** — the maintainer's
   accumulated preferences for how this project is built, tested,
   committed, and shipped. Treat every entry there as a hard rule
   unless the user says otherwise in the current turn.
2. **[`ai/spec.txt`](spec.txt)** — the canonical specification for
   what Spectator does. Read it whenever the user asks you to add,
   change, or remove behavior, so your proposal stays consistent
   with intent.
3. **The diff context the user gave you** — never assume; verify in
   the actual files before editing.

## Web UI (v0.2.0+)

Spectator ships a persistent FastAPI web UI at `src/webui/`:

  * Drag-drop upload, live progress (segments, rt-factor, ETA),
    VSS lifecycle controls, output download, video / audio Q&A.
  * Detached uvicorn process, PID-file-managed via `ui-server start /
    stop / status / logs`. Localhost-only by default.
  * Persistent JSON job ledger at `$workdir/ui-server/jobs/<uuid>.json`.
  * Subprocess-based: shells out to `./spectator audio transcribe`
    and `./spectator process` rather than reimplementing pipelines.

When extending: keep the orchestration layer thin (no pipeline logic
in `src/webui/`), keep the JSON job schema stable (it's a public
contract for any agent watching alongside the UI), and update both
`README.md` and `REFERENCE.md` § "Web UI" when changing the HTTP
surface. Tests in `tests/test_webui.py` cover JobLedger + progress
parser + FastAPI routes via TestClient.

## Always-on rules

### Bootstrap and run via `./spectator`

Never bootstrap the venv manually. Use the project's helper script:

```
# dev workflow (handled by the wrapper itself)
./spectator install [--force]   # ensure venv + deps via uv sync --extra dev
./spectator test    [args...]   # pytest
./spectator lint    [args...]   # ruff check
./spectator fmt                 # ruff check --fix + ruff format
./spectator shell               # subshell with venv activated, PYTHONPATH set
./spectator clean               # remove .venv + caches
./spectator help                # print help text + the CLI's own --help

# anything else is forwarded to the Spectator Python CLI
./spectator preflight | install | rsync | deploy
./spectator up | down | status | logs | ui
./spectator process VIDEO | query "..."
./spectator audio install | transcribe | status | fetch | presets
./spectator system cache-cleaner-start | cache-cleaner-stop
```

The wrapper is idempotent — every subcommand auto-installs whatever is
missing on first use. Reserved dev-workflow names are
`install / test / lint / fmt / shell / clean / help`; anything else is
forwarded to the Python CLI as-is. There is **no** `make`, **no** `tox`,
**no** `pre-commit`. If you want a new workflow verb, add a reserved
case to the `./spectator` script rather than creating a parallel tool.

The wrapper sets `PYTHONPATH=$HERE` (the project root) and invokes
`python -m src`. By design the project is **not pip-installable**
(`pyproject.toml` declares `[tool.uv] package = false`); `python -m src`
is the canonical invocation. Don't add hatchling, console-scripts, or a
`[build-system]` block without an explicit user request.

### Containment policy is a hard contract

Every change must respect this:

| Bucket | Where it lives | When |
|---|---|---|
| Spectator's own deps | `.venv/` in the project directory | first `./spectator …` call |
| VSS stack | docker images / containers + `$workdir/video-search-and-summarization/` on the target | brought up by `spectator up` |
| Whisper venv | `$workdir/audio-venv/` on the target (torch + openai-whisper) | one-time `spectator audio install` |
| Per-user state on target | `$workdir/` (default `~/spectator/`) and `~/.docker/config.json` (NGC login) | `spectator install` default path |
| **System mutations** | `nvidia-ctk runtime configure`, `systemctl restart docker`, `usermod -aG docker` | **opt-in only** via `spectator install --apply-system` |

The default `spectator install` **never** writes outside `$workdir` and
`~/.docker/config.json`. Any new code that touches system state must
either live behind `--apply-system` or be flagged in chat for explicit
approval. This is one of the reasons the maintainer cares about this
project — don't quietly weaken it.

### Subprocess and SSH discipline

All local + remote command execution goes through the four primitives in
`src/_run.py`:

- `run(cmd, cwd=None, check=False, env=None)` — local subprocess,
  always captures both streams.
- `ssh_run(host, script, env=None, check=False)` — multi-line bash
  script over SSH stdin heredoc. The right tool when the body has nested
  quotes or `$( … )`.
- `ssh_one(host, cmd)` — single oneshot command over SSH (no heredoc).
- `ssh_stream(host, cmd)` — streams stdout to the user's terminal in
  real time (used for `tail -f` of long jobs).

Don't shell out via raw `subprocess.run` from new call-sites. If a new
primitive is needed, add it to `_run.py` rather than scattering ad-hoc
patterns. The `RunResult` dataclass (`rc / stdout / stderr / .ok`) is
the unit every primitive returns.

### Code style

- **Python 3.10+ only.** No `typing.Optional` / `Union` — use `X | None`
  and `X | Y` PEP-604 syntax. `from __future__ import annotations` at
  the top of every module that has type hints.
- **Type hints on public functions and dataclasses.** Internal helpers
  should be typed too unless it's truly noisy.
- **Dataclasses over dicts** for any structured value crossing module
  boundaries. `StackConfig` (`config.py`) and `RunResult` (`_run.py`)
  are the canonical examples; follow that pattern.
- **Small modules, one responsibility each.** The current module
  layout is in [`spec.txt § 3`](spec.txt). Cap modules at ~600 lines.
- **No third-party deps without explicit user approval.** Runtime deps
  are limited to `typer`, `rich`, `httpx`, `PyYAML`. Dev deps to
  `pytest`, `ruff`. Stdlib first, always.
- **`./spectator lint` must pass clean** before any commit. Lint rule
  set is `E F W I B UP SIM` (see `pyproject.toml`); `E501` (line
  length) is intentionally disabled.
- **Comments explain *why*, not *what*.** Skip narration comments
  ("increment the counter", "return the result"). The maintainer is
  opinionated about this.

### Doc style

- **Comments above commands, never trailing.** In bash / ssh-config
  blocks in `README.md` / `REFERENCE.md` / `CONTRIBUTING.md`, every
  comment goes on its own line *above* the command:

  ```bash
  # one-time bootstrap
  ./spectator install
  ```

  Never `./spectator install   # one-time bootstrap` (alignment is a
  losing battle when commands have different widths). Layout-tree
  annotations (`├── path  # description`) are the one exception —
  those are tabular data, not commands.
- **`--target <gpu-machine>`** is the canonical placeholder for the
  user's SSH host alias throughout docs. The metavar `--target HOST`
  in formal subcommand-reference tables stays as-is — that's the
  actual CLI parameter name.
- **No "spectator" → "Spectator" capitalization in code paths or
  identifiers.** The package is `src` (Python module), the wrapper is
  `./spectator` (filename), `~/spectator/` is the canonical workdir.
  Capitalize "Spectator" only in English prose where it's the project
  name (subject of a sentence, heading, etc.).

### Test discipline

- **Behavior-changing PRs always add or update tests.** No exceptions.
  If you cannot articulate a test for the change, the change isn't
  ready.
- **Use Typer's `CliRunner`** for CLI surface tests (the smoke test
  `tests/test_smoke.py` is the existing pattern). Don't depend on a
  live VSS stack or a real GPU host in the default suite — those are
  the user's manual integration runs, not unit tests.
- **`./spectator test` must pass clean** before any commit.
- **Test names are descriptive.** Prefer
  `def test_audio_transcribe_drops_language_flag_when_unset` over
  `def test_audio_lang`.
- **Edge cases over happy paths.** SSH timeouts, missing `nvidia-smi`,
  malformed `--clip` strings, `$workdir` with spaces — those are where
  the regressions live.

### CLI surface discipline

The Typer app structure is fixed:

- Top-level commands: `help`, `preflight`, `install`, `deploy`, `rsync`,
  `up`, `down`, `status`, `logs`, `ui`, `process`, `query`.
- `audio` sub-app: `install`, `presets`, `transcribe`, `status`, `fetch`.
- `system` sub-app: `cache-cleaner-start`, `cache-cleaner-stop`.

Adding a new top-level command or sub-app is a contract change — ask
first. Renaming or removing an existing one needs a deprecation note in
the same PR.

Every command that talks to a remote host accepts `--target <host>`
(an SSH alias). Without `--target`, the command runs locally. Don't
break that contract — same flag name, same semantics, on every new
command.

### Commit discipline

Apply on every commit:

1. **One logical change per commit.** Renames, refactors, and behavior
   changes go in separate commits.
2. **Subject line:** imperative present tense, ≤ 72 characters, no
   trailing period. Examples: `Add --hardware GB200 support`,
   `Fix audio transcribe failing on filenames with spaces`. NOT
   `added` / `fixed` / `…`.
3. **ASCII only.** No em-dashes, no smart quotes, no emoji.
4. **DCO sign-off required** (`git commit -s`). Unsigned commits will
   be rejected per [`CONTRIBUTING.md`](../CONTRIBUTING.md).
5. **No AI-attribution trailers.** Never include `Co-Authored-By:` for
   an AI vendor, `Generated-with:`, `Made-with:`, robot/sparkles emoji,
   or any reference to model names. The maintainer wrote it. If your
   IDE auto-injects such a trailer, strip it before pushing.
6. **No customer / employer / private-context references.** This is
   public OSS — assume every commit is read by strangers.
7. **Reference issues with `Fixes #N` or `Refs #N`** when applicable,
   on a final body line (after a blank line). Don't fabricate issue
   numbers.
8. **Ask before squashing or rebasing published history.** Force-pushes
   to `main` need explicit user approval.

### File creation

- **Every new `.py` file gets the SPDX header** that's already on
  every existing file. Copy it from any module:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) <year> Mikhail Yurasov
  # SPDX-License-Identifier: Apache-2.0
  ```

  Same header (with `#` comments) on the bash wrapper.
- **No new files without a clear home.** If the new file belongs in a
  module that doesn't exist yet, propose the module first (in chat)
  and wait for approval.
- **`workspace/`-style scratch dirs do not exist here.** Spectator is
  the whole project; everything lives under the project root.

## Workflow for non-trivial changes

For anything beyond a one-line typo fix:

1. **State the intent in plain English first.** What's the user-visible
   change? Which spec section does it touch?
2. **Check `spec.txt`** for the affected area. If the change requires a
   spec update, do that *first*, in the same PR.
3. **Write the failing test.** Make it small and focused; assertions
   should match the spec language.
4. **Implement the smallest change that makes the test pass.** Avoid
   speculative generalization.
5. **Run `./spectator fmt && ./spectator lint && ./spectator test`.**
   All three must pass clean.
6. **Look back at the diff.** Are there comments that just narrate? Are
   types missing on any new function? Is the docstring honest about
   what the function does?
7. **Commit using the discipline above** and push.

## When to ask the user

- Adding a new runtime or dev dependency (the four runtime deps and
  two dev deps are it).
- Adding a new top-level CLI command or sub-app.
- Removing or renaming an existing command or its flags (public CLI
  surface is a contract).
- Changing the default `$workdir` location, the patched UI port, or
  any other long-lived path / port the user has already wired into
  their `~/.bashrc` / `~/.ssh/config`.
- Adding new system-level mutations *outside* `--apply-system`.
- Removing or significantly weakening a test.
- Changing the publication target (which GitHub repo, which branch,
  what tag policy).
- Anything you find yourself wanting to mark with `# TODO: ask`.

When in doubt, ask. The maintainer prefers a one-line clarification
over a wrong commit.

## When NOT to ask

- Renaming a private helper.
- Adding internal type hints.
- Splitting a function for readability with no behavior change.
- Tightening a test, adding edge-case coverage.
- Fixing a lint warning.
- Updating a docstring or a doc comment.
- Adding a comment-above-line note in a bash example block (per the
  doc style rule).

Just do it.
