# dev.memory — Spectator maintainer preferences

The dev.agent reads this file on every turn (after `dev.agent.md`,
before touching code). Each entry is a hard rule unless the maintainer
overrides it in the current conversation.

## Table of Contents

- [Workflow](#workflow)
- [Code style](#code-style)
- [Doc style](#doc-style)
- [CLI surface](#cli-surface)
- [Containment](#containment)
- [Tests](#tests)
- [Lint](#lint)
- [Commits](#commits)
- [Dependencies](#dependencies)
- [Audio device handling](#audio-device-handling)
- [Upstream contracts (don't break)](#upstream-contracts-dont-break)
- [Publication](#publication)
- [Ask-before-acting list](#ask-before-acting-list)
- [Maintainer preferences (free-form, append-only)](#maintainer-preferences-free-form-append-only)

---

This is a living document. Append a new bullet under the right section
whenever the maintainer says any of:

- *"always do X"* / *"never do X"*
- *"every time you …, do …"*
- *"I prefer X to Y"*
- *"remember that …"*
- *"in this project we …"*

Keep entries short, grouped, and dated only when the rule depends on a
specific event. Don't reorder existing entries unless asked.

---

## Workflow

- Use `./spectator install / test / lint / fmt / shell / clean` for
  the dev workflow. Never run `uv`, `pytest`, or `ruff` directly — the
  helper script handles bootstrap and the cloud-sync `.pth` workaround
  idempotently.
- Forwarded CLI calls (`./spectator deploy …`, `./spectator audio
  transcribe …`, etc.) auto-bootstrap too. There is no `./spectator
  run` — the pass-through is implicit; reserved dev-workflow names
  are `install / test / lint / fmt / shell / clean / help`.
- Prefer single-entry helper scripts (`./spectator`) that handle both
  dev workflow and CLI forwarding, over separate `./dev` +
  `./spectator` binaries. (May 2026)
- The wrapper sets `PYTHONPATH=$HERE` (the project root) and invokes
  `python -m src`. The project is intentionally **not pip-installable**
  (`[tool.uv] package = false`). Don't add hatchling, console-scripts,
  or a `[build-system]` block. (May 2026)

## Code style

- Python 3.10+ only. `from __future__ import annotations` on every
  module that has type hints. PEP-604 union syntax (`X | None`, not
  `Optional[X]`).
- Type hints on every public function and dataclass.
- Dataclasses over dicts for any structured value crossing module
  boundaries. `StackConfig` (`config.py`) and `RunResult` (`_run.py`)
  are the templates.
- Comments explain *why*, not *what*. Skip narration comments. (May 2026)
- SPDX header on every new `.py` and on the bash wrapper:

  ```
  # SPDX-FileCopyrightText: Copyright (c) <year> Mikhail Yurasov
  # SPDX-License-Identifier: Apache-2.0
  ```

  Copy/paste from any existing file.
- Modules cap at ~600 lines. If a feature would push past that,
  propose a new module (in chat) before writing it.
- Internal imports use **relative form** inside the package (`from .
  import config`, `from ._run import ssh_run`). Don't sprinkle absolute
  `from src.X` imports inside the package — that breaks portability if
  the package name changes.

## Doc style

- **Trailing comments are banned in command examples.** In bash /
  ssh-config blocks across all docs (README, REFERENCE, AGENTS,
  CONTRIBUTING), every comment goes on its own line *above* the
  command. Alignment of trailing `# …` comments is a losing battle
  when commands have variable widths. (May 2026)
- Layout-tree annotations (`├── path  # description`) are the one
  exception — those are tabular and naturally aligned.
- The placeholder for the user's SSH-host alias is `<gpu-machine>` in
  all examples (`./spectator deploy --target <gpu-machine>`). The
  formal metavar `--target HOST` in subcommand-reference tables stays
  as the parameter name. (May 2026)
- Capitalize "Spectator" in English prose only (subject of a sentence,
  heading, label). Keep `spectator` lowercase in commands, file names,
  Python identifiers, env vars, and pyproject `name = "spectator"`.
- All Markdown files have a `## Table of Contents` block. Anchors
  follow GitHub's algorithm (lowercase, strip non-`[a-z0-9 -]`,
  spaces → hyphens). Verify before pushing.

## CLI surface

- Top-level commands and sub-apps are fixed (see
  [`spec.txt § 5`](spec.txt)). Adding a new one is a contract change —
  ask first. Renaming or removing one needs a deprecation note in the
  same PR.
- Every command that talks to a remote accepts `--target HOST` (an
  SSH alias). Without `--target`, it runs locally. Same flag name,
  same semantics, on every new command. (May 2026)
- Subprocess + SSH calls go through `_run.run` / `ssh_run` /
  `ssh_one` / `ssh_stream`. Don't shell out via raw `subprocess.run`
  from new call-sites — adding a primitive to `_run.py` is the right
  factoring.

## Containment

- The default `spectator install` never writes outside `$workdir` and
  `~/.docker/config.json` on the target. System-level mutations
  (`nvidia-ctk runtime configure`, docker group, `systemctl restart
  docker`, `/usr/local/bin/` writes) live **only** behind
  `--apply-system`. New code that touches system state outside that
  flag is a bug. (May 2026)
- The user-local cache cleaner lives at
  `$workdir/bin/sys-cache-cleaner.sh` (not `/usr/local/bin/`). The
  script needs root to write `/proc/sys/vm/{nr_hugepages,drop_caches}`
  but the script *file* stays in user space.

## Tests

- Behavior-changing PRs always add or update tests. No exceptions.
- Use Typer's `CliRunner` for CLI surface tests
  (`tests/test_smoke.py` is the existing pattern). Don't require a
  live VSS stack or real GPU host in the default suite. (May 2026)
- Test names are descriptive
  (`test_audio_transcribe_drops_language_flag_when_unset`, not
  `test_audio_lang`).
- Edge cases over happy paths. SSH failures, missing `nvidia-smi`,
  spaces in filenames, malformed `--clip` strings — that's where
  the regressions live.

## Lint

- `./spectator lint` must pass clean before any commit.
- Rule set is `E F W I B UP SIM` (see `pyproject.toml`). Don't widen
  it without asking. `E501` (line length) is intentionally disabled.

## Commits

- One logical change per commit. Renames, refactors, behavior changes
  go in separate commits.
- Subject line: imperative present tense, ≤ 72 characters, no trailing
  period, ASCII only.
- **DCO sign-off required** (`git commit -s`). Per
  [`CONTRIBUTING.md`](../CONTRIBUTING.md). (May 2026)
- **Never** include AI-attribution trailers (`Co-Authored-By: Claude
  <…>`, `Made-with: Cursor`, `Generated-with: …`, robot/sparkles
  emoji, references to model names). The maintainer wrote it — strip
  any auto-injected trailer before pushing. (May 2026)
- No customer / employer / private-context references in commit
  messages. This is public OSS.
- Reference issues with `Fixes #N` or `Refs #N` on a final body line
  (after a blank line). Don't fabricate issue numbers.

## Dependencies

- Runtime deps: `typer >= 0.12`, `rich >= 13.7`, `httpx >= 0.27`,
  `PyYAML >= 6.0`, plus the Web UI trio added in v0.2.0:
  `fastapi >= 0.110`, `uvicorn[standard] >= 0.30`, `python-multipart
  >= 0.0.9`. **Always ask before adding another.**
- Dev deps: `pytest >= 7.4`, `ruff >= 0.4`. Same rule.
- Stdlib first.
- When deps change in `pyproject.toml`, update
  [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) in the same
  PR. (May 2026)

## Audio device handling

- `transcribe()` auto-detects the torch device by probing the
  audio-venv (cuda > mps > cpu). The CLI exposes `--device` to force
  a specific device. (May 2026, v0.1.1)
- `--fp16 True` is only emitted on CUDA. MPS has long-standing fp16
  quality regressions in openai-whisper (boundary segments come out
  garbled); CPU doesn't support fp16 in whisper at all. (May 2026, v0.1.1)
- Apple Silicon performance: ~2-4× faster than real-time via MPS,
  ~real-time to 2× slower via CPU. CUDA on Spark/H100/L40S is
  10-30× faster than real-time. (May 2026, v0.1.1)

## Upstream contracts (don't break)

- VSS Blueprint version is **v3.1**. The `dev-profile.sh` flags
  Spectator passes (`-p base -H DGX-SPARK --use-remote-llm --llm
  nvidia/nvidia-nemotron-nano-9b-v2`) are tied to that version. When
  v3.2+ ships, surface that as a separate PR — don't silently bump.
- The agent UI is patched from upstream's hardcoded `3000` to
  `${VSS_UI_PORT:-3030}` so port 3000 stays free for the user's other
  dev tooling. The patch lives in `install.py`'s
  `_user_install_script`; if upstream changes the compose file, the
  sed patches need updating in the same PR.
- Cosmos-Reason2-8B (VLM) runs locally; the LLM
  (`nvidia/nvidia-nemotron-nano-9b-v2`) runs on
  `https://integrate.api.nvidia.com/v1` by default. Don't change those
  defaults without an explicit user request — the choice is tuned for
  the GB10 unified-memory budget.

## Publication

- Canonical home: **https://github.com/myurasov/Spectator**.
  All references in docs, pyproject `[project.urls]`, and release
  notes point here.
- Releases are tagged `vX.Y.Z` (annotated tags, e.g. `v0.1.0`) and
  exposed via `gh release create`. Tag commit + GH release in the
  same step. (May 2026)
- The version string lives in three places that must stay in sync:
  `pyproject.toml [project].version`, `src/__init__.py`'s
  `__version__`, and `ai/spec.txt`'s `Version:` header. Bump them
  together.

## Ask-before-acting list

(Mirrors `dev.agent.md` § "When to ask the user". Keep them in sync
when the maintainer adds or removes items here.)

- Adding a new runtime or dev dependency.
- Adding a new top-level CLI command or sub-app.
- Removing or renaming an existing command or its flags.
- Changing the default `$workdir`, the UI port, or any long-lived path
  / port the user has already wired into shell config.
- Adding new system-level mutations outside `--apply-system`.
- Removing or significantly weakening a test.
- Changing the publication target (repo, branch, tag policy).

## Maintainer preferences (free-form, append-only)

<!-- Add new entries below this line. Newer entries last. Move stable
     rules into the structured sections above when they prove out. -->

- **v0.3.0 path defaults**: the default `$workdir` is
  `~/.spectator/` (dot-prefixed, hidden), and the rsynced
  project tree on the target lives at `$workdir/Spectator/` (capitalized
  — represents the project name as humans see it browsing `ls
  $workdir/`). Single source of truth: `config.DEFAULT_REMOTE_WORKDIR`
  and `config.TOOL_TREE_RELPATH`. New code that constructs paths under
  `$workdir/` MUST read from `cfg.workdir` and (when applicable)
  `config.TOOL_TREE_RELPATH` — never hardcode `~/spectator/` or
  `spectator-tool/` in bash heredocs, doc snippets, or Python literals.
  (May 2026, v0.3.0)
- **Capitalization exception for `Spectator/` on disk**: the rule "no
  Spectator capitalization in code paths or identifiers" still holds
  for the wrapper filename, Python module, env vars, and CLI verbs —
  but the on-target rsync target dir is capitalized because it's the
  project name humans see in `ls`. (May 2026, v0.3.0)
