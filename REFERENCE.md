# Spectator — Reference

Detailed reference for Spectator. For the friendly setup walk-through (designed so a non-engineer can get from zero to a working transcript), see [README.md](README.md).

## Table of Contents

- [Containment policy (no surprises)](#containment-policy-no-surprises)
- [Topology (v3.1 on Spark)](#topology-v31-on-spark)
- [Install paths](#install-paths)
  - [Why a wrapper at all?](#why-a-wrapper-at-all)
- [SSH access (for `--target` flows)](#ssh-access-for---target-flows)
- [Full quickstart](#full-quickstart)
  - [1. Load credentials (per shell)](#1-load-credentials-per-shell)
  - [2. Deploy + install on a Spark you can SSH into](#2-deploy--install-on-a-spark-you-can-ssh-into)
  - [3. Bring the VSS stack up (video pipeline)](#3-bring-the-vss-stack-up-video-pipeline)
  - [4. Process a video via the CLI](#4-process-a-video-via-the-cli)
  - [5. Audio-only transcription (Whisper)](#5-audio-only-transcription-whisper)
  - [6. Iterative development](#6-iterative-development)
- [Subcommand reference](#subcommand-reference)
- [Required env vars / keys](#required-env-vars--keys)
- [Hardware profiles](#hardware-profiles)
- [Audio device selection](#audio-device-selection)
- [Audio language handling](#audio-language-handling)
- [Notes & caveats](#notes--caveats)
- [Layout](#layout)

## Containment policy (no surprises)

Everything Spectator touches falls into one of these buckets:

| Bucket | Where it lives | When |
|---|---|---|
| **Spectator's own deps** | `.venv/` in the project directory | first `./spectator …` call |
| **VSS stack** | docker images / containers + `$workdir/video-search-and-summarization/` | brought up by `spectator up` |
| **Whisper venv** | `$workdir/audio-venv/` (torch + openai-whisper) | one-time `spectator audio install` |
| **Per-user state** | `$workdir/` (cloned VSS repo, user-local cache cleaner, audio-in/out/logs) and `~/.docker/config.json` (NGC login) | `spectator install` default |
| **System mutations** | `nvidia-ctk runtime configure`, `systemctl restart docker`, `usermod -aG docker` | **opt-in only** via `spectator install --apply-system` |

The default `spectator install` never writes outside `$workdir` and `~/.docker/config.json`. All system-level changes are gated behind `--apply-system` (with sudo prompts you'll see). `$workdir` defaults to `~/spectator/` on the target.

## Topology (v3.1 on Spark)

```
┌─ your laptop ───────┐          ┌─ DGX Spark (GB10) ──────────────────┐
│  Spectator CLI      │── ssh ──▶│  spectator-tool/   (rsynced)         │
│  (deploy / drive)   │          │  audio-venv/       (whisper + torch) │
└─────────────────────┘          │  video-search-and-summarization/     │
                                 │  ──────────────────────────────────  │
                                 │  Cosmos-Reason2-8B    (local NIM)    │
                                 │  Alert Bridge / VST / RT-VLM         │
                                 │  Agent UI :3030  /  API :8000        │
                                 └──────────────┬───────────────────────┘
                                                │
                                                ▼
                                build.nvidia.com (remote LLM endpoint)
                                nvidia/nvidia-nemotron-nano-9b-v2
```

VLM and Whisper run locally on the Spark's Blackwell GPU; the LLM runs on remote NIM endpoints. Server-class GPU profiles (`H100`, `L40S`, `RTXPRO6000BW`) can host the LLM locally — see [Hardware profiles](#hardware-profiles).

## Install paths

Spectator ships as a regular Python package. From a checkout:

```bash
git clone https://github.com/myurasov/Spectator.git spectator
cd spectator

# ensure venv + deps (idempotent)
./spectator install

# curated end-to-end overview
./spectator help
```

The `./spectator` wrapper handles the venv bootstrap and forwards anything outside its reserved dev-workflow names (`install / test / lint / fmt / shell / clean / help`) to the Python CLI as-is. If you've already activated a venv (or installed the package globally), you can also just call `spectator …` directly — both forms are equivalent.

Reserved dev-workflow subcommands the wrapper handles itself:

```bash
# ensure venv + deps (idempotent)
./spectator install [--force]

# pytest
./spectator test [args...]

# ruff check
./spectator lint [args...]

# ruff check --fix + ruff format
./spectator fmt

# venv-activated subshell
./spectator shell

# remove .venv + caches
./spectator clean

# show this help (then prints CLI --help)
./spectator help
```

Anything else is forwarded to the Spectator Python CLI.

### Why a wrapper at all?

The wrapper exists so the source tree is self-bootstrapping (a fresh clone runs without any prior Python setup beyond [`uv`](https://docs.astral.sh/uv/)) and it keeps the dev workflow (`test`, `lint`, `fmt`) discoverable next to the user-facing commands. It also sets `PYTHONPATH=$HERE` (the project root) before invoking `python -m src`, which sidesteps editable installs entirely — cloud-synced filesystems sometimes mark setuptools' `.pth` shim as hidden, breaking Python 3.12.13+'s import machinery. By design, this project is **not pip-installable** (`pyproject.toml` declares `[tool.uv] package = false`); use `./spectator …` or `uv run python -m src …` to invoke it.

## SSH access (for `--target` flows)

Every `--target HOST` command relies on plain `ssh HOST` working without prompts. `deploy`, `up`, `audio transcribe`, and `logs --follow` each make many SSH calls in sequence — typing a password every time kills the flow, and long-running `tmux` / `tail -f` sessions over hotel or coffee-shop networks need keepalives. Set the host up once in `~/.ssh/config`:

```ssh-config
Host <gpu-machine>
    # IP or DNS of your GPU host
    HostName 10.0.0.42
    # whoever owns ~/spectator/ on the target
    User ubuntu
    # SSH key, not password
    IdentityFile ~/.ssh/id_ed25519

    # Reuse one TCP connection across rapid-fire SSH calls. `deploy` makes
    # ~6+ ssh round-trips back-to-back; multiplexing cuts re-handshake
    # overhead from seconds to milliseconds.
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m

    # Keep tmux / tail -f sessions alive over flaky networks.
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

Copy your public key to the host, then smoke-test:

```bash
# one-time, prompts for password
ssh-copy-id <gpu-machine>

ssh <gpu-machine> "uname -a && nvidia-smi --query-gpu=name --format=csv,noheader | head -1"
./spectator preflight --target <gpu-machine>
```

`spectator preflight` is read-only — it never writes to the target — so it's a safe first end-to-end check before you run `deploy`.

If you regularly work with several hosts, pick distinct aliases (`spark-1`, `spark-mel`, `spark-loaner-3`) — Spectator threads `--target` through every subcommand.

## Full quickstart

### 1. Load credentials (per shell)

```bash
# from https://org.ngc.nvidia.com/setup/api-keys
export NGC_CLI_API_KEY=...

# from https://build.nvidia.com (Get API Key)
export NVIDIA_API_KEY=...

# optional — only if self-hosting the LLM
export LLM_ENDPOINT_URL=https://your.self.hosted/v1
```

If you've put the keys in a file, source it before running Spectator. On the Spark itself, putting the exports in `~/.bashrc` (or sourcing `~/.spectator-env` from there) so direct `ssh` sessions inherit them is convenient.

### 2. Deploy + install on a Spark you can SSH into

```bash
./spectator deploy --target <gpu-machine>
```

`deploy` rsyncs the tool to `~/spectator/spectator-tool/` on the Spark, runs `uv sync` there, then runs `spectator install` over SSH (clones the VSS repo, writes a user-local cache-cleaner script, NGC docker login).

### 3. Bring the VSS stack up (video pipeline)

First run pulls multi-GB images and takes 30–45 min. Runs in tmux on the Spark, so the laptop can disconnect:

```bash
./spectator up --target <gpu-machine>
./spectator logs --target <gpu-machine> --follow
./spectator status --target <gpu-machine>

# prints ssh -L recipe → http://localhost:3030
./spectator ui --target <gpu-machine>
```

### 4. Process a video via the CLI

The video must be a path the VSS Agent can read on the target host:

```bash
ssh <gpu-machine>
cd ~/spectator/spectator-tool
./spectator process /path/to/meeting.mp4 \
  --prompt "Summarize the meeting; list action items with timestamps; flag every slide change."

./spectator query "What did Alice say about Isaac Sim at minute 23?"
```

### 5. Audio-only transcription (Whisper)

Sibling pipeline that runs **without** VSS — just Whisper on a target with a GPU.

```bash
# one-time, builds the audio-venv
./spectator audio install --target <gpu-machine>

# show the quality presets
./spectator audio presets

./spectator audio transcribe call.mp3 --target <gpu-machine> --quality phone
./spectator audio transcribe meeting.mp3 --target <gpu-machine> --quality meeting

# running jobs + completed transcripts
./spectator audio status --target <gpu-machine>

./spectator audio fetch --target <gpu-machine> -o ./transcripts/
```

Quality presets at a glance (run `./spectator audio presets` for the full table):

| Preset | Model | Decode | Use case |
|---|---|---|---|
| `studio` | large-v3-turbo | greedy (beam=1) | clean studio mic, podcast feed |
| `meeting` (default) | large-v3-turbo | beam=5 + temp fallback | typical video-conferencing recordings |
| `phone` | large-v3 | beam=5, VAD-aware | voice-coded ≤ 32 kbps audio |
| `extreme` | large-v3 | beam=10 + patience=2 | distant mic, heavy noise/crosstalk |

Override with `--model large-v3-turbo` (or any whisper model name). Slice with `--clip "0,2700"` (first 45 min). Bias technical vocab with `--initial-prompt "..."`.

### 6. Iterative development

When you're editing Spectator's own source and just want to push code changes (no `uv sync`, no install):

```bash
# ~5s
./spectator rsync --target <gpu-machine>
```

Use `./spectator deploy …` when you've changed `pyproject.toml` (deps) or want a full re-install.

## Subcommand reference

| Command | What it does |
|---|---|
| `spectator help [command]` | Curated end-to-end overview, or per-subcommand help |
| `spectator preflight [--target HOST]` | Driver / CUDA / docker / nvidia-ctk / NGC checks |
| `spectator install [--target HOST] [--ngc-key …]` | **User-space only**: clone VSS repo, write user-local cache-cleaner, NGC login |
| `spectator install --apply-system` | …plus: `sudo nvidia-ctk runtime configure`, `sudo systemctl restart docker`, `sudo usermod -aG docker` |
| `spectator rsync --target HOST` | Just rsync the source tree (no `uv sync`, no install) — for iterative dev |
| `spectator deploy --target HOST` | Full sync: rsync + remote `uv sync` + remote `install` |
| `spectator up [--target HOST]` | Bring up the stack via `dev-profile.sh up -p base -H DGX-SPARK --use-remote-llm --llm nvidia/nvidia-nemotron-nano-9b-v2` (in tmux) |
| `spectator down [--target HOST]` | Stop the stack |
| `spectator status [--target HOST]` | tmux / docker compose ps / UI port / GPU |
| `spectator logs [--target HOST] [--follow] [--service NAME]` | Tail the bring-up log or a specific docker service |
| `spectator ui [--target HOST]` | Print UI URL + SSH port-forward recipe |
| `spectator process VIDEO [--prompt …] [--target HOST]` | Upload a video and get a summary |
| `spectator query "question…" [--target HOST]` | Q&A against indexed videos (OpenAI-compat chat) |
| `spectator audio install [--target HOST]` | Bootstrap whisper + torch venv at `$workdir/audio-venv/` |
| `spectator audio presets` | Show the 4 quality presets |
| `spectator audio transcribe AUDIO [options]` | Upload audio, run whisper (in tmux), write to `$workdir/audio-out/` |
| `spectator audio status [--target HOST]` | List running whisper jobs + completed transcripts |
| `spectator audio fetch [--target HOST] -o DIR` | rsync `$workdir/audio-out/` back to a local dir |
| `spectator system cache-cleaner-start [--target HOST]` | sudo-launch the user-local cache cleaner in the background |
| `spectator system cache-cleaner-stop [--target HOST]` | sudo pkill the cache cleaner |

All commands accept `--target HOST` (an SSH alias). Without `--target`, the command runs on the local machine.

## Required env vars / keys

| Variable | When | Where to get it |
|---|---|---|
| `NGC_CLI_API_KEY` | always (pulls images from nvcr.io) | https://org.ngc.nvidia.com/setup/api-keys |
| `NVIDIA_API_KEY` | when using remote LLM endpoints (default) | https://build.nvidia.com (Get API Key) |
| `LLM_ENDPOINT_URL` | overrides the default `https://integrate.api.nvidia.com/v1` | only if self-hosting LLM |

Spectator never reads outside its own folder; it only sees the env vars.

## Hardware profiles

`spectator up --hardware DGX-SPARK` is the default. Other v3.1 profiles:

- `H100`, `L40S`, `RTXPRO6000BW` — workstation/server GPUs (full local LLM possible)
- `IGX-THOR`, `AGX-THOR` — Jetson edge

Pick by passing `--hardware <profile>` (and adjust `--llm` accordingly — server-class GPUs can run the LLM locally).

## Audio device selection

Spectator auto-detects the best torch device available in the audio-venv before each `audio transcribe` run, in this order: **`cuda`** (NVIDIA GPU) → **`mps`** (Apple Silicon) → **`cpu`** (fallback). The detection is a one-shot probe via the audio-venv's Python — no caching, so a hot-swap of the audio-venv between runs is picked up correctly.

Override with `--device {cuda|mps|cpu}`:

```bash
# default — auto-detect
./spectator audio transcribe call.mp3

# force CPU even if a GPU is available (useful for debugging or low-priority background runs)
./spectator audio transcribe call.mp3 --device cpu

# force MPS on a Mac if you want to bypass auto-detection
./spectator audio transcribe call.mp3 --device mps

# force CUDA on a remote host (errors loudly if cuda isn't available, instead of falling back silently)
./spectator audio transcribe call.mp3 --target <gpu-machine> --device cuda
```

`--fp16` is wired to the device automatically: `True` on CUDA only; `False` on MPS and CPU. MPS has long-standing fp16 quality regressions in `openai-whisper` (boundary segments come out garbled); CPU doesn't support fp16 at all. Don't try to force `--fp16` outside the CUDA path.

Performance expectations (orientation only, model = `large-v3-turbo`):

| Hardware | Real-time factor |
|---|---|
| NVIDIA GPU via CUDA (Spark / H100 / L40S) | 10-30× faster than real-time |
| Apple Silicon via MPS | 2-4× faster than real-time |
| Apple Silicon via CPU | real-time to 2× slower |
| Intel Mac via CPU | 5-15× slower than real-time |

## Audio language handling

`--language` is unset by default → Whisper auto-detects per 30-second window (the right behavior for bilingual / code-switched recordings, e.g. en ↔ hi standup or ru ↔ en customer call). Lock to a single language with an ISO-639-1 code: `--language en|es|ru|hi|ja|zh|fr|de|pt…`. For non-English audio prefer `--model large-v3` over `large-v3-turbo` (turbo is mildly degraded on lower-resource languages).

**Translation mode**: `--task translate` renders any source language as English in the transcript (timestamps preserved).

```bash
# bilingual standup — auto-detect language switches
./spectator audio transcribe standup.mp3 --target <gpu-machine> --quality meeting

# Spanish-only customer call → Spanish transcript
./spectator audio transcribe call.mp3 --target <gpu-machine> \
  --quality phone --language es --model large-v3

# Russian interview → English transcript
./spectator audio transcribe interview.mp3 --target <gpu-machine> \
  --language ru --task translate --model large-v3
```

## Notes & caveats

- **Cloud-synced source dirs**: some cloud-storage daemons occasionally mark setuptools' editable-install `.pth` shim as hidden, which Python 3.12.13+ then refuses to load. Spectator dodges this entirely by declaring `[tool.uv] package = false` and invoking via `python -m src` (with `PYTHONPATH=<project-root>`) — no editable install, no `.pth` shim, no hidden-file gotcha. If you'd rather host the venv outside the cloud-synced tree anyway (faster fs ops, smaller cloud-sync upload), set `UV_PROJECT_ENVIRONMENT=$HOME/.cache/spectator/venv` before running the wrapper.
- **Cache cleaner location**: installed at `$workdir/bin/sys-cache-cleaner.sh` (user-space), not `/usr/local/bin/`. The script needs root to write `/proc/sys/vm/{nr_hugepages,drop_caches}`, so launch with `spectator system cache-cleaner-start` (sudo prompt) — but the script file itself doesn't pollute system bin paths.
- **First-run image pulls**: ~30 GB total — make sure you have ≥ 50 GB free on `$HOME` of the target.
- **Bring-up time**: 30–45 min on first `spectator up`. Subsequent `up` calls reuse the local image cache.
- **Ports**: agent UI on 3030 (the bring-up patches the upstream compose, which hardcodes 3000, so port 3000 stays free for other dev tooling), agent API on 8000. Forward both via `ssh -L` (or use `spectator ui` for the recipe).
- **Spark LLM constraint**: the GB10's 130 GB unified memory comfortably runs the VLM (Cosmos-Reason2-8B) but the LLM is configured to run remotely on `build.nvidia.com` for v3.1. Server-class GPU profiles (`H100`, `L40S`, `RTXPRO6000BW`) can run the LLM locally — pick the profile + `--llm` accordingly.

## Layout

```
spectator/
├── pyproject.toml            # name = "spectator", [tool.uv] package = false
├── README.md                 # beginner-friendly setup walk-through
├── REFERENCE.md              # this file — full reference
├── AGENTS.md                 # entry point for AI / IDE assistants
├── LICENSE                   # Apache-2.0
├── spectator                 # thin shell wrapper (install/test/lint/fmt + CLI forwarder)
├── src/                      # invoked as `python -m src` with PYTHONPATH=<root>
│   ├── __init__.py
│   ├── __main__.py           # `python -m src` entrypoint
│   ├── cli.py                # typer entrypoint (help, install, deploy, rsync, up,
│   │                         # down, status, logs, ui, process, query, audio, system)
│   ├── config.py             # constants + StackConfig dataclass
│   ├── _run.py               # local + ssh subprocess primitives
│   ├── preflight.py          # driver / CUDA / docker / NGC checks
│   ├── install.py            # idempotent install bash script
│   ├── deploy.py             # rsync_only + full deploy (rsync + uv sync + install)
│   ├── stack.py              # up/down/status/logs (wraps dev-profile.sh)
│   ├── api.py                # upload / summarize / query (REST + OpenAI-compat)
│   └── audio.py              # whisper install + transcribe + status + fetch
└── tests/
    └── test_smoke.py         # import + --help round-trip
```

The whole tool is < 1.5k lines of Python; the heavy lifting is delegated to upstream VSS and Whisper.
