# Spectator

Drop a meeting recording onto NVIDIA's [Video Search & Summarization (VSS) Blueprint](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) and get back a clean transcript or a structured summary. Spectator is a thin CLI wrapper that handles the install, deployment, and lifecycle so you can focus on work — not IT infrastructure.

Copyright (c) 2026 Mikhail Yurasov. Licensed under the [Apache License 2.0](LICENSE).

## TL;DR

```bash
# one-time setup
git clone https://github.com/myurasov/Spectator.git spectator && cd spectator

# install local venv + deps
./spectator install

# add your GPU host to ~/.ssh/config (see step 2 below)

# from https://org.ngc.nvidia.com/setup/api-keys
export NGC_CLI_API_KEY="nvapi-..."

# from https://build.nvidia.com (Get API Key)
export NVIDIA_API_KEY="nvapi-..."

# push the tool to the GPU host (~5 min)
./spectator deploy --target <gpu-machine>

# bring up the VSS stack (~30–45 min first time)
./spectator up --target <gpu-machine>

# everyday use:

# transcribe a meeting recording
./spectator audio transcribe meeting.mp3 --target <gpu-machine>

# summarize a video
./spectator process video.mp4 --target <gpu-machine>

# ask follow-up questions about indexed videos
./spectator query "What did Alice say about ...?" --target <gpu-machine>
```

For details on each step, read on. For deeper reference (full SSH config, hardware profiles, all subcommands, gotchas), see [REFERENCE.md](REFERENCE.md).

## Table of Contents

- [Spectator](#spectator)
  - [TL;DR](#tldr)
  - [Table of Contents](#table-of-contents)
  - [What it does](#what-it-does)
  - [What you'll need](#what-youll-need)
  - [Setup](#setup)
    - [1. Install Spectator on your laptop](#1-install-spectator-on-your-laptop)
    - [2. Set up an SSH alias for your GPU host](#2-set-up-an-ssh-alias-for-your-gpu-host)
    - [3. Set your API keys](#3-set-your-api-keys)
    - [4. Deploy to your GPU host](#4-deploy-to-your-gpu-host)
    - [5. Bring the VSS stack up](#5-bring-the-vss-stack-up)
  - [Common tasks](#common-tasks)
    - [Transcribe a meeting recording](#transcribe-a-meeting-recording)
    - [Summarize a video](#summarize-a-video)
    - [Ask follow-up questions](#ask-follow-up-questions)
    - [Open the web UI](#open-the-web-ui)
    - [Tear down when you're done](#tear-down-when-youre-done)
  - [Where to go next](#where-to-go-next)
  - [Contributing](#contributing)

## What it does

Two pipelines, one CLI:

- **Audio (Whisper)** — upload a meeting / call / interview, get back a clean transcript with timestamps. Auto-detects bilingual recordings; quality presets for clean / standard / phone / very-noisy audio.
- **Video (VSS)** — upload a recording, get back a structured summary with timestamps and action items, then ask follow-up questions in plain English.

Spectator does **not** reimplement either pipeline — it automates the install, deployment, and lifecycle steps for you. Your laptop drives the GPU host over SSH; the VLM and Whisper run on the GPU; the LLM is called remotely on `build.nvidia.com`.

## What you'll need

- A **macOS or Linux laptop** with [`uv`](https://docs.astral.sh/uv/) installed (`brew install uv` on macOS, or `curl -LsSf https://astral.sh/uv/install.sh | sh`).
- A **GPU host you can SSH into**. Default target is [DGX Spark (GB10)](https://build.nvidia.com/spark/vss); the same workflow runs on H100, L40S, RTX PRO 6000, and Jetson THOR. Your team's hardware lead can point you at one.
- Two **NVIDIA credentials** (free, takes ~2 min to set up):
  - **NGC API key** — https://org.ngc.nvidia.com/setup/api-keys (used to pull docker images)
  - **NVIDIA API key** — https://build.nvidia.com → "Get API Key" (used by the remote LLM endpoint VSS calls during summarization)
- ~50 GB free disk space on the GPU host (one-time, for the docker image cache).

## Setup

### 1. Install Spectator on your laptop

```bash
git clone https://github.com/myurasov/Spectator.git spectator
cd spectator
./spectator install

# confirms it's working — should print a curated overview
./spectator help
```

### 2. Set up an SSH alias for your GPU host

Every command takes `--target <gpu-machine>`, so pick a short alias for your host and put it in `~/.ssh/config`. Use whatever alias name fits — `spark`, `dgx-1`, `lab-box`. The minimum entry (example, host alias and ip address, key/username will be yours):

```ssh-config
Host <gpu-machine>
    HostName 10.0.0.42
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
```

Smoke-test it:

```bash
ssh <gpu-machine> "nvidia-smi --query-gpu=name --format=csv,noheader"
```

If you see your GPU's name printed back, you're good. The recommended config (connection multiplexing + keepalives — they make `deploy` ~10× faster and survive flaky networks) is in [REFERENCE.md → SSH access](REFERENCE.md#ssh-access-for---target-flows). Use that on your real working setup.

The rest of this README and [REFERENCE.md](REFERENCE.md) use `<gpu-machine>` as the placeholder for whatever alias name you picked — substitute your own when you copy commands.

### 3. Set your API keys

Add the two API keys to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
export NGC_CLI_API_KEY="nvapi-..."
export NVIDIA_API_KEY="nvapi-..."
```

Reload (`source ~/.zshrc`) and confirm: `echo $NGC_CLI_API_KEY` should print your key.

### 4. Deploy to your GPU host

```bash
# check driver / CUDA / docker / NGC reachability
./spectator preflight --target <gpu-machine>

# rsync + uv sync + install (~5 min)
./spectator deploy --target <gpu-machine>
```

If `preflight` flags a missing piece (e.g. user not in the `docker` group, or the NVIDIA Container Toolkit not registered with docker), run:

```bash
./spectator install --apply-system --target <gpu-machine>
```

This is the only command Spectator runs that touches anything outside `~/spectator/` and `~/.docker/config.json` on the host — it asks for `sudo` over SSH for each system change.

### 5. Bring the VSS stack up

First run pulls multi-GB images and takes 30–45 minutes. The stack runs in `tmux` on the host, so you can close your laptop and come back later:

```bash
./spectator up --target <gpu-machine>

# watch progress; Ctrl-C to detach (the tmux job keeps running)
./spectator logs --target <gpu-machine> --follow

# quick health check any time
./spectator status --target <gpu-machine>
```

When `status` shows the agent UI on port 3030 and the API on port 8000, you're ready to use the stack.

## Common tasks

### Transcribe a meeting recording

```bash
# one-time: install Whisper on the host (separate from VSS — runs in its own venv)
./spectator audio install --target <gpu-machine>

# transcribe — uploads, runs in tmux, ~5× real-time on a Spark
./spectator audio transcribe meeting.mp3 --target <gpu-machine> --quality meeting

# pull the transcript back to your laptop
./spectator audio fetch --target <gpu-machine> -o ./transcripts/
```

Quality presets:

| Preset | Use case |
|---|---|
| `studio` | Clean studio mic, podcast feed |
| `meeting` (default) | Teams / Zoom / Webex / Google Meet recordings |
| `phone` | Voice-coded / low-bitrate phone calls |
| `extreme` | Distant mic, lots of noise, heavy crosstalk |

For non-English recordings, bilingual / code-switched audio, or "translate to English" mode, see [REFERENCE.md → Audio language handling](REFERENCE.md#audio-language-handling).

### Summarize a video

The video has to live on the GPU host (where VSS can read it). The simplest workflow: SSH in, drop the file under `~/spectator/`, run `process`:

```bash
ssh <gpu-machine>
cd ~/spectator/spectator-tool
./spectator process /home/ubuntu/spectator/meeting.mp4 \
    --prompt "Summarize the meeting; list action items with timestamps."
```

`--prompt` is free-form. Useful starters:

- `"List every action item with the owner's name and a timestamp."`
- `"Summarize each slide as a single bullet."`
- `"Pull out every customer requirement, grouped by topic."`

### Ask follow-up questions

Once a video is processed (and indexed), you can ask questions about it from any terminal:

```bash
./spectator query "What did Alice say about Isaac Sim?" --target <gpu-machine>
```

### Open the web UI

```bash
./spectator ui --target <gpu-machine>
```

The command prints an `ssh -L` recipe — paste it into another terminal, then open http://localhost:3030 in your browser. The UI gives you drag-and-drop upload, the same Q&A, and a timeline view.

### Tear down when you're done

Stops the docker stack and frees the GPU. The image cache stays, so the next `up` is fast:

```bash
./spectator down --target <gpu-machine>
```

## Where to go next

[**REFERENCE.md**](REFERENCE.md) covers everything else:

- Full SSH config (multiplexing, keepalives, multi-host setup)
- Architecture, topology, and containment policy (what writes where, what doesn't)
- Full subcommand reference table
- Hardware profiles (H100 / L40S / RTX PRO 6000 / Jetson THOR)
- Audio language handling (bilingual, translate-to-English)
- Self-hosted LLM endpoints (override the default `build.nvidia.com`)
- Notes & caveats (iCloud / OneDrive interactions, port conflicts, bring-up timing)
- Iterative development (the `rsync`-only flow for code edits)
- Project layout

For AI agents / IDE assistants working in the codebase, see [AGENTS.md](AGENTS.md).

## Contributing

Issues and PRs welcome at https://github.com/myurasov/Spectator. See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow, coding conventions, and the DCO sign-off requirement (`git commit -s`).

Dev workflow at a glance:

```bash
# rebuild the venv from a clean state
./spectator install --force

# pytest
./spectator test

# ruff check
./spectator lint

# ruff check --fix + ruff format
./spectator fmt
```

For security-sensitive issues, please follow the responsible-disclosure process in [SECURITY.md](SECURITY.md) (do not open a public issue). Third-party dependencies and their licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
