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
    - [audio diarize (speaker diarization)](#audio-diarize)
  - [6. Iterative development](#6-iterative-development)
- [Web UI](#web-ui)
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
| **Web UI state** | `$workdir/ui-server/` (PID file, server log, per-job JSON ledger, uploaded files) | started by `spectator ui-server start` (v0.2.0+) |
| **Per-user state** | `$workdir/` (cloned VSS repo, user-local cache cleaner, audio-in/out/logs, ui-server/) and `~/.docker/config.json` (NGC login) | `spectator install` default |
| **System mutations** | `nvidia-ctk runtime configure`, `systemctl restart docker`, `usermod -aG docker` | **opt-in only** via `spectator install --apply-system` |

The default `spectator install` never writes outside `$workdir` and `~/.docker/config.json`. All system-level changes are gated behind `--apply-system` (with sudo prompts you'll see). `$workdir` defaults to `~/.spectator/` on the target.

## Topology (v3.1 on Spark)

```
┌─ your laptop ───────┐          ┌─ DGX Spark (GB10) ──────────────────┐
│  Spectator CLI      │── ssh ──▶│  Spectator/        (rsynced)         │
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
    # whoever owns ~/.spectator/ on the target
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

`deploy` rsyncs the tool to `~/.spectator/Spectator/` on the Spark, runs `uv sync` there, then runs `spectator install` over SSH (clones the VSS repo, writes a user-local cache-cleaner script, NGC docker login).

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
cd ~/.spectator/Spectator
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

### audio diarize

Speaker diarization via [`pyannote.audio`](https://github.com/pyannote/pyannote-audio). Tells you *who* is speaking, where whisper only tells you *what* they're saying. Runs in the same audio-venv as whisper; default pipeline is `pyannote/speaker-diarization-3.1`.

#### When is the Hugging Face token needed?

- **NEVER at install time.** `./spectator audio install` (with or without `--with-diarize`) does **not** touch Hugging Face. It only pulls the pyannote.audio Python wheel from PyPI, which is public.
- **ONLY when you actually run diarization** — `audio diarize` or `audio transcribe --diarize`. At that point pyannote downloads the model weights from Hugging Face; the download requires a read-scope token AND a one-time license acceptance on the model pages.
- If the token isn't set when you run diarize, Spectator exits with code 2 and prints the URLs you need to visit. The install on disk is fine; you can come back later, set up the token, and re-run without re-installing.

```bash
# one-time setup, before your first diarize call:

# 1. Accept the model licenses in the HF web UI. Pyannote's gate is a
#    multi-field form (Company, Website, Country, Use case), NOT just a
#    checkbox — fill it out and Submit on each page. Just ticking
#    "I accept" leaves the gate locked. README.md downloads as public
#    metadata before the form is submitted, so don't take "I can see
#    the model card" as evidence that access is granted; the weights
#    stay gated until the form is in.
#    Three repos are required because pyannote.audio 4.x reuses the
#    x-vector embedding model from `speaker-diarization-community-1`
#    inside every pipeline (including 3.1):
#    https://huggingface.co/pyannote/speaker-diarization-3.1
#    https://huggingface.co/pyannote/segmentation-3.0
#    https://huggingface.co/pyannote/speaker-diarization-community-1

# 2. Create a read-scope access token at
#    https://huggingface.co/settings/tokens
#    (name it however you like, e.g. "spectator-diarize")

# 3. Persist it via Spectator (writes to $workdir/.creds with chmod 600;
#    subsequent runs source it automatically — no need to re-export):
./spectator audio diarize <any-recording.mp3> \
  --target <gpu-machine> \
  --hf-token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# After step 3 succeeds once, the token is on the target's $workdir/.creds
# and future invocations don't need --hf-token. Rotate keys by editing the
# file directly.
```

```bash
# already installed alongside whisper via `audio install` by default;
# pass --no-with-diarize to skip it for transcription-only hosts.
# Install does NOT need HF auth — it only pulls the public PyPI wheel:
./spectator audio install --target <gpu-machine>

# standalone diarize — writes <stem>.diar.{rttm,json}
./spectator audio diarize meeting.mp3 --target <gpu-machine>

# chained — whisper + diarize + merge in one tmux session
./spectator audio transcribe meeting.mp3 --target <gpu-machine> --diarize

# constrain the detected speaker count
./spectator audio diarize meeting.mp3 --target <gpu-machine> --num-speakers 6
./spectator audio diarize meeting.mp3 --target <gpu-machine> --min-speakers 2 --max-speakers 8
```

Flags shared by `audio diarize` and `audio transcribe --diarize`:

| Flag | Default | Notes |
|---|---|---|
| `--model M` (or `--diarize-model M`) | `pyannote/speaker-diarization-3.1` | Any pyannote pipeline model. The audio-venv pins `pyannote.audio>=4.0,<5` — the 4.x line is the first version that imports cleanly against the cu128 torchaudio wheels we install for Blackwell / GB10 (3.x references the removed `torchaudio.AudioMetaData` type). The 3.1 pipeline file still loads under the 4.x runtime. |
| `--num-speakers N` | unset | Force pyannote to detect exactly this many speakers. Mutually exclusive with `--min-speakers` / `--max-speakers`. |
| `--min-speakers N` | unset | Lower bound. |
| `--max-speakers N` | unset | Upper bound. |
| `--hf-token T` | env: `$HUGGING_FACE_HUB_TOKEN`, fallback `$HF_TOKEN`, then `$workdir/.creds` | First-use captures into `.creds` via Spectator's standard cred-persistence flow. |
| `--device D` | auto-detect (cuda > cpu) | `cuda` / `mps` / `cpu`. Same auto-detect probe as whisper. |
| `--auto-merge / --no-auto-merge` | auto-merge on | (standalone `audio diarize` only) After diarization, if `<stem>.json` from a prior whisper run is on disk, merge into `<stem>.diarized.{json,txt}`. The chained `transcribe --diarize` always merges. |

Output files under `$workdir/audio-out/<stem>/`:

| File | Source | Notes |
|---|---|---|
| `<stem>.diar.rttm` | pyannote | RTTM v1.5 standard format (one `SPEAKER` line per turn). |
| `<stem>.diar.json` | Spectator | Structured turns + per-speaker totals + run metadata. Schema in `spec.txt` § 6. |
| `<stem>.diarized.json` | merge | Whisper segments augmented with a `speaker` field. Same shape as whisper's `<stem>.json`, plus `speaker: "SPEAKER_NN" \| null`. |
| `<stem>.diarized.txt` | merge | Human-readable, grouped by speaker block: `[H:MM:SS] SPEAKER_00:` followed by indented utterance lines. |

Merge algorithm: each whisper segment is assigned the speaker with the largest cumulative overlap against pyannote's turns. Ties go to the alphabetically-first speaker label (deterministic). Segments outside every turn — silence, music interludes, edits — get `speaker: null`.

Known limitations:

- **Shared-room mics** collapse multiple in-room participants into one cluster. pyannote separates voices, not seats; this is the algorithm's nature, not a Spectator bug. Use `--num-speakers` when you know the real count.
- **Model license acceptance is per HF account, per repo, and the gate is a form, not just a checkbox**: a working token isn't enough. Three pyannote repos are gated and each has its own multi-field access form (Company / University, Website, Country, Use case) — you have to fill all the fields AND submit on each page, not just tick the checkbox. The three repos:
  - `pyannote/speaker-diarization-3.1` — the pipeline file.
  - `pyannote/segmentation-3.0` — the segmentation backbone.
  - `pyannote/speaker-diarization-community-1` — the x-vector embedding model. pyannote.audio 4.x bundles the community-1 embedding inside *every* pipeline it ships, including the 3.1 default, so this gate trips even when you never explicitly ask for community-1.

  README downloads as public metadata even before the form is in, which can fool you into thinking access is granted. Once each form is submitted, access is auto-granted within seconds (no manual review queue). Spectator catches the `GatedRepoError` from `Pipeline.from_pretrained(...)` and exits with code 2 + all three URLs so you know where to go.
- **v3.x is no longer supported**: the audio-venv pins `pyannote.audio>=4.0,<5`. The 3.x line references the removed `torchaudio.AudioMetaData` type and fails to import on every modern torchaudio. If you need the older 3.x behavior, install it manually into the audio-venv at your own risk — it won't import against the cu128 wheels Spectator's `audio install` ships.

Performance on a DGX Spark (NVIDIA GB10 Grace Blackwell): ~5-10 s model load + warmup, then ~50-150× faster than real-time for the diarization itself. A 1-hour recording diarizes in about 30-60 s once the model is in GPU memory.

### 6. Iterative development

When you're editing Spectator's own source and just want to push code changes (no `uv sync`, no install):

```bash
# ~5s
./spectator rsync --target <gpu-machine>
```

Use `./spectator deploy …` when you've changed `pyproject.toml` (deps) or want a full re-install.

## Web UI

`spectator ui-server start` launches a long-lived FastAPI service (uvicorn) that wraps Spectator's CLI behind a single-page UI. Drag-drop upload, live progress per job, VSS lifecycle controls, output download, video / audio Q&A. The server is detached from the launching shell and survives `Ctrl-C`; its PID is tracked at `$workdir/ui-server/server.pid`.

### Lifecycle commands

```bash
# default: bind 127.0.0.1:7777 against the local target
./spectator ui-server start

# remote target — all jobs submitted via the UI will use this SSH alias
./spectator ui-server start --target <gpu-machine>

# expose on the LAN — no built-in auth, only do this on a network you trust
./spectator ui-server start --bind 0.0.0.0 --port 7777

./spectator ui-server status
./spectator ui-server logs --follow
./spectator ui-server stop
```

### State layout (under `$workdir/ui-server/`)

- `server.pid` — current server's PID (used by `stop` / `status`)
- `server.log` — uvicorn stdout / stderr; tailed by `ui-server logs`
- `server.json` — the running server's actual `bind` / `port` / `target` / `workdir` (v0.3.1+); written by `start`, deleted by `stop`. Used to surface the real config in `status` and to detect when a follow-up `start` was passed conflicting flags
- `jobs/<uuid>.json` — one persistent JSON file per job (the schema is the public contract for any agent watching alongside the UI)
- `jobs/<uuid>.log` — captured stdout / stderr of the spawned subprocess (audio transcribe, video process)
- `uploads/` — uploaded files; audio jobs are then copied to `$workdir/audio-in/<basename>` for Spectator to pick up

### Conflict detection on repeated `start`

`ui-server start` is idempotent on its happy path — re-running it while the server is already up just prints a "Web UI already running" reminder. The idempotency check compares the requested `--bind` / `--port` / `--target` against the running server's persisted config (`server.json`) and refuses with an error if any of them differ:

```
$ ./spectator ui-server start                    # default bind 127.0.0.1
Web UI started (pid 12345).
  url: http://127.0.0.1:7777/

$ ./spectator ui-server start --bind 0.0.0.0     # asking for a different bind
Conflict: Web UI already running (pid 12345) with:
  --bind 127.0.0.1 (asked for 0.0.0.0)

Stop it first to apply the new flags:

  ./spectator ui-server stop
  ./spectator ui-server start --bind 0.0.0.0 --port 7777
```

Before v0.3.1 the second invocation silently no-op'd, leaving the server bound to the original address and printing a misleading `url:` line that just echoed the user's new flags.

### Legacy-workdir scan (v0.3.3+)

The same-`$workdir` conflict check above only sees servers whose state lives at the requested `$workdir`. When the default `$workdir` itself changes between releases (`v0.2.x ~/spectator → v0.3.0 ~/.spectator-workdir → v0.3.2 ~/.spectator`), a Web UI server started before the upgrade keeps running under the **old** path, invisible to the new `start` invocation.

`ui-server start` therefore also scans every historical default `$workdir` (the `LEGACY_WORKDIRS` constant in `webui/server.py`) for any pid-alive Web UI server. On a hit:

- **Same port** → hard error. The new uvicorn would fail to bind anyway, and the old server's stale `$workdir` would also break VSS calls (`cd $workdir/video-search-and-summarization` against a path that no longer holds the install).
- **Different port** → yellow warning. The two servers can coexist, but if the old one is leftover from an upgrade it's worth retiring it explicitly with `./spectator ui-server stop --workdir <legacy>`.

### HTTP surface (REST + WebSocket)

| Method + path | Purpose |
|---|---|
| `GET  /` | single-page UI (static HTML/JS/CSS) |
| `GET  /api/status` | overall: VSS reachable, audio-venv installed (local), jobs in flight |
| `POST /api/vss/up` / `down` | lifecycle wrappers around `spectator up` / `down` |
| `GET  /api/vss/status` | wrapper around `spectator status` |
| `GET  /api/jobs` | list all jobs (newest first) |
| `POST /api/jobs` | multipart upload + form params; spawns `spectator audio transcribe` or `spectator process` |
| `GET  /api/jobs/{id}` | single-job detail incl. metrics and PID |
| `DELETE /api/jobs/{id}` | kill (SIGTERM the subprocess group, `tmux kill-session` for remote) |
| `GET  /api/jobs/{id}/log` | tail of the subprocess log (default last 64 KB) |
| `GET  /api/jobs/{id}/output/{filename}` | download an output file (path-traversal-blocked) |
| `WS   /api/jobs/{id}/progress` | live JSON snapshots (~every 2 s) — segments processed, rt-factor, ETA, device, finished/exit-code |
| `POST /api/query/video` | body `{question, file_ids?}` — proxies to VSS's `/v1/chat/completions` |
| `POST /api/query/audio` | body `{job_id, question}` — feeds the completed audio job's transcript text to the same NIM endpoint VSS uses (needs `NVIDIA_API_KEY` in the server's environment) |

The `/api/jobs` form fields:

| Field | Required | Notes |
|---|---|---|
| `kind` | yes | `audio` or `video` |
| `file` | yes | the audio / video upload (multipart) |
| `quality` | no (audio) | Spectator preset: `studio` / `meeting` / `phone` / `extreme` |
| `language` | no (audio) | ISO-639-1 code; omit for auto-detect |
| `task` | no (audio) | `transcribe` (default) or `translate` |
| `model` | no | override model name |
| `device` | no | force `cuda` / `mps` / `cpu` |
| `prompt` | no (video) | summarization prompt |

### Security defaults

- **Bind**: `127.0.0.1` (localhost-only) — the UI has no auth. Pass `--bind 0.0.0.0` only on a network you trust.
- **No CSRF**: the API is plain JSON / multipart; we assume same-origin from the bundled UI. Don't expose this to browsers from third-party sites.
- **No write outside `$workdir`**: jobs follow Spectator's containment policy; the UI doesn't relax it.

### Live performance metrics

The WebSocket payload at `/api/jobs/{id}/progress` looks like:

```json
{
  "job_id": "...",
  "kind": "audio",
  "status": "running",
  "snapshot": {
    "audio_duration_s": 4522.0,
    "processed_s": 765.3,
    "wall_clock_s": 252.1,
    "rt_factor": 3.04,
    "percent": 16.93,
    "eta_s": 1235.4,
    "device": "mps",
    "finished": false,
    "exit_code": null
  },
  "audio_duration_human": "1h 15m 22s",
  "processed_human": "12m 45s",
  "wall_clock_human": "4m 12s",
  "eta_human": "20m 35s"
}
```

The frontend updates the per-job row's progress bar, rt-factor, wall-clock, and ETA cells in place. When `finished: true` the snapshot includes the exit code and the WebSocket closes; the ledger transitions the job to `completed` (rc=0) or `failed` (rc≠0) and the row's status pill updates on the next poll.

### Backend / frontend code layout

```
src/webui/
├── __init__.py
├── _launch.py             # uvicorn-importable; reads SPECTATOR_UI_{WORKDIR,TARGET}
├── server.py              # create_app() factory + state-paths helper
├── jobs.py                # Job dataclass + JobLedger persistence
├── pipeline.py            # subprocess wrapper for audio / video; PID + tmux kill
├── progress.py            # whisper-segment parser + ffprobe duration probe
├── routes/
│   ├── __init__.py
│   ├── status.py          # GET /api/status
│   ├── vss.py             # /api/vss/up | down | status
│   ├── jobs.py            # POST/GET/DELETE /api/jobs[/{id}], log, output
│   ├── ws.py              # WS /api/jobs/{id}/progress
│   └── query.py           # POST /api/query/{video,audio}
└── static/
    ├── index.html         # single-page UI
    ├── app.js             # vanilla JS — no framework, no build step
    └── style.css
```

## Subcommand reference

| Command | What it does |
|---|---|
| `spectator help [command]` | Curated end-to-end overview, or per-subcommand help |
| `spectator preflight [--target HOST]` | Driver / CUDA / docker / nvidia-ctk / NGC checks |
| `spectator install [--target HOST] [--ngc-key …]` | **User-space only**: clone VSS repo, write user-local cache-cleaner, NGC login |
| `spectator install --apply-system` | …plus: `sudo nvidia-ctk runtime configure`, `sudo systemctl restart docker`, `sudo usermod -aG docker` |
| `spectator uninstall [--target HOST] [--force]` | Inverse of `install`: stop everything, then `rm -rf $workdir/`. Confirms before acting; pass `--force` for non-interactive scripts. Leaves docker images, `~/.docker/config.json`, and `--apply-system` mutations alone (each needs separate manual cleanup). |
| `spectator rsync --target HOST` | Just rsync the source tree (no `uv sync`, no install) — for iterative dev |
| `spectator deploy --target HOST` | Full sync: rsync + remote `uv sync` + remote `install` |
| `spectator up [--target HOST]` | Bring up the stack via `dev-profile.sh up -p base -H DGX-SPARK --use-remote-llm --llm nvidia/nvidia-nemotron-nano-9b-v2` (in tmux) |
| `spectator down [--target HOST]` | Stop everything Spectator launches on the target: VSS docker stack, the `spectator-up` tmux session, and any `audio-*` tmux sessions (per-job transcribe jobs). **Non-destructive** — `$workdir/` stays on disk so the next `up` is fast. The user-local cache cleaner is left running (it needs sudo) — surface only; retire with `spectator system cache-cleaner-stop` separately. For full disk removal, use `spectator uninstall`. |
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
| `spectator ui-server start [--port N] [--bind ADDR] [--target HOST]` | start the persistent Web UI server (detached uvicorn process); see [Web UI](#web-ui) |
| `spectator ui-server stop` | SIGTERM the running Web UI server |
| `spectator ui-server status` | report whether the Web UI is running + state-dir paths |
| `spectator ui-server logs [--follow] [--lines N]` | tail the Web UI server's log |

All commands accept `--target HOST` (an SSH alias). Without `--target`, the command runs on the local machine.

## Required env vars / keys

| Variable | When | Where to get it |
|---|---|---|
| `NGC_CLI_API_KEY` | always (pulls images from nvcr.io) | https://org.ngc.nvidia.com/setup/api-keys |
| `NVIDIA_API_KEY` | when using remote LLM endpoints (default) | https://build.nvidia.com (Get API Key) |
| `LLM_ENDPOINT_URL` | overrides the default `https://integrate.api.nvidia.com/v1` | only if self-hosting LLM |
| `SPECTATOR_ALLOW_MPS_AUTO` | optional — set to `1` to re-enable MPS in the auto-detect path on Apple Silicon | see [Audio device selection](#audio-device-selection) |

Spectator never reads outside its own folder; it only sees the env vars.

### Persistent creds at `$workdir/.creds`

As of v0.4.4, the first `spectator install` writes `$workdir/.creds` (chmod 600) capturing whatever Spectator-managed env vars are set in the install shell. Subsequent bash payloads (install / audio install / up) source that file at the top, so the values in it become authoritative — they override anything passed via `--ngc-key`, `--nvidia-key`, `LLM_ENDPOINT_URL`, or shell exports.

Read priority (highest first):

1. `$workdir/.creds` — sourced by every bash payload that needs creds.
2. SSH-propagated env / process env — what the user passed via flags or shell.
3. None — caller errors if a required value is empty.

Format: shell-source-able (`export VAR=VALUE`, one per line, value quoted via `printf %q`).

Rotating keys: edit `.creds` directly. Spectator never overwrites an existing `.creds`; the file is yours after the first install.

Removing the file: `spectator uninstall` removes `$workdir/` (which includes `.creds`). Manual `rm $workdir/.creds` is also fine — the next install will recreate it from current env vars.

`.creds` is excluded from `rsync` and packaging — it never travels off the host it was written on. (`creds.txt`, `.creds`, and `*.creds` are all in the rsync exclude list in `deploy.py`.)

## Hardware profiles

`spectator up --hardware DGX-SPARK` is the default. Other v3.1 profiles:

- `H100`, `L40S`, `RTXPRO6000BW` — workstation/server GPUs (full local LLM possible)
- `IGX-THOR`, `AGX-THOR` — Jetson edge

Pick by passing `--hardware <profile>` (and adjust `--llm` accordingly — server-class GPUs can run the LLM locally).

## Audio device selection

Spectator auto-detects the best torch device available in the audio-venv before each `audio transcribe` run. The default order is **`cuda`** (NVIDIA GPU) → **`cpu`** (fallback). **MPS (Apple Silicon GPU) is deliberately skipped from the auto-detect path** as of v0.4.1 — see the [MPS limitation](#mps-limitation-apple-silicon) note below for why. Detection is a one-shot probe via the audio-venv's Python — no caching, so a hot-swap of the audio-venv between runs is picked up correctly.

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
| Apple Silicon via MPS | 2-4× faster than real-time *(when working — see below)* |
| Apple Silicon via CPU | real-time to 2× slower |
| Intel Mac via CPU | 5-15× slower than real-time |

### MPS limitation (Apple Silicon)

**Bottom line**: as of `openai-whisper 20250625` × `torch 2.11`, MPS is **not a usable option for any Whisper model Spectator could meaningfully ship**. CPU is the only working local path on Apple Silicon. For real GPU acceleration, send work to a CUDA target via `--target <gpu-machine>`.

**`large-v3` family crash**:

```
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework
doesn't support float64. Please use float32 instead.
```

All four of Spectator's quality presets (`studio` / `meeting` / `phone` / `extreme`) use `large-v3` or `large-v3-turbo`, so this affects every transcribe invocation that lands on MPS via auto-detect. Tracked upstream: [openai/whisper#2151](https://github.com/openai/whisper/issues/2151).

**Tried-and-doesn't-work table** (M-series Mac, `openai-whisper 20250625`, `torch 2.11.0`, tested 2026-05-09):

| `--model`         | MPS behavior |
|---|---|
| `base`            | Different crash: `-inf logits` saturate the categorical distribution |
| `small`           | `float64` crash on first decode block |
| `medium` (≤ 30 s) | "Succeeds" with rc=0 but **silently writes empty transcripts** — only the `[00:00.000 --> 00:30.000]` timestamp header, no segments |
| `medium` (full)   | `float64` crash partway through |
| `large-v3-turbo`  | `float64` crash on first decode block |
| `large-v3`        | `float64` crash (same as `large-v3-turbo`) |

So no `--model` override + `--device mps` produces a usable transcript on the current upstream stack.

**v0.4.1 mitigation**: auto-detect skips MPS by default and prefers CPU on Apple Silicon, with a one-line warning explaining the downgrade.

Real-world impact: `~real-time` on Apple Silicon CPU for a `meeting`-quality 30-min recording on an M-series Mac. Tractable for occasional use; for anything > 30 min, the CUDA target is dramatically faster (~10-30×).

**Override knobs (won't actually help, but documented for completeness)**:

- **Per-invocation**: `--device mps` is always honored; expect a crash or empty output.
- **Globally**: set `SPECTATOR_ALLOW_MPS_AUTO=1` in your shell to re-include MPS in the auto-detect order.

When upstream lands a fix in `openai/whisper`, this section gets updated and the auto-detect default likely flips back to `cuda > mps > cpu`.

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
- **`down` vs `uninstall` (lifecycle vs cleanup)**: `spectator down` is **non-destructive** — it stops the VSS docker stack, kills the `spectator-up` tmux session, and kills any per-job `audio-*` tmux sessions, but leaves `$workdir/` on disk so a subsequent `spectator up` is fast (no re-clone, no audio-venv re-install, no docker image re-pull). Use `down` between work sessions. Use `spectator uninstall` only when you actually want the install gone — it runs `down` first, then `rm -rf $workdir/`. Three categories of state are deliberately left alone by `uninstall` so it doesn't clobber things shared with the rest of the system: docker images cached locally (~30 GB; `docker images | grep nvcr.io` to list, `docker rmi <id>` to remove), the NGC docker login at `~/.docker/config.json` (revoke with `docker logout nvcr.io`), and any system-level mutations from `install --apply-system` (nvidia-ctk runtime config, docker group membership — these need explicit admin reversal because `--apply-system` itself was opt-in).
- **Migrating to the v0.3.2 default layout**: the on-disk layout has shifted twice in the v0.3 line. Current default is `~/.spectator/` (outer `$workdir`) with the rsynced project tree at `~/.spectator/Spectator/` (governed by `config.TOOL_TREE_RELPATH`). To get there from any prior version:

  | Coming from | Outer dir | Inner-tree dir | One-shot mv |
  |---|---|---|---|
  | v0.2.x | `~/spectator/` | `spectator-tool/` | `mv ~/spectator/spectator-tool ~/spectator/Spectator && mv ~/spectator ~/.spectator` |
  | v0.3.0 / v0.3.1 | `~/.spectator-workdir/` | `Spectator/` | `mv ~/.spectator-workdir ~/.spectator` |

  After the `mv`, redeploy from your laptop so v0.3.2 lands on the host:

  ```bash
  ssh <gpu-machine> '<one-shot mv from the table above>'
  ./spectator deploy --target <gpu-machine>
  ```

  Rsync excludes `.venv/`, so the audio venv and other cached state under the renamed tree survive the move. If you'd rather keep the old `$workdir` path on the host (e.g. you have shell history / ssh-config snippets pointing at it), pass `--workdir ~/spectator` (or `~/.spectator-workdir`) on every call instead — but the inner-tree dir still needs to be `Spectator/`, so you may still need one `mv` if you're coming from v0.2.x. New installs need no migration — the v0.3.2 default just works.

## Layout

```
spectator/
├── pyproject.toml            # name = "spectator", [tool.uv] package = false
├── README.md                 # beginner-friendly setup walk-through
├── REFERENCE.md              # this file — full reference
├── AGENTS.md                 # entry point for AI / IDE assistants
├── CONTRIBUTING.md           # PR workflow + DCO sign-off requirement
├── SECURITY.md               # private-disclosure process
├── THIRD_PARTY_NOTICES.md    # runtime / dev / external-service licenses
├── LICENSE                   # Apache-2.0
├── spectator                 # thin shell wrapper (install/test/lint/fmt + CLI forwarder)
├── ai/                       # AI-agent-readable spec + maintainer rules
│   ├── dev.agent.md          # primary instruction file for AI IDEs
│   ├── dev.memory.md         # accumulated maintainer preferences
│   └── spec.txt              # canonical architecture / CLI surface spec
├── src/                      # invoked as `python -m src` with PYTHONPATH=<root>
│   ├── __init__.py           # __version__
│   ├── __main__.py           # `python -m src` entrypoint
│   ├── cli.py                # typer entrypoint (help, install, deploy, rsync,
│   │                         # up, down, status, logs, ui, process, query,
│   │                         # audio, system, ui-server)
│   ├── config.py             # constants (DEFAULT_REMOTE_WORKDIR,
│   │                         # TOOL_TREE_RELPATH, ports) + StackConfig dataclass
│   ├── _run.py               # local + ssh subprocess primitives
│   ├── preflight.py          # driver / CUDA / docker / NGC checks
│   ├── install.py            # idempotent install bash script
│   ├── deploy.py             # rsync_only + full deploy (rsync + uv sync + install)
│   ├── stack.py              # up/down/status/logs (wraps dev-profile.sh)
│   ├── api.py                # upload / summarize / query (REST + OpenAI-compat)
│   ├── audio.py              # whisper install + transcribe + status + fetch
│   │                         # (auto-detects cuda > mps > cpu)
│   └── webui/                # persistent FastAPI Web UI (added in v0.2.0)
│       ├── _launch.py        # uvicorn-importable entry point
│       ├── server.py         # create_app() factory + state-paths helper
│       ├── jobs.py           # Job dataclass + JobLedger persistence
│       ├── pipeline.py       # subprocess wrapper for audio / video
│       ├── progress.py       # whisper-segment parser + ffprobe duration probe
│       ├── routes/           # FastAPI routers (status, vss, jobs, ws, query)
│       └── static/           # vanilla single-page UI (index.html + app.js + style.css)
└── tests/
    ├── test_smoke.py         # CLI surface, version, device detection, config pins
    └── test_webui.py         # JobLedger persistence, progress parser, FastAPI routes
```

Roughly 2.4 kLoC of Python in the non-Web-UI core, plus 1.2 kLoC of FastAPI routers and 700 LoC of static-frontend HTML / CSS / JS. The heavy lifting is delegated to upstream VSS and Whisper.
