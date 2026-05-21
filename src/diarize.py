# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Speaker diarization via pyannote.audio.

Sibling pipeline to :mod:`audio` (Whisper transcription). Same containment
contract: never writes outside ``$workdir`` on the target. The on-disk
layout mirrors transcription's:

    $workdir/audio-in/<basename>           # uploaded audio
    $workdir/audio-out/<stem>/             # per-recording output dir
        <stem>.diar.rttm                   # RTTM standard format
        <stem>.diar.json                   # structured turns + meta
        <stem>.diarized.json               # whisper segments + speaker (merge)
        <stem>.diarized.txt                # human-readable merged form

The merged forms only exist when a sibling ``<stem>.json`` from
Whisper is present alongside the diarization output (the
``audio transcribe --diarize`` flow produces both; standalone
``audio diarize`` produces only the ``.diar.*`` pair).

Auth: pyannote model downloads need a Hugging Face read-token. Spectator
persists it at ``$workdir/.creds`` as ``HUGGING_FACE_HUB_TOKEN`` (the HF
standard variable name). One-time setup the operator does outside
Spectator:

  1. Create a read-scope token at https://huggingface.co/settings/tokens
  2. Accept the model licenses on each of THREE repo pages (pyannote.audio
     4.x reuses the community-1 x-vector embedding inside every pipeline,
     including the 3.1 default). Each gate is a multi-field FORM
     (Company, Website, Country, Use case) — fill all fields and Submit
     on each page; the README of a gated repo downloads as public
     metadata even before the form is in, which can fool a quick check
     into thinking access is granted:
       https://huggingface.co/pyannote/speaker-diarization-3.1
       https://huggingface.co/pyannote/segmentation-3.0
       https://huggingface.co/pyannote/speaker-diarization-community-1

After that the token round-trips through ``.creds`` like any other
Spectator credential.

The pipeline default is ``pyannote/speaker-diarization-3.1`` — the most
widely deployed pyannote pipeline as of late 2024, stable across the
3.1.x line. Override with ``--model`` per call. The audio-venv pins
``pyannote.audio>=3.1,<4`` so the 4.x line (newer, different API
shape) is an explicit opt-in once the maintainer validates it.
"""

from __future__ import annotations

import shlex
import textwrap
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from . import _creds, audio, config
from ._run import RunResult, run, ssh_run, ssh_stream

console = Console()


# ---------------------------------------------------------------------------
# defaults + pin range
# ---------------------------------------------------------------------------

DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-3.1"

# Pin baseline. 4.x is the current line that imports cleanly against
# torchaudio >= 2.2 (the 3.x line referenced the now-removed
# ``torchaudio.AudioMetaData`` type, which broke import on every host
# running a modern torch wheel — including the cu128 wheels we install
# for Blackwell / GB10). 4.x renamed ``use_auth_token`` to ``token``
# on ``Pipeline.from_pretrained`` but kept the rest of the surface
# (``pipeline(audio, num_speakers=N, ...)`` and the
# ``Annotation.itertracks`` / ``write_rttm`` outputs) compatible with
# what Spectator uses. The default model (3.1) still loads on 4.x.
PYANNOTE_PIN = "pyannote.audio>=4.0,<5"

# Environment variables that hold the HF token, in priority order. The
# canonical name is HUGGING_FACE_HUB_TOKEN (used by huggingface_hub itself);
# HF_TOKEN is the shorter alias most code in the wild also recognizes.
HF_TOKEN_ENV_VARS: tuple[str, ...] = ("HUGGING_FACE_HUB_TOKEN", "HF_TOKEN")


# ---------------------------------------------------------------------------
# install pyannote.audio into the audio-venv (idempotent)
# ---------------------------------------------------------------------------


def install_diarize_deps(host: str | None, cfg: config.StackConfig) -> RunResult:
    """Install :mod:`pyannote.audio` into the audio-venv at ``$workdir/audio-venv/``.

    Idempotent: the bash payload probes ``import pyannote.audio`` first and
    only invokes ``uv pip install`` when the import fails. Designed to run
    alongside :func:`audio.install_audio_venv` (which already installs
    ``torch + openai-whisper``); pyannote piggy-backs on the same torch.

    Called automatically from ``audio install --with-diarize`` (default
    ON) and also exposed standalone so the operator can backfill an
    existing audio-venv without re-running the whole audio install.
    """
    workdir_bash = audio._expand_tilde(cfg.workdir)
    venv = audio._venv_dir(cfg)
    script = textwrap.dedent(f"""
        set -e
        export PATH="$HOME/.local/bin:$PATH"

        {_creds.source_block(workdir_bash)}

        VENV={audio._bash_dq(venv)}
        if [ ! -d "$VENV" ]; then
          echo "==== audio-venv missing at $VENV — run 'spectator audio install' first ===="
          exit 1
        fi
        PY="$VENV/bin/python"

        if "$PY" -c "import pyannote.audio" 2>/dev/null; then
          PYANNOTE_VER=$("$PY" -c "import pyannote.audio as m; print(m.__version__)" 2>/dev/null)
          echo "==== pyannote.audio already installed (version $PYANNOTE_VER) ===="
        else
          echo "==== installing pyannote.audio ($PYANNOTE_PIN range) ===="
          uv pip install --python "$PY" "{PYANNOTE_PIN}"
        fi

        echo
        echo "==== verify ===="
        "$PY" - <<'PY_EOF'
import pyannote.audio as pa
import torch

print("pyannote.audio:", pa.__version__)
print("torch:", torch.__version__, " cuda available:", torch.cuda.is_available())
PY_EOF
        echo "==== diarize deps ready ===="
    """).strip()
    if host:
        return ssh_run(host, script, env=cfg.env_block())
    return run(["bash", "-c", script])


# ---------------------------------------------------------------------------
# diarize python script: emitted onto target and run inside the audio-venv
# ---------------------------------------------------------------------------


def _diarize_python_script(
    audio_path_bash: str,
    out_dir_bash: str,
    stem: str,
    *,
    model: str,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    device: str,
) -> str:
    """Emit the Python source executed on the target by the audio-venv.

    The script lives inside a bash heredoc; values that may contain shell
    metacharacters are interpolated as bash variables (``$AUDIO_PATH`` etc.)
    rather than baked into the Python source, so a stem with spaces or
    quotes can't break the surrounding script.

    Output paths emitted by the script:
      ``<out_dir>/<stem>.diar.rttm`` — RTTM v1.5 format
      ``<out_dir>/<stem>.diar.json`` — structured turns + per-speaker totals
    """
    constraint_args: list[str] = []
    if num_speakers is not None:
        constraint_args.append(f"num_speakers={num_speakers}")
    elif min_speakers is not None or max_speakers is not None:
        if min_speakers is not None:
            constraint_args.append(f"min_speakers={min_speakers}")
        if max_speakers is not None:
            constraint_args.append(f"max_speakers={max_speakers}")
    pipeline_call_tail = (", " + ", ".join(constraint_args)) if constraint_args else ""

    script_body = textwrap.dedent(f"""
        import json
        import os
        import sys
        import time
        from pathlib import Path

        AUDIO_PATH = os.environ["DIARIZE_AUDIO_PATH"]
        OUT_DIR = Path(os.environ["DIARIZE_OUT_DIR"])
        STEM = os.environ["DIARIZE_STEM"]
        MODEL = {model!r}
        DEVICE = {device!r}

        # HF token resolution mirrors the CLI's: explicit env var wins; if
        # neither standard name is set, surface the same actionable error
        # the CLI would have surfaced earlier (defense in depth — the bash
        # payload also pre-checks).
        token = None
        for var in ({", ".join(repr(v) for v in HF_TOKEN_ENV_VARS)}):
            v = os.environ.get(var)
            if v:
                token = v
                break
        if not token:
            print(
                "ERROR: HUGGING_FACE_HUB_TOKEN is not set.\\n"
                "Get a read-scope token at https://huggingface.co/settings/tokens\\n"
                "Then accept the model licenses (one-time, three repos —\\n"
                "pyannote.audio 4.x bundles the community-1 embedding in\\n"
                "every pipeline; each gate is a multi-field form, not just\\n"
                "a checkbox):\\n"
                "  https://huggingface.co/pyannote/speaker-diarization-3.1\\n"
                "  https://huggingface.co/pyannote/segmentation-3.0\\n"
                "  https://huggingface.co/pyannote/speaker-diarization-community-1\\n"
                "Persist via:\\n"
                "  echo 'export HUGGING_FACE_HUB_TOKEN=hf_...' >> $WORKDIR/.creds",
                file=sys.stderr,
            )
            sys.exit(2)

        print(f"loading pipeline: {{MODEL}}")
        t0 = time.monotonic()
        from pyannote.audio import Pipeline
        import torch
        try:
            from huggingface_hub.errors import GatedRepoError
        except ImportError:
            from huggingface_hub.utils import GatedRepoError  # older HF hub

        try:
            pipeline = Pipeline.from_pretrained(MODEL, token=token)
        except GatedRepoError as e:
            # The token authenticates but the user hasn't completed the
            # license form on a model page. README downloads as public
            # metadata; the weights and config.yaml stay gated until the
            # form is filled out (not just the checkbox — pyannote's
            # gate has a multi-field form). pyannote.audio 4.x reuses
            # the x-vector embedding model from
            # ``pyannote/speaker-diarization-community-1`` regardless of
            # which pipeline you ask for, so its license also has to be
            # accepted. Surface all three URLs so the operator hits
            # the right page even on first-fail.
            print(
                "ERROR: pyannote model is gated and the licenses aren't accepted "
                "for this HF account yet.\\n"
                "Log in to https://huggingface.co/, then visit each URL below "
                "and fill out the access form (not just the checkbox):\\n"
                "  https://huggingface.co/pyannote/speaker-diarization-3.1\\n"
                "  https://huggingface.co/pyannote/segmentation-3.0\\n"
                "  https://huggingface.co/pyannote/speaker-diarization-community-1\\n"
                "    (^ x-vector embedding reused by pyannote.audio 4.x for "
                "every pipeline, including 3.1)\\n"
                "Re-run after all three forms are submitted (access is granted "
                "within a few seconds).\\n"
                "Underlying error: " + str(e)[:200],
                file=sys.stderr,
            )
            sys.exit(2)

        if pipeline is None:
            print(
                "ERROR: pipeline load returned None. This usually means the\\n"
                "token works but the model file at the requested revision is\\n"
                "incompatible with this pyannote.audio version. Try a\\n"
                "different --model (default pyannote/speaker-diarization-3.1\\n"
                "is known to load on pyannote.audio>=4.0).",
                file=sys.stderr,
            )
            sys.exit(2)

        if DEVICE != "cpu":
            try:
                pipeline.to(torch.device(DEVICE))
            except Exception as e:
                print(f"WARNING: pipeline.to({{DEVICE!r}}) failed, falling back to cpu: {{e}}",
                      file=sys.stderr)

        # Workaround for new-silicon-meets-recent-cu: nvrtc (NVIDIA's runtime
        # kernel compiler bundled with PyTorch's cu128 wheels) doesn't know
        # about Blackwell GB10's sm_121 compute capability, so the elementwise
        # ``.abs()`` fusion path on complex tensors — used by pyannote's
        # wespeaker fbank computation as ``torch.fft.rfft(x).abs()`` — fails
        # with ``nvrtc: error: invalid value for --gpu-architecture (-arch)``.
        # The fix lands when PyTorch ships nvrtc with an sm_121 allowlist
        # (cu129+ or torch nightlies on cu13). Until then, probe at runtime
        # and surgically patch ``Tensor.abs`` to compute the magnitude
        # manually (``sqrt(real**2 + imag**2)``) for the complex-CUDA case
        # only. Each of those ops is a basic eager kernel that doesn't
        # trip the JIT, so the rest of the pipeline runs unmodified on GPU.
        # Real-valued ``.abs()`` calls and CPU complex ``.abs()`` calls are
        # left untouched.
        if DEVICE == "cuda" and torch.cuda.is_available():
            try:
                torch.fft.rfft(torch.randn(8, device="cuda")).abs()
            except RuntimeError as e:
                if "nvrtc" in str(e) or "gpu-architecture" in str(e):
                    _orig_abs = torch.Tensor.abs

                    def _safe_abs(self):
                        if self.is_cuda and self.is_complex():
                            return torch.sqrt(self.real * self.real + self.imag * self.imag)
                        return _orig_abs(self)

                    torch.Tensor.abs = _safe_abs
                    print("NOTE: patched Tensor.abs to use manual magnitude for "
                          "complex CUDA tensors (cu128 nvrtc lacks sm_121+ "
                          "support for this GPU); other ops unaffected.",
                          file=sys.stderr)
                else:
                    raise

        print(f"pipeline ready in {{time.monotonic() - t0:.1f}}s; running on {{AUDIO_PATH}}")
        # Custom plain-text hook for pyannote's pipeline callback protocol.
        # pyannote ships ``ProgressHook`` under
        # ``pyannote.audio.pipelines.utils.hook``, but it's built on
        # ``rich.progress.Progress`` with a default Console — which
        # auto-detects ``sys.stdout.isatty() == False`` (our tmux+redirected
        # log case) and silently suppresses every render. So pyannote's hook
        # is useless for the way we run diarize. The protocol itself is
        # public + stable: any callable with the signature below is invoked
        # by pyannote on each step transition + intra-step progress event.
        # Emitting plain lines makes the otherwise-silent inference middle
        # (segmentation → embeddings → discrete_diarization, often minutes
        # on long files) visible in the run log.
        class _LineProgressHook:
            def __init__(self):
                self._step = None
                self._step_t0 = None
                self._last_pct = -1
                self._t0 = time.monotonic()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return

            def __call__(self, step_name, step_artifact, file=None,
                         total=None, completed=None):
                now = time.monotonic()
                if step_name != self._step:
                    if self._step is not None and self._step_t0 is not None:
                        print(
                            f"  [{{now - self._step_t0:5.1f}}s] "
                            f"{{self._step}}: done",
                            flush=True,
                        )
                    self._step = step_name
                    self._step_t0 = now
                    self._last_pct = -1
                    print(
                        f"→ {{step_name}} "
                        f"(started at {{now - self._t0:.1f}}s)",
                        flush=True,
                    )
                if total and completed is not None and total > 0:
                    pct = int(100 * completed / total)
                    if pct >= self._last_pct + 10 or completed >= total:
                        bar_len = 30
                        filled = int(bar_len * completed / total)
                        bar = "#" * filled + "." * (bar_len - filled)
                        print(
                            f"  [{{now - self._step_t0:5.1f}}s] "
                            f"{{step_name}} [{{bar}}] {{pct:3d}}% "
                            f"({{completed}}/{{total}})",
                            flush=True,
                        )
                        self._last_pct = pct

        t1 = time.monotonic()
        with _LineProgressHook() as hook:
            result = pipeline(AUDIO_PATH{pipeline_call_tail}, hook=hook)
        elapsed = time.monotonic() - t1

        # pyannote.audio 4.x returns a ``DiarizeOutput`` object with
        # attributes ``speaker_diarization`` (Annotation), ``exclusive_
        # speaker_diarization`` (Annotation, no overlapping turns), and
        # ``speaker_embeddings`` (numpy array). 3.x returned the
        # ``Annotation`` directly. Probe and unwrap so this code stays
        # forward-compatible if the maintainer drops back to 3.x or
        # adopts a future 4.x variant.
        if hasattr(result, "speaker_diarization"):
            diarization = result.speaker_diarization
        else:
            diarization = result

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        rttm_path = OUT_DIR / f"{{STEM}}.diar.rttm"
        json_path = OUT_DIR / f"{{STEM}}.diar.json"

        # Build the JSON turn list FIRST so we never lose the diarization
        # result to a downstream write failure (the pyannote pipeline call
        # is the expensive step — 5-15 min on this hardware; we don't want
        # to recompute over a serialization bug).
        turns: list[dict] = []
        for segment, _, label in diarization.itertracks(yield_label=True):
            turns.append({{
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "duration": round(float(segment.end) - float(segment.start), 3),
                "speaker": label,
            }})
        turns.sort(key=lambda t: t["start"])

        # RTTM is a space-separated format; pyannote's writer refuses
        # file URIs containing spaces (would corrupt the columns).
        # Set a safe slug URI before writing — the URI is metadata only,
        # downstream tooling reads it from the RTTM's filename anyway.
        import re
        safe_uri = re.sub(r"[^A-Za-z0-9._-]+", "-", STEM).strip("-") or "audio"
        diarization.uri = safe_uri
        try:
            with open(rttm_path, "w") as fh:
                diarization.write_rttm(fh)
        except Exception as e:
            print(f"WARNING: RTTM write failed ({{e}}); JSON output is still written below.",
                  file=sys.stderr)

        per_speaker: dict[str, float] = {{}}
        for t in turns:
            per_speaker[t["speaker"]] = per_speaker.get(t["speaker"], 0.0) + t["duration"]
        per_speaker_sorted = sorted(per_speaker.items(),
                                     key=lambda kv: kv[1], reverse=True)

        total_speech = sum(per_speaker.values())
        audio_duration = float(turns[-1]["end"]) if turns else 0.0

        payload = {{
            "schema_version": 1,
            "audio": AUDIO_PATH,
            "model": MODEL,
            "device": DEVICE,
            "elapsed_s": round(elapsed, 2),
            "audio_duration_s": round(audio_duration, 2),
            "rt_factor": round(audio_duration / elapsed, 2) if elapsed > 0 else None,
            "num_speakers": len(per_speaker),
            "speakers": [
                {{"label": label, "total_speech_s": round(secs, 2)}}
                for label, secs in per_speaker_sorted
            ],
            "total_speech_s": round(total_speech, 2),
            "turns": turns,
        }}
        with open(json_path, "w") as fh:
            json.dump(payload, fh, indent=2)

        print(
            f"diarization complete: {{len(turns)}} turns across "
            f"{{len(per_speaker)}} speakers in {{elapsed:.1f}}s "
            f"(rt-factor {{payload['rt_factor']}}x).\\n"
            f"  RTTM: {{rttm_path}}\\n"
            f"  JSON: {{json_path}}"
        )
    """).strip()

    return script_body


def build_diarize_command(
    audio_basename: str,
    cfg: config.StackConfig,
    *,
    model: str,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    device: str,
    stem: str,
) -> str:
    """Bash one-liner that runs the diarize Python script under the audio-venv.

    Public so :func:`audio.transcribe` can compose it into the same
    runner script as whisper when the operator passes ``--diarize``
    (one tmux session does both, in sequence). Sets ``DIARIZE_*`` env
    vars instead of CLI args so paths with spaces / quotes don't have
    to be escaped twice. Python reads them on entry.
    """
    venv_py = f"{audio._venv_dir(cfg)}/bin/python"
    audio_path = f"{audio._in_dir(cfg)}/{audio_basename}"
    out_dir = f"{audio._out_dir(cfg)}/{stem}"

    py_script = _diarize_python_script(
        audio_path_bash=audio_path,
        out_dir_bash=out_dir,
        stem=stem,
        model=model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        device=device,
    )
    workdir_bash = audio._expand_tilde(cfg.workdir)

    # Source `.creds` from the head of this command so the bash payload
    # is self-sufficient: the token resolves whether it's set via
    # `--hf-token` (passed through ssh_run env), already in the parent
    # shell's environment, or persisted in `$workdir/.creds` from a
    # previous run. Then ensure the token is on disk for next time —
    # `ensure_var_block` is the late-arrival path that appends to an
    # existing `.creds` (e.g. one that already has NGC/NVIDIA keys from
    # a prior VSS install but no HF token yet). Idempotent: no-op when
    # the var is unset or already in the file.
    return textwrap.dedent(f"""\
        {_creds.source_block(workdir_bash)}
        {_creds.ensure_var_block(workdir_bash, "HUGGING_FACE_HUB_TOKEN")}
        export DIARIZE_AUDIO_PATH={audio._bash_dq(audio_path)}
        export DIARIZE_OUT_DIR={audio._bash_dq(out_dir)}
        export DIARIZE_STEM={shlex.quote(stem)}
        {audio._bash_dq(venv_py)} - <<'PY_EOF'
{py_script}
PY_EOF""")


# ---------------------------------------------------------------------------
# diarize (standalone) — parallel to audio.transcribe
# ---------------------------------------------------------------------------


def diarize(
    audio_local: Path,
    *,
    host: str | None,
    cfg: config.StackConfig,
    model: str = DEFAULT_DIARIZE_MODEL,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    hf_token: str | None = None,
    device_override: str | None = None,
    session_name: str | None = None,
    detach: bool | None = None,
    follow: bool | None = None,
    skip_upload: bool = False,
    auto_merge_whisper: bool = True,
) -> RunResult:
    """Upload (if needed) and run pyannote speaker diarization.

    Defaults mirror :func:`audio.transcribe`: ``detach`` auto-on for
    ``--target``, ``follow`` auto-on when detached. Device auto-detect
    via the audio-venv's torch (the same probe whisper uses).

    ``auto_merge_whisper`` (default ``True``): after a successful run,
    if a sibling ``<stem>.json`` from a prior whisper transcribe is
    found in ``$workdir/audio-out/<stem>/``, merge into
    ``<stem>.diarized.json`` and ``<stem>.diarized.txt``. Disable by
    passing ``False`` (no-op when whisper hasn't run yet).
    """
    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise ValueError("num_speakers is mutually exclusive with min_speakers / max_speakers")
    if (min_speakers is not None and max_speakers is not None
            and min_speakers > max_speakers):
        raise ValueError(
            f"min_speakers ({min_speakers}) > max_speakers ({max_speakers})")

    basename = audio_local.name
    stem = audio_local.stem
    if session_name is None:
        safe = "".join(c if c.isalnum() else "-" for c in stem)[:40]
        session_name = f"diar-{safe}"
    if detach is None:
        detach = host is not None
    if follow is None:
        follow = detach and host is not None

    setup = textwrap.dedent(f"""
        set -e
        mkdir -p {audio._in_dir(cfg)} {audio._out_dir(cfg)}/{shlex.quote(stem)} {audio._log_dir(cfg)}
    """).strip()
    r = ssh_run(host, setup) if host else run(["bash", "-c", setup])
    if not r.ok:
        return r

    if host and not skip_upload:
        size = audio_local.stat().st_size
        console.print(
            f"[bold]→[/bold] uploading {audio_local.name} "
            f"({size / 1e6:.1f} MB) to {host}:{audio._in_dir(cfg)}/"
        )
        import subprocess
        proc = subprocess.run(
            ["rsync", "-avh", "--progress",
             str(audio_local),
             f"{host}:{audio._in_dir(cfg)}/"],
        )
        if proc.returncode != 0:
            console.print(f"[red]upload failed (rc={proc.returncode})[/red]")
            return RunResult(rc=proc.returncode, stdout="", stderr="")

    if not host and not skip_upload:
        import shutil
        dest = Path(cfg.workdir).expanduser() / audio.AUDIO_IN_RELPATH / basename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.resolve() != audio_local.resolve():
            shutil.copy2(audio_local, dest)

    if device_override is not None:
        if device_override not in audio.VALID_DEVICES:
            raise ValueError(
                f"unknown device {device_override!r}; "
                f"choose from {audio.VALID_DEVICES}"
            )
        device = device_override
    else:
        device = audio._detect_device(cfg, host)
    console.print(
        f"  device: [cyan]{device}[/cyan]"
        + (" (auto-detected)" if device_override is None else " (forced via --device)")
    )

    cmd = build_diarize_command(
        audio_basename=basename,
        cfg=cfg,
        model=model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        device=device,
        stem=stem,
    )

    log_path = f"{audio._log_dir(cfg)}/{session_name}.log"
    workdir_bash = audio._expand_tilde(cfg.workdir)

    ssh_env: dict[str, str] = dict(cfg.env_block() or {})
    if hf_token:
        ssh_env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    creds_source = _creds.source_block(workdir_bash)

    if detach:
        runner_path = f"{audio._log_dir(cfg)}/{session_name}.sh"
        wrapper = textwrap.dedent(f"""
            set -e
            : > {log_path}
            if tmux has-session -t {shlex.quote(session_name)} 2>/dev/null; then
              echo "✗ tmux session {session_name} already running" >&2
              exit 1
            fi
            cat > {runner_path} <<'RUNNER_EOF'
#!/bin/bash
export PYTHONUNBUFFERED=1
{creds_source}
exec >> {log_path} 2>&1
echo "==== START $(date) ===="
set -o pipefail
{cmd}
RC=$?
echo "==== END rc=$RC $(date) ===="
RUNNER_EOF
            chmod +x {runner_path}
            tmux new-session -d -s {shlex.quote(session_name)} {runner_path}
            sleep 2
            echo "tmux: $(tmux list-sessions | grep {shlex.quote(session_name)})"
            echo "log : {log_path}"
            echo "out : {audio._out_dir(cfg)}/{stem}"
            tail -10 {log_path} 2>/dev/null
        """).strip()
    else:
        wrapper = textwrap.dedent(f"""
            {creds_source}
            echo "==== START $(date) ===="
            set -o pipefail
            {cmd}
            RC=$?
            echo "==== END rc=$RC $(date) ===="
            exit $RC
        """).strip()

    if host:
        r = ssh_run(host, wrapper, env=(ssh_env or None))
    else:
        import os
        local_env: dict[str, str] | None = None
        if hf_token:
            local_env = {**os.environ, "HUGGING_FACE_HUB_TOKEN": hf_token}
        if detach:
            # tmux dispatch — the bash wrapper is short (kicks off the tmux
            # session and prints a few status lines), buffering is fine.
            r = run(["bash", "-c", wrapper], env=local_env)
        else:
            # Foreground local — the bash wrapper runs the full pyannote
            # pipeline inline. Capturing stdout would buffer every progress
            # line (segmentation / embeddings / discrete_diarization) until
            # the run completes, making it look hung. Inherit the parent
            # stdio so the operator sees live progress as it streams.
            import subprocess
            proc = subprocess.run(
                ["bash", "-c", wrapper],
                env=local_env,
                stdout=None,
                stderr=None,
            )
            r = RunResult(rc=proc.returncode, stdout="", stderr="")

    if not r.ok:
        return r
    if r.stdout:
        # Foreground local path inherits stdio so r.stdout is empty — only
        # print when we actually have captured output (detach / host paths).
        console.print(r.stdout)

    if detach and follow and host is not None:
        console.print(
            f"\n[bold]→[/bold] live progress from [italic]{log_path}[/italic]\n"
            f"  (Ctrl-C to detach; the tmux session [cyan]{session_name}[/cyan] "
            f"will keep running on {host}.)\n"
        )
        try:
            ssh_stream(
                host,
                f"tail -n +1 -F {log_path} | "
                f"awk '{{print}} /==== END rc=/{{exit 0}}'",
            )
        except KeyboardInterrupt:
            console.print(
                f"\n[yellow]detached.[/yellow] tmux session [cyan]{session_name}[/cyan] "
                f"is still running on {host}.\n"
                f"  spectator audio status --target {host}\n"
                f"  ssh {host} 'tail -f {log_path}'"
            )

    if auto_merge_whisper and r.ok:
        _maybe_merge_remote_outputs(host, cfg, stem)

    return r


# ---------------------------------------------------------------------------
# merge: assign each whisper segment a speaker by max-overlap voting
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _MergeStats:
    segments_total: int
    segments_with_speaker: int
    segments_without_overlap: int
    speakers_seen: tuple[str, ...]


def merge_whisper_with_diarization(
    whisper_segments: list[dict],
    diarization_turns: list[dict],
) -> tuple[list[dict], _MergeStats]:
    """Assign each whisper segment a speaker label by maximum-overlap voting.

    For each whisper segment ``(seg.start, seg.end)``:

    1. Sum overlap durations against every diarization turn, grouped by
       turn label.
    2. Assign ``seg.speaker = argmax`` of those per-label sums.
    3. Deterministic tie-break: alphabetically-first speaker label wins
       (matters when two speakers contribute equal overlap, which pyannote
       can produce on perfectly bisected segments).
    4. ``seg.speaker = None`` when no diarization turn overlaps at all
       (e.g. whisper segment falls inside a non-speech region pyannote
       silently dropped).

    Returns ``(segments_with_speaker, stats)`` — the input list is not
    mutated; each output dict is a shallow copy of the input with a
    ``speaker`` key added.
    """
    out: list[dict] = []
    n_with_speaker = 0
    n_no_overlap = 0
    speakers_seen: set[str] = set()

    for seg in whisper_segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        per_label: dict[str, float] = {}
        for turn in diarization_turns:
            t_start = float(turn["start"])
            t_end = float(turn["end"])
            overlap = max(0.0, min(seg_end, t_end) - max(seg_start, t_start))
            if overlap > 0:
                label = turn["speaker"]
                per_label[label] = per_label.get(label, 0.0) + overlap
        if per_label:
            best_overlap = max(per_label.values())
            picked = sorted(label for label, dur in per_label.items()
                            if dur == best_overlap)[0]
            speaker: str | None = picked
            n_with_speaker += 1
            speakers_seen.add(picked)
        else:
            speaker = None
            n_no_overlap += 1
        merged = dict(seg)
        merged["speaker"] = speaker
        out.append(merged)

    stats = _MergeStats(
        segments_total=len(whisper_segments),
        segments_with_speaker=n_with_speaker,
        segments_without_overlap=n_no_overlap,
        speakers_seen=tuple(sorted(speakers_seen)),
    )
    return out, stats


def merged_segments_to_txt(merged: list[dict]) -> str:
    """Render merged segments as a human-readable block-per-speaker form.

    Consecutive segments with the same speaker collapse into one block;
    timestamps are rendered as ``H:MM:SS`` for readability. Mirrors the
    shape Teams's accessibility transcript uses, so the output is
    diff-friendly against a Teams capture of the same recording.
    """
    def _hms(s: float) -> str:
        h, rem = divmod(int(s), 3600)
        m, sec = divmod(rem, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

    lines: list[str] = []
    cur_speaker: str | None | object = object()
    cur_block: list[str] = []
    cur_start: float | None = None

    def _flush() -> None:
        if not cur_block or cur_start is None:
            return
        spk = cur_speaker if cur_speaker is not None else "(unknown)"
        lines.append(f"[{_hms(cur_start)}] {spk}:")
        for chunk in cur_block:
            lines.append(f"  {chunk}")
        lines.append("")

    for seg in merged:
        spk = seg.get("speaker")
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if spk != cur_speaker:
            _flush()
            cur_speaker = spk
            cur_block = []
            cur_start = float(seg["start"])
        cur_block.append(text)
    _flush()
    return "\n".join(lines)


def build_merge_command(cfg: config.StackConfig, stem: str) -> str:
    """Bash that produces ``<stem>.diarized.{json,txt}`` on target.

    If both ``<stem>.json`` (whisper) and ``<stem>.diar.json`` (pyannote)
    are present in ``$workdir/audio-out/<stem>/``, the merge runs and
    writes the speaker-attributed segments. If either is missing, the
    bash payload prints a friendly "merge skipped" line and exits 0 —
    safe to call unconditionally.

    Public so :func:`audio.transcribe` can append it after the
    diarize step inside the same runner script. Standalone
    :func:`diarize` calls it via :func:`_maybe_merge_remote_outputs`.
    """
    out_dir = f"{audio._out_dir(cfg)}/{stem}"
    venv_py = f"{audio._venv_dir(cfg)}/bin/python"

    # We inline a minimal merge implementation in the on-target Python
    # rather than importing from this module's :func:`merge_whisper_
    # with_diarization`. Rationale: the audio-venv on target has only
    # whisper + pyannote + their deps installed — it doesn't have
    # Spectator's source on PYTHONPATH. Inlining keeps the merge step
    # self-contained and survives even when the operator never ran
    # ``spectator deploy`` (e.g. local-only flows that copy the audio
    # via shutil instead of rsync). The algorithm is small enough
    # (~30 lines of Python) that the duplication cost is bounded.
    py_inline = textwrap.dedent("""
        import json
        import os
        import sys
        from pathlib import Path

        OUT_DIR = Path(os.environ["MERGE_OUT_DIR"])
        STEM = os.environ["MERGE_STEM"]
        whisper_path = OUT_DIR / f"{STEM}.json"
        diar_path = OUT_DIR / f"{STEM}.diar.json"
        if not whisper_path.is_file() or not diar_path.is_file():
            print(f"merge skipped: missing whisper or diarization output in {OUT_DIR}")
            sys.exit(0)
        with open(whisper_path) as fh:
            wj = json.load(fh)
        with open(diar_path) as fh:
            dj = json.load(fh)
        segments = wj.get("segments", [])
        turns = dj.get("turns", [])

        merged = []
        seen_speakers = set()
        n_with, n_without = 0, 0
        for seg in segments:
            s, e = float(seg["start"]), float(seg["end"])
            per = {}
            for t in turns:
                ov = max(0.0, min(e, float(t["end"])) - max(s, float(t["start"])))
                if ov > 0:
                    per[t["speaker"]] = per.get(t["speaker"], 0.0) + ov
            if per:
                best = max(per.values())
                spk = sorted(k for k, v in per.items() if v == best)[0]
                n_with += 1
                seen_speakers.add(spk)
            else:
                spk = None
                n_without += 1
            m = dict(seg)
            m["speaker"] = spk
            merged.append(m)

        out_json = OUT_DIR / f"{STEM}.diarized.json"
        with open(out_json, "w") as fh:
            json.dump({
                "schema_version": 1,
                "whisper_source": str(whisper_path),
                "diarization_source": str(diar_path),
                "segments_total": len(segments),
                "segments_with_speaker": n_with,
                "segments_without_overlap": n_without,
                "speakers_seen": sorted(seen_speakers),
                "segments": merged,
            }, fh, indent=2)

        # Plain-text grouped-by-speaker form for human reading.
        def hms(s):
            h, rem = divmod(int(s), 3600)
            m, sec = divmod(rem, 60)
            return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

        out_txt = OUT_DIR / f"{STEM}.diarized.txt"
        lines = []
        cur, block, start = object(), [], None
        for seg in merged:
            spk = seg.get("speaker") or "(unknown)"
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            if spk != cur:
                if block and start is not None:
                    lines.append(f"[{hms(start)}] {cur if cur is not None else '(unknown)'}:")
                    for c in block:
                        lines.append(f"  {c}")
                    lines.append("")
                cur, block, start = spk, [], float(seg["start"])
            block.append(text)
        if block and start is not None:
            lines.append(f"[{hms(start)}] {cur}:")
            for c in block:
                lines.append(f"  {c}")
            lines.append("")
        out_txt.write_text("\\n".join(lines))

        print(
            f"merged: {n_with}/{len(segments)} segments with speaker "
            f"({len(seen_speakers)} speakers seen); "
            f"{n_without} without overlap."
        )
        print(f"  JSON: {out_json}")
        print(f"  TXT : {out_txt}")
    """).strip()

    return textwrap.dedent(f"""\
        export MERGE_OUT_DIR={audio._bash_dq(out_dir)}
        export MERGE_STEM={shlex.quote(stem)}
        {audio._bash_dq(venv_py)} - <<'PY_EOF'
{py_inline}
PY_EOF""")


def _maybe_merge_remote_outputs(
    host: str | None,
    cfg: config.StackConfig,
    stem: str,
) -> None:
    """Best-effort merge: runs ``build_merge_command`` on target.

    Called automatically from the standalone :func:`diarize` after a
    successful run; the combined ``audio transcribe --diarize`` flow
    chains the merge command inline (no separate SSH round-trip).
    """
    workdir_bash = audio._expand_tilde(cfg.workdir)
    creds_source = _creds.source_block(workdir_bash)
    wrapper = textwrap.dedent(f"""
        {creds_source}
        {build_merge_command(cfg, stem)}
    """).strip()

    r = ssh_run(host, wrapper) if host else run(["bash", "-c", wrapper])
    if r.stdout:
        console.print(r.stdout)
    if not r.ok and r.stderr:
        console.print(f"[yellow]merge step warning[/yellow]\n{r.stderr}")


__all__ = [
    "DEFAULT_DIARIZE_MODEL",
    "HF_TOKEN_ENV_VARS",
    "PYANNOTE_PIN",
    "build_diarize_command",
    "build_merge_command",
    "diarize",
    "install_diarize_deps",
    "merge_whisper_with_diarization",
    "merged_segments_to_txt",
]
