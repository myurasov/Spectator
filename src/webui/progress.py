# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Progress parsing for live job updates.

Audio jobs emit whisper segment lines of the form
``[MM:SS.mmm --> MM:SS.mmm] <text>``; this module turns those plus a known
audio duration into a percentage / wall-clock / rt-factor / ETA snapshot.

Video jobs (VSS) don't emit segment-level progress on stdout, so for those
we report the coarser status returned by VSS's ``/jobs/<id>`` endpoint.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

# ``[02:13.000 --> 02:48.000]  text``  or
# ``[01:23:45.123 --> 01:24:01.456]  text`` (h:mm:ss for very long files)
_SEGMENT_RE = re.compile(
    r"\[(?:(\d+):)?(\d{1,2}):(\d{2})\.(\d{3})\s*-->\s*"
    r"(?:(\d+):)?(\d{1,2}):(\d{2})\.(\d{3})\]"
)
_END_RE = re.compile(r"==== END rc=(-?\d+) ")
_DEVICE_HINT_RE = re.compile(r"^\s*device:\s*(\S+)", re.MULTILINE)


@dataclass(slots=True)
class ProgressSnapshot:
    """A point-in-time view of a job's progress."""

    audio_duration_s: float | None  # ffprobe-derived, None if unknown
    processed_s: float  # high-water mark of the latest segment END timestamp
    wall_clock_s: float  # elapsed since job.started_at
    rt_factor: float  # processed_s / max(wall_clock_s, eps); 0 until first segment
    percent: float  # 0..100; processed_s / audio_duration_s * 100
    eta_s: float | None  # remaining audio / current rt-factor; None if unknown
    device: str | None  # cuda / mps / cpu, parsed from the log
    finished: bool  # END line seen
    exit_code: int | None  # parsed from END line; None if not finished

    def to_dict(self) -> dict:
        return asdict(self)


def probe_duration(audio_path: Path) -> float | None:
    """Return audio duration in seconds via ffprobe, or None if unavailable.

    ffprobe is bundled with ffmpeg, which Spectator's audio-venv installs as
    a torch+whisper dep. If neither is on PATH, return None and let the UI
    show "duration unknown" rather than blocking.
    """
    if shutil.which("ffprobe") is None:
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(audio_path)],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return None
        return float(r.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        return None


def _parse_timestamp(h: str | None, m: str, s: str, ms: str) -> float:
    hours = int(h) if h else 0
    return hours * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_log(text: str, *, audio_duration_s: float | None,
              wall_clock_s: float) -> ProgressSnapshot:
    """Build a snapshot from the current contents of the job's log file.

    Tolerates partially-written final lines (whisper streams stdout).
    Always returns *something* — if no segments have been emitted yet, the
    snapshot reports zero processed-seconds and zero rt-factor.
    """
    processed_s = 0.0
    for m in _SEGMENT_RE.finditer(text):
        # we want the END of the latest segment seen, which is groups 5-8.
        end_s = _parse_timestamp(m.group(5), m.group(6), m.group(7), m.group(8))
        if end_s > processed_s:
            processed_s = end_s

    end_match = _END_RE.search(text)
    finished = end_match is not None
    exit_code = int(end_match.group(1)) if end_match else None

    device_match = _DEVICE_HINT_RE.search(text)
    device = device_match.group(1) if device_match else None

    rt_factor = processed_s / wall_clock_s if wall_clock_s > 0 else 0.0
    if audio_duration_s and audio_duration_s > 0:
        percent = min(100.0, processed_s / audio_duration_s * 100.0)
        if rt_factor > 0:
            remaining_audio = max(0.0, audio_duration_s - processed_s)
            eta_s: float | None = remaining_audio / rt_factor
        else:
            eta_s = None
    else:
        percent = 0.0
        eta_s = None

    return ProgressSnapshot(
        audio_duration_s=audio_duration_s,
        processed_s=processed_s,
        wall_clock_s=wall_clock_s,
        rt_factor=rt_factor,
        percent=percent,
        eta_s=eta_s,
        device=device,
        finished=finished,
        exit_code=exit_code,
    )


def format_duration(seconds: float | None) -> str:
    """Format seconds as ``Hh MMm SSs`` (or ``MMm SSs`` / ``SSs`` for short)."""
    if seconds is None or seconds < 0:
        return "—"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
