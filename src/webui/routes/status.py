# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""GET /api/status — overall server health snapshot for the UI dashboard."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import APIRouter, Request

from ... import __version__, config

router = APIRouter(prefix="/api", tags=["status"])


def _audio_venv_present(workdir: str) -> bool:
    """Probe ``$workdir/audio-venv/bin/python`` for the Whisper venv.

    We don't actually run torch.cuda.is_available() here — that's the job
    of Spectator's own ``audio.detect_device`` when a transcribe is
    submitted. This is just the cheapest "did install run?" check.
    """
    venv_py = Path(os.path.expanduser(workdir)) / "audio-venv" / "bin" / "python"
    return venv_py.is_file() and os.access(venv_py, os.X_OK)


def _vss_repo_present(workdir: str) -> bool:
    """Has ``spectator install`` cloned the VSS Blueprint?"""
    repo = Path(os.path.expanduser(workdir)) / config.DEFAULT_VSS_CHECKOUT
    return (repo / ".git").is_dir()


def _vss_api_reachable(target: str | None) -> bool:
    """Cheap probe of VSS Agent's /health endpoint (3-second timeout)."""
    base = (
        f"http://{target}:{config.AGENT_API_PORT}"
        if target and target not in ("localhost", "127.0.0.1")
        else f"http://localhost:{config.AGENT_API_PORT}"
    )
    try:
        r = httpx.get(f"{base}/health", timeout=3.0)
        return r.status_code < 500
    except Exception:
        return False


@router.get("/status")
async def get_status(request: Request) -> dict:
    workdir: str = request.app.state.workdir
    target: str | None = request.app.state.target
    ledger = request.app.state.ledger

    jobs_in_flight = sum(1 for j in ledger.list() if j.status in ("queued", "running"))

    return {
        "spectator_version": __version__,
        "target": target or "local",
        "workdir": workdir,
        "audio_venv_installed": _audio_venv_present(workdir) if not target else None,
        "vss_repo_cloned": _vss_repo_present(workdir) if not target else None,
        "vss_api_reachable": _vss_api_reachable(target),
        "jobs_in_flight": jobs_in_flight,
        "jobs_total": len(ledger.list()),
    }
