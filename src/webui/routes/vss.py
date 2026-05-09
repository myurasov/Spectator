# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""POST /api/vss/up | down | status — VSS stack lifecycle controls.

Wraps ``stack.up`` / ``stack.down`` / ``stack.status`` from the Spectator
core. Long-running operations (``up`` first run is 30-45 minutes) return
immediately — the actual work happens in tmux on the target via Spectator's
existing tmux orchestration.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ... import config, stack

router = APIRouter(prefix="/api/vss", tags=["vss"])


def _resolve_cfg(workdir: str) -> config.StackConfig:
    return config.StackConfig.from_env(workdir=workdir)


@router.post("/up")
async def vss_up(request: Request) -> dict:
    cfg = _resolve_cfg(request.app.state.workdir)
    target: str | None = request.app.state.target
    r = stack.up(cfg, host=target)
    return {
        "ok": r.ok,
        "rc": r.rc,
        "stdout": r.stdout,
        "stderr": r.stderr,
        "note": "VSS bring-up runs in tmux on the target; first run is 30-45 min.",
    }


@router.post("/down")
async def vss_down(request: Request) -> dict:
    cfg = _resolve_cfg(request.app.state.workdir)
    target: str | None = request.app.state.target
    r = stack.down(cfg, host=target)
    return {"ok": r.ok, "rc": r.rc, "stdout": r.stdout, "stderr": r.stderr}


@router.get("/status")
async def vss_status(request: Request) -> dict:
    cfg = _resolve_cfg(request.app.state.workdir)
    target: str | None = request.app.state.target
    r = stack.status(cfg, host=target)
    if not r.ok:
        raise HTTPException(status_code=502, detail=r.stderr or r.stdout)
    return {"ok": True, "stdout": r.stdout}
