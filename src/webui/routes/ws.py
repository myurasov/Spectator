# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""WS /api/jobs/{id}/progress — live progress stream.

Polls the job's stdout/stderr log every 2 s, parses whisper segments via
:mod:`spectator.webui.progress`, and pushes a JSON snapshot per tick. The
client is responsible for closing the WebSocket; we close server-side once
the END line appears.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import pipeline, progress

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

POLL_INTERVAL_S = 2.0


@router.websocket("/{job_id}/progress")
async def job_progress_ws(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    ledger = ws.app.state.ledger
    job = ledger.get(job_id)
    if job is None:
        await ws.send_json({"error": "no_such_job", "job_id": job_id})
        await ws.close(code=4404)
        return

    log_path = Path(job.log_path)
    audio_duration_s: float | None = None
    if job.kind == "audio":
        audio_duration_s = progress.probe_duration(Path(job.input_path))

    try:
        while True:
            text = log_path.read_text(errors="replace") if log_path.exists() else ""
            wall_clock = time.time() - (job.started_at or job.created_at)
            snap = progress.parse_log(
                text,
                audio_duration_s=audio_duration_s,
                wall_clock_s=wall_clock,
            )
            payload = {
                "job_id": job_id,
                "kind": job.kind,
                "status": job.status,
                "snapshot": snap.to_dict(),
                "audio_duration_human": progress.format_duration(snap.audio_duration_s),
                "processed_human": progress.format_duration(snap.processed_s),
                "wall_clock_human": progress.format_duration(snap.wall_clock_s),
                "eta_human": progress.format_duration(snap.eta_s),
            }
            await ws.send_json(payload)

            # Reflect-the-truth: keep the ledger consistent for any HTTP
            # readers polling /api/jobs/{id} alongside this WS.
            if snap.finished:
                if snap.exit_code == 0:
                    ledger.update_status(
                        job_id, "completed",
                        metrics={
                            "duration_s": snap.audio_duration_s,
                            "wall_clock_s": snap.wall_clock_s,
                            "rt_factor": snap.rt_factor,
                            "device": snap.device,
                        },
                    )
                else:
                    ledger.update_status(
                        job_id, "failed",
                        error=f"non-zero exit code {snap.exit_code}",
                    )
                break

            # If the subprocess died but we haven't seen an END line yet
            # (crash, kill, etc.), close cleanly so the client can poll.
            if job.pid and not pipeline.is_pid_alive(job.pid) and not snap.finished:
                ledger.update_status(
                    job_id, "failed",
                    error="subprocess died without END marker",
                )
                break

            await asyncio.sleep(POLL_INTERVAL_S)

        # Send one final snapshot reflecting the terminal status.
        job = ledger.get(job_id)
        if job is not None:
            await ws.send_json({"job_id": job_id, "final": job.to_dict()})
    except WebSocketDisconnect:
        # Client closed first — that's fine, we just stop polling.
        return
    finally:
        try:
            await ws.close()
        except RuntimeError:
            # Already closed; ignore.
            pass
