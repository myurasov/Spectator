# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Job lifecycle routes: submit / list / detail / kill / log / output download.

Submit endpoint accepts a multipart upload + form params and shells out to
``./spectator audio transcribe`` or ``./spectator process`` via
:mod:`spectator.webui.pipeline`. Everything else just reads the persistent
job ledger.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, PlainTextResponse

from ..jobs import Job
from .. import pipeline

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# How many bytes of the log to return on /log requests by default. Caller
# can override via ?lines=N (interpreted as line count, not bytes).
DEFAULT_LOG_TAIL_BYTES = 64 * 1024


@router.get("")
async def list_jobs(request: Request) -> dict:
    ledger = request.app.state.ledger
    return {"jobs": [j.to_dict() for j in ledger.list()]}


@router.post("")
async def submit_job(
    request: Request,
    kind: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    quality: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    task: Annotated[str, Form()] = "transcribe",
    model: Annotated[str | None, Form()] = None,
    device: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
) -> dict:
    if kind not in ("audio", "video"):
        raise HTTPException(status_code=400, detail=f"unknown kind {kind!r}")
    if task not in ("transcribe", "translate"):
        raise HTTPException(status_code=400, detail=f"unknown task {task!r}")

    ledger = request.app.state.ledger
    workdir: str = request.app.state.workdir
    target: str | None = request.app.state.target
    uploads_dir: Path = request.app.state.uploads_dir

    # Persist the upload to disk before we have a job id (we need the path
    # to record on the Job, and FastAPI's UploadFile is a SpooledTempFile
    # we can't keep around past request scope).
    if not file.filename:
        raise HTTPException(status_code=400, detail="upload missing filename")
    safe_name = Path(file.filename).name  # strip any dirname components
    dest = uploads_dir / safe_name
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    # Derive output paths.
    stem = dest.stem
    log_path = str(uploads_dir / f"{stem}.log")
    if kind == "audio":
        output_dir = str(Path(os.path.expanduser(workdir)) / "audio-out" / stem)
    else:
        output_dir = str(uploads_dir / f"{stem}.summary.json")

    job = Job.new(
        kind=kind,
        input_path=str(dest),
        output_dir=output_dir,
        params={
            "quality": quality, "language": language, "task": task,
            "model": model, "device": device, "prompt": prompt,
        },
        target=target,
        log_path=log_path,
    )
    ledger.put(job)

    if kind == "audio":
        pid = pipeline.spawn_audio_transcribe(
            audio_path=str(dest), log_path=log_path,
            target=target, workdir=workdir,
            quality=quality, language=language, task=task,
            model=model, device=device,
        )
    else:
        pid = pipeline.spawn_video_process(
            video_path=str(dest), log_path=log_path,
            target=target, prompt=prompt, output_path=output_dir,
        )
    job.pid = pid
    ledger.update_status(job.id, "running")

    return job.to_dict()


@router.get("/{job_id}")
async def get_job(request: Request, job_id: str) -> dict:
    job = request.app.state.ledger.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
    # Reflect-the-truth: if the PID died, transition to completed/failed.
    if job.status == "running" and job.pid and not pipeline.is_pid_alive(job.pid):
        request.app.state.ledger.update_status(job_id, "completed")
        job = request.app.state.ledger.get(job_id)  # type: ignore[assignment]
    return job.to_dict()  # type: ignore[union-attr]


@router.delete("/{job_id}")
async def cancel_job(request: Request, job_id: str) -> dict:
    ledger = request.app.state.ledger
    job = ledger.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
    killed = False
    if job.status in ("queued", "running"):
        if job.pid and pipeline.is_pid_alive(job.pid):
            killed = pipeline.kill_pid(job.pid)
        if job.target and job.tmux_session:
            killed = pipeline.kill_remote_tmux(job.target, job.tmux_session) or killed
        ledger.update_status(job_id, "cancelled")
    return {"ok": True, "killed": killed, "job": ledger.get(job_id).to_dict()}  # type: ignore[union-attr]


@router.get("/{job_id}/log")
async def get_log(request: Request, job_id: str,
                  bytes_: int = DEFAULT_LOG_TAIL_BYTES) -> PlainTextResponse:
    job = request.app.state.ledger.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
    p = Path(job.log_path)
    if not p.exists():
        return PlainTextResponse("", media_type="text/plain")
    size = p.stat().st_size
    with p.open("rb") as fh:
        if size > bytes_:
            fh.seek(size - bytes_)
        text = fh.read().decode("utf-8", errors="replace")
    return PlainTextResponse(text, media_type="text/plain")


@router.get("/{job_id}/output/{filename}")
async def download_output(request: Request, job_id: str, filename: str) -> FileResponse:
    job = request.app.state.ledger.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
    # Refuse path-traversal attempts.
    safe = Path(filename).name
    if safe != filename or safe.startswith("."):
        raise HTTPException(status_code=400, detail="invalid filename")
    target_path = Path(job.output_dir) / safe
    if not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"no output {safe!r}")
    return FileResponse(target_path, filename=safe)
