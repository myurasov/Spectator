# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Query routes: Q&A against indexed videos (VSS) and transcript text (audio).

Video Q&A proxies to VSS's OpenAI-compat /v1/chat/completions endpoint
(handled by :mod:`spectator.api`). Audio Q&A loads the transcript text
file, slices it to a token-budget-friendly chunk, and asks the same NIM
endpoint VSS uses (NVIDIA_API_KEY) for an answer with timestamp citations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import APIRouter, Body, HTTPException, Request

from ... import api, config

router = APIRouter(prefix="/api/query", tags=["query"])

# Coarse upper bound on transcript chars passed into the audio-QA prompt.
# Conservative — many small NIM models cap at ~32k tokens; ~80k chars is
# well within that. Truncate for longer transcripts and tell the user.
AUDIO_QA_MAX_CHARS = 80_000


@router.post("/video")
async def query_video(
    request: Request,
    body: Annotated[dict, Body()],
) -> dict:
    """Q&A against indexed videos via VSS's chat-completions API."""
    question = (body.get("question") or "").strip()
    file_ids = body.get("file_ids") or None
    if not question:
        raise HTTPException(status_code=400, detail="missing 'question'")

    target: str | None = request.app.state.target
    try:
        answer = api.query(question, host=target, file_ids=file_ids)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"VSS query failed: {exc}") from exc
    return {"answer": answer}


@router.post("/audio")
async def query_audio(
    request: Request,
    body: Annotated[dict, Body()],
) -> dict:
    """Q&A against an audio transcript using the same NIM endpoint VSS uses.

    Body: ``{"job_id": "...", "question": "..."}``. The job must be
    ``kind == "audio"`` and ``status == "completed"``; we read its
    ``<stem>.txt`` from the audio-out dir.
    """
    job_id = body.get("job_id")
    question = (body.get("question") or "").strip()
    if not job_id or not question:
        raise HTTPException(status_code=400, detail="need 'job_id' and 'question'")

    job = request.app.state.ledger.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
    if job.kind != "audio":
        raise HTTPException(status_code=400, detail="audio Q&A requires an audio job")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"job {job_id} is {job.status}, not completed",
        )

    stem = Path(job.input_path).stem
    txt_path = Path(job.output_dir) / f"{stem}.txt"
    if not txt_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"transcript text not found at {txt_path}",
        )

    transcript = txt_path.read_text(errors="replace")
    truncated = False
    if len(transcript) > AUDIO_QA_MAX_CHARS:
        transcript = transcript[:AUDIO_QA_MAX_CHARS]
        truncated = True

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="NVIDIA_API_KEY not set on the server's environment; "
                   "audio Q&A needs it to call the NIM endpoint",
        )

    endpoint = os.environ.get("LLM_ENDPOINT_URL", config.DEFAULT_LLM_ENDPOINT)
    payload = {
        "model": config.DEFAULT_REMOTE_LLM,
        "messages": [
            {"role": "system",
             "content": ("You are answering questions about a transcript "
                         "of a recorded meeting / call / interview. Cite "
                         "specific quotes from the transcript when relevant.")},
            {"role": "user",
             "content": (f"TRANSCRIPT:\n\n{transcript}\n\n"
                         f"QUESTION: {question}")},
        ],
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    try:
        r = httpx.post(
            f"{endpoint}/chat/completions",
            json=payload, headers=headers, timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        answer = data["choices"][0]["message"]["content"]
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"NIM query failed: {exc}",
        ) from exc

    return {
        "answer": answer,
        "model": config.DEFAULT_REMOTE_LLM,
        "transcript_chars": len(transcript),
        "truncated": truncated,
    }
