# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Thin client for the running VSS Agent (REST + OpenAI-compat chat).

The VSS Agent UI lives on port 3000; the Agent's HTTP API on port 8000
(default). VSS exposes:
  POST /files                        — multipart upload, returns a file id
  POST /summarize                    — summarize a previously uploaded file id
  POST /v1/chat/completions          — OpenAI-compatible chat (Q&A)

This module is best-effort: if the upstream API surface drifts in a
future v3.x, edit the three URLs below — the rest of spectator stays
the same.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
from rich.console import Console

from . import config

console = Console()


def _base(host: str | None) -> str:
    """Resolve the API base URL.

    - host is None / "localhost" → assume we're on the Spark itself.
    - Otherwise use the SSH host alias (we'll port-forward via the
      caller's `ssh -L 3000:localhost:3000 -L 8000:localhost:8000 host`
      session, or this command runs over ssh on the remote).
    """
    if host in (None, "localhost", "127.0.0.1"):
        return f"http://localhost:{config.AGENT_API_PORT}"
    return f"http://{host}:{config.AGENT_API_PORT}"


def health(host: str | None = None, timeout: float = 5.0) -> bool:
    try:
        r = httpx.get(f"{_base(host)}/health", timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


def upload(video: Path, host: str | None = None) -> str:
    """Upload a video; return the VSS file id."""
    url = f"{_base(host)}/files"
    with open(video, "rb") as f:
        r = httpx.post(
            url,
            files={"file": (video.name, f, "video/mp4")},
            timeout=httpx.Timeout(connect=10, read=600, write=600, pool=600),
        )
    r.raise_for_status()
    data = r.json()
    fid = data.get("id") or data.get("file_id") or data.get("data", {}).get("id")
    if not fid:
        raise RuntimeError(f"upload OK but no file id in response: {data!r}")
    return fid


def summarize(file_id: str, *, host: str | None = None,
              prompt: str | None = None, max_wait_s: int = 1800) -> dict:
    """Run the LVS (long-video summarization) agent on `file_id`.

    Polls until the job finishes or `max_wait_s` elapses. Returns the
    raw result JSON.
    """
    url = f"{_base(host)}/summarize"
    payload: dict = {"file_id": file_id}
    if prompt:
        payload["prompt"] = prompt
    r = httpx.post(url, json=payload, timeout=30)
    r.raise_for_status()
    body = r.json()

    job_id = body.get("job_id") or body.get("id")
    if not job_id:
        # Some configurations return the result inline.
        return body

    # poll
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        rr = httpx.get(f"{_base(host)}/jobs/{job_id}", timeout=15)
        rr.raise_for_status()
        j = rr.json()
        st = (j.get("status") or "").lower()
        if st in ("succeeded", "completed", "done"):
            return j
        if st in ("failed", "error"):
            raise RuntimeError(f"summarize job {job_id} failed: {j}")
        time.sleep(5)
    raise TimeoutError(f"summarize job {job_id} did not finish in {max_wait_s}s")


def query(question: str, *, host: str | None = None,
          file_ids: list[str] | None = None,
          model: str = "vss") -> str:
    """Q&A against the indexed video corpus via OpenAI-compat chat."""
    url = f"{_base(host)}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You are a video-understanding agent. Answer the user's question using the indexed video content. Cite timestamps."},
            {"role": "user", "content": question},
        ],
    }
    if file_ids:
        payload["file_ids"] = file_ids
    r = httpx.post(url, json=payload, timeout=120)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]
