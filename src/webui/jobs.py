# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Persistent job ledger for the Web UI.

Each job is a JSON file at ``$workdir/ui-server/jobs/<uuid>.json`` plus a
sidecar ``<uuid>.log`` capturing the spawned subprocess's stdout/stderr.
The ledger is loaded into memory on server startup and refreshed on every
mutation; readers (status / list / detail) can sample the in-memory state
without touching disk.

The JSON shape is the public contract for any agent that wants to follow
along outside the WebUI — see ``Job.to_dict()`` for the schema.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

JOB_KINDS = ("audio", "video")
JOB_STATES = ("queued", "running", "completed", "failed", "cancelled")


@dataclass(slots=True)
class Job:
    """One unit of work submitted via the Web UI.

    All time fields are unix seconds (float); convert at the edge for display.
    """

    id: str
    kind: str  # one of JOB_KINDS
    status: str  # one of JOB_STATES
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    input_path: str = ""  # local path on the machine running spectator
    output_dir: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    pid: int | None = None  # local subprocess pid, when running locally
    tmux_session: str | None = None  # remote tmux session name, when --target is set
    target: str | None = None  # SSH alias, None for local
    log_path: str = ""
    error: str | None = None

    @classmethod
    def new(cls, kind: str, *, input_path: str, output_dir: str,
            params: dict[str, Any], target: str | None,
            log_path: str) -> "Job":
        if kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind {kind!r}; must be one of {JOB_KINDS}")
        return cls(
            id=str(uuid.uuid4()),
            kind=kind,
            status="queued",
            created_at=time.time(),
            input_path=input_path,
            output_dir=output_dir,
            params=params,
            target=target,
            log_path=log_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        # tolerate forward-compatible additions: ignore unknown keys.
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class JobLedger:
    """Persistent on-disk + in-memory job tracker.

    All writes go through ``put()`` which atomically rewrites the JSON file
    (temp + rename). All reads are served from the in-memory dict.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._load()

    def _load(self) -> None:
        for p in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                job = Job.from_dict(data)
                self._jobs[job.id] = job
            except (json.JSONDecodeError, TypeError, ValueError):
                # skip malformed entries; user can clean them up manually.
                continue

    def _path_for(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def put(self, job: Job) -> None:
        self._jobs[job.id] = job
        path = self._path_for(job.id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job.to_dict(), indent=2, sort_keys=True))
        tmp.replace(path)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def delete(self, job_id: str) -> bool:
        """Remove a job from the ledger. Does NOT touch the log file or output dir."""
        if job_id not in self._jobs:
            return False
        del self._jobs[job_id]
        path = self._path_for(job_id)
        if path.exists():
            path.unlink()
        return True

    def update_status(self, job_id: str, status: str, *,
                      error: str | None = None,
                      metrics: dict[str, Any] | None = None) -> Job | None:
        if status not in JOB_STATES:
            raise ValueError(f"unknown status {status!r}; must be one of {JOB_STATES}")
        job = self._jobs.get(job_id)
        if job is None:
            return None
        job.status = status
        if status == "running" and job.started_at is None:
            job.started_at = time.time()
        if status in ("completed", "failed", "cancelled") and job.completed_at is None:
            job.completed_at = time.time()
        if error is not None:
            job.error = error
        if metrics is not None:
            job.metrics.update(metrics)
        self.put(job)
        return job
