# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""FastAPI app factory for the Spectator Web UI.

Exposed surface:

  GET  /                           — single-page UI (static)
  GET  /api/status                 — server + VSS + audio-venv health
  POST /api/vss/up                 — bring VSS stack up
  POST /api/vss/down               — bring VSS stack down
  GET  /api/vss/status             — proxy to VSS API
  POST /api/audio-install          — idempotent `spectator audio install`
  GET  /api/jobs                   — list all jobs
  POST /api/jobs                   — submit (multipart upload + params)
  GET  /api/jobs/{id}              — job detail (incl. metrics)
  DELETE /api/jobs/{id}            — kill + remove
  GET  /api/jobs/{id}/log          — tail of the job's stdout/stderr log
  GET  /api/jobs/{id}/output/{f}   — download an output file
  WS   /api/jobs/{id}/progress     — live progress stream
  POST /api/query/video            — Q&A against indexed videos via VSS
  POST /api/query/audio            — Q&A against transcript text via NIM

The app is mounted by the CLI's ``ui-server start`` verb under uvicorn.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .jobs import JobLedger


def workdir_root(workdir: str) -> Path:
    """Resolve the WebUI's state root: ``$workdir/ui-server/``."""
    return Path(os.path.expanduser(workdir)) / "ui-server"


def static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def create_app(*, workdir: str, target: str | None) -> FastAPI:
    """Build a FastAPI app pre-configured with workdir + target.

    The factory pattern lets uvicorn launch the app fresh in each worker
    process while still threading per-instance config through.
    """
    root = workdir_root(workdir)
    jobs_dir = root / "jobs"
    uploads_dir = root / "uploads"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Recover jobs from disk, then yield. On shutdown we don't need to
        # do anything — the ledger writes through on every mutation.
        app.state.ledger = JobLedger(jobs_dir)
        app.state.workdir = workdir
        app.state.target = target
        app.state.uploads_dir = uploads_dir
        yield

    from .. import __version__ as _spectator_version

    app = FastAPI(
        title="Spectator Web UI",
        version=_spectator_version,
        lifespan=lifespan,
    )

    # Routers are imported lazily so test harnesses can pull the factory
    # without paying the import cost when they only want the helpers.
    from .routes import jobs as jobs_routes
    from .routes import query as query_routes
    from .routes import status as status_routes
    from .routes import vss as vss_routes
    from .routes import ws as ws_routes

    app.include_router(status_routes.router)
    app.include_router(vss_routes.router)
    app.include_router(jobs_routes.router)
    app.include_router(ws_routes.router)
    app.include_router(query_routes.router)

    # Static UI
    sd = static_dir()
    if sd.exists():
        app.mount("/static", StaticFiles(directory=str(sd)), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(sd / "index.html")

    return app


def server_state_paths(workdir: str) -> dict[str, Path]:
    """Canonical paths used by the CLI's ui-server start/stop/status verbs."""
    root = workdir_root(workdir)
    return {
        "root": root,
        "pid_file": root / "server.pid",
        "log_file": root / "server.log",
        "config_file": root / "server.json",
        "jobs_dir": root / "jobs",
        "uploads_dir": root / "uploads",
    }


def read_server_config(workdir: str) -> dict[str, Any] | None:
    """Load the persisted server config (bind, port, target, workdir) of
    a running server. Returns ``None`` when the config file is missing or
    unreadable — callers should treat that as "no info; skip the check".

    Pairs with :func:`write_server_config`. Used by ``ui-server start``
    to detect when a follow-up ``start`` would otherwise be a silent
    no-op (e.g. when the user passes ``--bind 0.0.0.0`` while a
    127.0.0.1-bound instance is already running)."""
    cf = server_state_paths(workdir)["config_file"]
    if not cf.exists():
        return None
    try:
        data = json.loads(cf.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_server_config(
    workdir: str, *, bind: str, port: int, target: str | None
) -> None:
    """Persist the running server's bind / port / target / workdir into
    ``$workdir/ui-server/server.json``.

    Written by ``ui-server start`` immediately after spawning uvicorn;
    deleted by ``ui-server stop``. The file is the canonical answer to
    "where is the running server actually bound?", which the
    user-supplied flag value is not (the flag may differ from a
    previously-started running server)."""
    cf = server_state_paths(workdir)["config_file"]
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(
        json.dumps(
            {
                "bind": bind,
                "port": port,
                "target": target or "",
                "workdir": workdir,
            }
        )
    )


__all__ = [
    "create_app",
    "workdir_root",
    "static_dir",
    "server_state_paths",
    "read_server_config",
    "write_server_config",
]
