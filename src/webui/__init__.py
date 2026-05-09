# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Persistent web UI for Spectator (added in v0.2.0).

A FastAPI service that wraps Spectator's CLI:

  * upload audio / video files
  * launch transcribe / process / query jobs
  * stream live progress (segments, rt-factor, ETA) via WebSocket
  * stop / start the VSS stack
  * download transcript / summary outputs
  * Q&A against indexed videos (and against transcripts for audio)

Run via the CLI:

    ./spectator ui-server start [--port 7777] [--target HOST] [--bind 127.0.0.1]
    ./spectator ui-server stop
    ./spectator ui-server status

The server is a long-lived uvicorn process detached from the launching shell;
its PID is tracked at ``$workdir/ui-server/server.pid`` and its log at
``$workdir/ui-server/server.log``.
"""

from __future__ import annotations

DEFAULT_PORT = 7777
DEFAULT_BIND = "127.0.0.1"
