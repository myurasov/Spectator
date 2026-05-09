# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Uvicorn-importable entry point.

The CLI's ``ui-server start`` verb spawns a detached
``python -m uvicorn src.webui._launch:app`` process. uvicorn loads the
``app`` attribute from this module by import-string, so the parent
must communicate config (workdir / target) via environment variables.

Env vars consumed:

  * SPECTATOR_UI_WORKDIR  (required) — path to Spectator's $workdir
  * SPECTATOR_UI_TARGET   (optional) — SSH alias; empty string = local
"""

from __future__ import annotations

import os

from .server import create_app

_workdir = os.environ.get("SPECTATOR_UI_WORKDIR") or "~/spectator"
_target_raw = os.environ.get("SPECTATOR_UI_TARGET", "")
_target: str | None = _target_raw if _target_raw else None

app = create_app(workdir=_workdir, target=_target)
