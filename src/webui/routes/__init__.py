# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""HTTP + WebSocket routers for the Spectator Web UI.

Each router is a small, focused FastAPI APIRouter that's mounted by
``server.create_app``. Keeping them separate lets us unit-test them
in isolation without standing up the full app.
"""
