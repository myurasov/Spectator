# SPDX-FileCopyrightText: Copyright (c) 2026 Mikhail Yurasov
# SPDX-License-Identifier: Apache-2.0

"""Smoke + unit tests for the Web UI added in v0.2.0."""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# JobLedger persistence
# ---------------------------------------------------------------------------


def test_job_ledger_round_trips_through_disk(tmp_path: Path) -> None:
    from src.webui.jobs import Job, JobLedger

    ledger = JobLedger(tmp_path / "jobs")
    job = Job.new(
        kind="audio",
        input_path="/tmp/audio.mp3",
        output_dir="/tmp/out",
        params={"quality": "meeting", "language": None},
        target=None,
        log_path="/tmp/log.txt",
    )
    ledger.put(job)
    assert ledger.get(job.id) is not None

    # Re-open from disk; ledger should rehydrate.
    ledger2 = JobLedger(tmp_path / "jobs")
    rehydrated = ledger2.get(job.id)
    assert rehydrated is not None
    assert rehydrated.kind == "audio"
    assert rehydrated.params["quality"] == "meeting"
    assert rehydrated.status == "queued"


def test_job_ledger_status_transitions_set_timestamps(tmp_path: Path) -> None:
    from src.webui.jobs import Job, JobLedger

    ledger = JobLedger(tmp_path / "jobs")
    job = Job.new(kind="audio", input_path="/tmp/a", output_dir="/tmp/o",
                  params={}, target=None, log_path="/tmp/l")
    ledger.put(job)

    ledger.update_status(job.id, "running")
    assert ledger.get(job.id).started_at is not None
    assert ledger.get(job.id).completed_at is None

    ledger.update_status(job.id, "completed", metrics={"rt_factor": 3.4})
    fresh = ledger.get(job.id)
    assert fresh.completed_at is not None
    assert fresh.metrics["rt_factor"] == 3.4

    ledger.delete(job.id)
    assert ledger.get(job.id) is None
    assert not (tmp_path / "jobs" / f"{job.id}.json").exists()


def test_job_kind_must_be_audio_or_video() -> None:
    from src.webui.jobs import Job

    with pytest.raises(ValueError):
        Job.new(kind="image", input_path="/x", output_dir="/y", params={},
                target=None, log_path="/z")


# ---------------------------------------------------------------------------
# Progress parser
# ---------------------------------------------------------------------------


_SAMPLE_LOG_RUNNING = """\
==== START Sat May  9 16:30:01 PDT 2026 ====
device: mps
[00:00.000 --> 00:08.420]  Welcome everyone to the architecture review.
[00:08.420 --> 00:12.500]  We'll start with the diagrams.
[00:12.500 --> 00:18.330]  Specifically the cosmos integration with...
"""

_SAMPLE_LOG_DONE = _SAMPLE_LOG_RUNNING + """\
[00:18.330 --> 00:24.000]  Final thoughts before we wrap.
==== END rc=0 Sat May  9 16:33:42 PDT 2026 ====
"""


def test_progress_parser_running_segments() -> None:
    from src.webui.progress import parse_log

    snap = parse_log(_SAMPLE_LOG_RUNNING, audio_duration_s=60.0, wall_clock_s=10.0)
    assert snap.processed_s == pytest.approx(18.33, abs=0.01)
    assert 0.0 < snap.rt_factor < 5.0
    assert snap.percent == pytest.approx(18.33 / 60.0 * 100, abs=0.1)
    assert snap.eta_s is not None
    assert snap.device == "mps"
    assert not snap.finished
    assert snap.exit_code is None


def test_progress_parser_finished_run() -> None:
    from src.webui.progress import parse_log

    snap = parse_log(_SAMPLE_LOG_DONE, audio_duration_s=24.0, wall_clock_s=222.0)
    assert snap.finished is True
    assert snap.exit_code == 0
    assert snap.processed_s == pytest.approx(24.0, abs=0.1)
    assert snap.percent == pytest.approx(100.0, abs=0.1)


def test_progress_parser_handles_unknown_duration() -> None:
    from src.webui.progress import parse_log

    snap = parse_log(_SAMPLE_LOG_RUNNING, audio_duration_s=None, wall_clock_s=10.0)
    assert snap.percent == 0.0
    assert snap.eta_s is None
    # rt-factor is still meaningful (processed audio / wall-clock)
    assert snap.rt_factor > 0


def test_progress_parser_long_form_timestamp() -> None:
    """Whisper switches to ``HH:MM:SS.mmm`` format for runs longer than ~1h."""
    from src.webui.progress import parse_log

    log = (
        "device: cuda\n"
        "[01:23:45.123 --> 01:24:01.456]  some long-form text\n"
    )
    snap = parse_log(log, audio_duration_s=2 * 3600, wall_clock_s=300.0)
    expected_end = 1 * 3600 + 24 * 60 + 1 + 0.456
    assert snap.processed_s == pytest.approx(expected_end, abs=0.01)


def test_format_duration_handles_all_ranges() -> None:
    from src.webui.progress import format_duration

    assert format_duration(None) == "—"
    assert format_duration(7) == "7s"
    assert format_duration(95) == "1m 35s"
    assert format_duration(3725) == "1h 02m 05s"


# ---------------------------------------------------------------------------
# FastAPI app — TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def app(tmp_path):
    from src.webui.server import create_app

    return create_app(workdir=str(tmp_path), target=None)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def test_status_endpoint_responds(client) -> None:
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert "spectator_version" in body
    assert "jobs_in_flight" in body
    assert "jobs_total" in body
    assert body["target"] == "local"


def test_jobs_list_starts_empty(client) -> None:
    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert r.json()["jobs"] == []


def test_unknown_job_returns_404(client) -> None:
    r = client.get("/api/jobs/this-id-does-not-exist")
    assert r.status_code == 404


def test_jobs_submit_audio_records_a_job(client, tmp_path, monkeypatch) -> None:
    """Submit an audio job; we mock spawn so no actual whisper runs."""
    from src.webui import pipeline as pipeline_mod
    from src.webui.routes import jobs as jobs_routes

    monkeypatch.setattr(jobs_routes.pipeline, "spawn_audio_transcribe",
                        lambda **kw: 99999)

    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"\x00" * 1024)
    with audio.open("rb") as fh:
        r = client.post(
            "/api/jobs",
            data={"kind": "audio", "quality": "meeting", "task": "transcribe"},
            files={"file": ("sample.mp3", fh, "audio/mpeg")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "audio"
    assert body["status"] == "running"
    assert body["pid"] == 99999
    assert body["params"]["quality"] == "meeting"

    # Listing should include it.
    r2 = client.get("/api/jobs")
    ids = [j["id"] for j in r2.json()["jobs"]]
    assert body["id"] in ids


def test_invalid_kind_rejected(client, tmp_path) -> None:
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"\x00")
    with audio.open("rb") as fh:
        r = client.post("/api/jobs",
                        data={"kind": "image"},
                        files={"file": ("x.mp3", fh, "audio/mpeg")})
    assert r.status_code == 400


def test_path_traversal_blocked_on_output_download(client, tmp_path,
                                                    monkeypatch) -> None:
    """We refuse `/api/jobs/{id}/output/../foo` -style filenames."""
    from src.webui.jobs import Job
    ledger = client.app.state.ledger

    job = Job.new(kind="audio", input_path="/tmp/a.mp3", output_dir="/tmp/out",
                  params={}, target=None, log_path="/tmp/l.log")
    ledger.put(job)

    r = client.get(f"/api/jobs/{job.id}/output/..%2Fetc%2Fpasswd")
    # ".." → not a single name; route returns 400.
    assert r.status_code in (400, 404)


# ---------------------------------------------------------------------------
# server.json persistence (v0.3.1) — backs the start-conflict detection so
# `ui-server start --bind 0.0.0.0` while a 127.0.0.1-bound instance is
# already running surfaces a clear error instead of a silent no-op.
# ---------------------------------------------------------------------------


def test_state_paths_exposes_config_file(tmp_path: Path) -> None:
    from src.webui.server import server_state_paths

    paths = server_state_paths(str(tmp_path))
    assert "config_file" in paths
    assert paths["config_file"].name == "server.json"
    assert paths["config_file"].parent == paths["root"]


def test_read_server_config_returns_none_when_absent(tmp_path: Path) -> None:
    from src.webui.server import read_server_config

    assert read_server_config(str(tmp_path)) is None


def test_server_config_round_trips(tmp_path: Path) -> None:
    from src.webui.server import (
        read_server_config,
        server_state_paths,
        write_server_config,
    )

    write_server_config(
        str(tmp_path), bind="0.0.0.0", port=7777, target="myspark1-local"
    )
    cf = server_state_paths(str(tmp_path))["config_file"]
    assert cf.is_file()

    cfg = read_server_config(str(tmp_path))
    assert cfg == {
        "bind": "0.0.0.0",
        "port": 7777,
        "target": "myspark1-local",
        "workdir": str(tmp_path),
    }


def test_server_config_target_normalized_to_empty_string_when_none(
    tmp_path: Path,
) -> None:
    """A None target serializes as "" so the file stays type-stable for
    the conflict-detection comparison in `ui-server start`."""
    from src.webui.server import read_server_config, write_server_config

    write_server_config(str(tmp_path), bind="127.0.0.1", port=7777, target=None)
    cfg = read_server_config(str(tmp_path))
    assert cfg is not None
    assert cfg["target"] == ""


def test_read_server_config_returns_none_on_corrupt_json(tmp_path: Path) -> None:
    """If server.json is unreadable / malformed, callers should get None
    rather than an exception — the config file is advisory, not load-bearing."""
    from src.webui.server import read_server_config, server_state_paths

    cf = server_state_paths(str(tmp_path))["config_file"]
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text("{not valid json")

    assert read_server_config(str(tmp_path)) is None


def test_read_server_config_returns_none_on_non_dict_payload(
    tmp_path: Path,
) -> None:
    """A JSON payload that's syntactically valid but not an object (e.g.
    a list) is treated like a missing file — defensive, in case a
    future schema change writes a different top-level shape."""
    from src.webui.server import read_server_config, server_state_paths

    cf = server_state_paths(str(tmp_path))["config_file"]
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text("[1, 2, 3]")

    assert read_server_config(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Legacy-workdir scan (v0.3.3) — when the user upgrades across a default-
# workdir change (v0.2.x ~/spectator -> v0.3.0 ~/.spectator-workdir ->
# v0.3.2 ~/.spectator), `ui-server start` should surface any Web UI server
# still running under the prior default rather than silently spawning a
# second one (which would either lose the port or run with a stale workdir).
# ---------------------------------------------------------------------------


def test_legacy_workdirs_constant_includes_known_prior_defaults() -> None:
    """The constant is the public contract for the scan — pin both names
    so a careless edit doesn't drop one and break the upgrade UX."""
    from src.webui.server import LEGACY_WORKDIRS

    assert "~/spectator" in LEGACY_WORKDIRS
    assert "~/.spectator-workdir" in LEGACY_WORKDIRS


def test_find_legacy_running_server_returns_none_on_clean_state(
    tmp_path: Path,
) -> None:
    """No legacy state-dirs anywhere -> None. Sanity check before the
    positive cases: this is the path most users hit."""
    from src.webui import server as ui_server

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    import os

    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(fake_home)
    try:
        assert ui_server.find_legacy_running_server(str(tmp_path / "current")) is None
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home


def test_find_legacy_running_server_skips_dead_pid(
    tmp_path: Path, monkeypatch
) -> None:
    """A pid_file pointing at a long-dead PID -> ignored. The scan must
    not mistake stale state for a running server."""
    from src.webui import server as ui_server

    fake_home = tmp_path / "home"
    legacy_root = fake_home / ".spectator-workdir" / "ui-server"
    legacy_root.mkdir(parents=True)
    (legacy_root / "server.pid").write_text("999999")  # almost certainly dead

    monkeypatch.setenv("HOME", str(fake_home))
    assert ui_server.find_legacy_running_server(str(tmp_path / "current")) is None


def test_find_legacy_running_server_finds_live_pid(
    tmp_path: Path, monkeypatch
) -> None:
    """A pid_file pointing at this very test process (definitely alive) plus
    a server.json -> the helper returns the recorded config."""
    import os

    from src.webui import server as ui_server

    fake_home = tmp_path / "home"
    legacy_root = fake_home / ".spectator-workdir" / "ui-server"
    legacy_root.mkdir(parents=True)
    (legacy_root / "server.pid").write_text(str(os.getpid()))
    (legacy_root / "server.json").write_text(
        '{"bind": "127.0.0.1", "port": 7777, '
        '"target": "myspark1-local", "workdir": "~/.spectator-workdir"}'
    )

    monkeypatch.setenv("HOME", str(fake_home))
    hit = ui_server.find_legacy_running_server(str(tmp_path / "current"))
    assert hit is not None
    assert hit["workdir"] == "~/.spectator-workdir"
    assert hit["pid"] == os.getpid()
    assert hit["bind"] == "127.0.0.1"
    assert hit["port"] == 7777
    assert hit["target"] == "myspark1-local"


def test_find_legacy_running_server_skips_when_legacy_path_is_current(
    tmp_path: Path, monkeypatch
) -> None:
    """If the user is *intentionally* running on a legacy path (e.g. they
    explicitly passed `--workdir ~/.spectator-workdir`), the helper
    should return None for that path — it's the current target, not a
    legacy leftover."""
    import os

    from src.webui import server as ui_server

    fake_home = tmp_path / "home"
    legacy_root = fake_home / ".spectator-workdir" / "ui-server"
    legacy_root.mkdir(parents=True)
    (legacy_root / "server.pid").write_text(str(os.getpid()))

    monkeypatch.setenv("HOME", str(fake_home))
    # Pass the same path as `current_workdir` — should be skipped.
    assert (
        ui_server.find_legacy_running_server("~/.spectator-workdir") is None
    )
