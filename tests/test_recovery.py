from __future__ import annotations

import threading
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from fakes import FakeBackend
from soriham_stt.server import create_app
from test_api import wait_done


def test_repeated_request_reuses_completed_job(settings):
    backend = FakeBackend()
    request_id = str(uuid4())
    with TestClient(create_app(settings, lambda: backend)) as client:
        first = client.post(
            "/jobs", data={"request_id": request_id}, files={"file": ("a.wav", b"x")}
        )
        job_id = first.json()["job_id"]
        wait_done(client, job_id)
        again = client.post(
            "/jobs", data={"request_id": request_id}, files={"file": ("a.wav", b"x")}
        )
        assert again.json()["job_id"] == job_id
        assert len(backend.calls) == 1


def test_dead_worker_rejects_new_jobs_and_marks_active_error(settings, monkeypatch):
    import soriham_stt.server as server

    entered, release = threading.Event(), threading.Event()

    def crash(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        raise SystemExit("injected thread exit")

    monkeypatch.setattr(server, "run_job", crash)
    with TestClient(create_app(settings, FakeBackend)) as client:
        job_id = client.post("/jobs", files={"file": ("a.wav", b"x")}).json()["job_id"]
        assert entered.wait(2)
        queued = client.post("/jobs", files={"file": ("b.wav", b"x")}).json()["job_id"]
        release.set()
        body = wait_done(client, job_id)
        assert body["status"] == "error"
        assert client.get("/health").status_code == 503
        assert client.get(f"/jobs/{queued}").json()["status"] == "error"
        assert client.post("/jobs", files={"file": ("b.wav", b"x")}).status_code == 503
    assert not list(settings.work_dir.iterdir())


def test_stalled_worker_fails_queue_and_discards_late_result(settings, monkeypatch):
    import soriham_stt.jobs as jobs
    import soriham_stt.server as server
    from soriham_stt.schemas import JobResult

    entered, release = threading.Event(), threading.Event()
    clock = [100.0]
    monkeypatch.setattr(jobs.time, "monotonic", lambda: clock[0])

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        kwargs["on_progress"]("transcribe", 1.0)
        return JobResult(language="ko", segments=[])

    monkeypatch.setattr(server, "run_job", blocked)
    with TestClient(create_app(settings, FakeBackend)) as client:
        try:
            active = client.post(
                "/jobs", data={"timeout_sec": "10"}, files={"file": ("a.wav", b"x")}
            ).json()["job_id"]
            assert entered.wait(2)
            queued = client.post("/jobs", files={"file": ("b.wav", b"x")}).json()["job_id"]
            assert client.post("/jobs", files={"file": ("c.wav", b"x")}).status_code == 503
            clock[0] += 11
            assert client.get("/health").status_code == 503
            for job_id in (active, queued):
                assert client.get(f"/jobs/{job_id}").json()["status"] == "error"
            assert client.post("/jobs", files={"file": ("d.wav", b"x")}).status_code == 503
        finally:
            release.set()
    assert client.app.state.worker.is_alive() is False
    assert client.app.state.store.get(active).result is None
    assert client.app.state.store.get(active).progress is None
    assert not list(settings.work_dir.iterdir())


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_invalid_timeout_is_rejected(settings, timeout):
    with TestClient(create_app(settings, FakeBackend)) as client:
        response = client.post(
            "/jobs", data={"timeout_sec": timeout}, files={"file": ("a.wav", b"x")}
        )
        assert response.status_code == 422


def test_request_replay_during_processing_does_not_queue_again(settings, monkeypatch):
    import soriham_stt.server as server
    from soriham_stt.schemas import JobResult

    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return JobResult(language="ko", segments=[])

    monkeypatch.setattr(server, "run_job", blocked)
    request_id = str(uuid4())
    data = {"request_id": request_id, "model": "tiny"}
    with TestClient(create_app(settings, FakeBackend)) as client:
        try:
            first = client.post("/jobs", data=data, files={"file": ("a.wav", b"x")})
            assert entered.wait(2)
            again = client.post("/jobs", data=data, files={"file": ("a.wav", b"x")})
            assert first.json() == again.json()
            conflict = client.post(
                "/jobs", data={**data, "model": "different"}, files={"file": ("a.wav", b"x")}
            )
            assert conflict.status_code == 409
        finally:
            release.set()
        wait_done(client, first.json()["job_id"])
    assert calls == [1]
