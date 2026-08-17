from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from fakes import FakeBackend
from soriham_stt.config import Settings
from soriham_stt.server import create_app


def make_client(settings: Settings, backend: FakeBackend) -> TestClient:
    app = create_app(settings=settings, backend_factory=lambda: backend)
    return TestClient(app)


def wait_done(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.02)
    raise AssertionError("잡이 제한 시간 안에 끝나지 않음")


def test_upload_job_returns_contract_shape(settings: Settings) -> None:
    with make_client(settings, FakeBackend()) as client:
        resp = client.post("/jobs", files={"file": ("a.wav", b"pcm", "audio/wav")})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        body = wait_done(client, job_id)
        assert body["status"] == "done"
        assert body["error"] is None
        result = body["result"]
        assert result["language"] == "ko"
        seg = result["segments"][0]
        assert seg == {
            "start": 0.0,
            "end": 1.2,
            "text": "안녕하세요",
            "speaker": None,
            "words": [["안녕하세요", 0.0, 1.2]],
        }
        assert result["meta"]["model"] == "tiny"
        assert result["meta"]["diarized"] is False


def test_upload_temp_file_removed_after_done(settings: Settings) -> None:
    with make_client(settings, FakeBackend()) as client:
        job_id = client.post("/jobs", files={"file": ("a.wav", b"x", "audio/wav")}).json()["job_id"]
        wait_done(client, job_id)
        assert not (settings.work_dir / job_id).exists()


def test_path_job_uses_file_in_place(settings: Settings) -> None:
    assert settings.shared_dir is not None
    settings.shared_dir.mkdir(parents=True)
    audio = settings.shared_dir / "rec.m4a"
    audio.write_bytes(b"x")

    backend = FakeBackend()
    with make_client(settings, backend) as client:
        job_id = client.post("/jobs", data={"path": str(audio)}).json()["job_id"]
        wait_done(client, job_id)
        assert backend.calls == [audio]
        assert audio.exists()


def test_path_outside_shared_dir_rejected(settings: Settings, tmp_path: Path) -> None:
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"x")
    with make_client(settings, FakeBackend()) as client:
        assert client.post("/jobs", data={"path": str(outside)}).status_code == 403
        assert client.post("/jobs", data={"path": "../../etc/passwd"}).status_code == 403


def test_path_missing_file_404(settings: Settings) -> None:
    assert settings.shared_dir is not None
    settings.shared_dir.mkdir(parents=True)
    with make_client(settings, FakeBackend()) as client:
        resp = client.post("/jobs", data={"path": str(settings.shared_dir / "no.wav")})
        assert resp.status_code == 404


def test_file_and_path_are_mutually_exclusive(settings: Settings) -> None:
    assert settings.shared_dir is not None
    settings.shared_dir.mkdir(parents=True)
    audio = settings.shared_dir / "a.wav"
    audio.write_bytes(b"x")
    with make_client(settings, FakeBackend()) as client:
        assert client.post("/jobs").status_code == 422
        resp = client.post(
            "/jobs",
            files={"file": ("a.wav", b"x", "audio/wav")},
            data={"path": str(audio)},
        )
        assert resp.status_code == 422


def test_failed_job_reports_error_and_worker_survives(settings: Settings) -> None:
    with make_client(settings, FakeBackend(fail=True)) as client:
        job_id = client.post("/jobs", files={"file": ("a.wav", b"x", "audio/wav")}).json()["job_id"]
        body = wait_done(client, job_id)
        assert body["status"] == "error"
        assert "가짜 백엔드 실패" in body["error"]
        assert body["result"] is None

        # 실패 후에도 워커는 다음 잡을 계속 처리한다
        ok_backend = FakeBackend()
    with make_client(settings, ok_backend) as client:
        job_id = client.post("/jobs", files={"file": ("b.wav", b"x", "audio/wav")}).json()["job_id"]
        assert wait_done(client, job_id)["status"] == "done"


def test_unknown_job_404(settings: Settings) -> None:
    with make_client(settings, FakeBackend()) as client:
        assert client.get("/jobs/deadbeef").status_code == 404


def test_health(settings: Settings) -> None:
    with make_client(settings, FakeBackend()) as client:
        body = client.get("/health").json()
        assert body["device"] == "cpu"
        assert body["model"] == "tiny"
        assert "soriham-stt" in body["versions"]

        # 잡을 하나 처리해 백엔드가 로드되면 백엔드 버전도 노출된다
        job_id = client.post("/jobs", files={"file": ("a.wav", b"x", "audio/wav")}).json()["job_id"]
        wait_done(client, job_id)
        assert client.get("/health").json()["versions"]["fake"] == "0"
