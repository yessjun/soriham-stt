"""FastAPI 앱: HTTP 잡 API 3개 엔드포인트."""

from __future__ import annotations

import shutil
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, UploadFile

import soriham_stt
from soriham_stt.backends import detect_device, select_backend
from soriham_stt.backends.base import TranscribeBackend
from soriham_stt.config import Settings, load_settings
from soriham_stt.jobs import Job, JobParams, JobStore, JobWorker, new_job_id
from soriham_stt.pipeline import run_job
from soriham_stt.schemas import (
    HealthResponse,
    JobCreateResponse,
    JobResult,
    JobStage,
    JobStatusResponse,
)


def create_app(
    settings: Settings | None = None,
    backend_factory: Callable[[], TranscribeBackend] | None = None,
) -> FastAPI:
    cfg = settings or load_settings()
    factory = backend_factory or (lambda: select_backend(cfg))

    # 백엔드는 첫 잡에서 lazy 생성 — 실제 사용은 워커 스레드 하나뿐이지만
    # /health도 읽을 수 있으므로 생성만 락으로 보호한다
    backend_holder: dict[str, TranscribeBackend] = {}
    backend_lock = threading.Lock()

    def get_backend() -> TranscribeBackend:
        with backend_lock:
            if "backend" not in backend_holder:
                backend_holder["backend"] = factory()
            return backend_holder["backend"]

    store = JobStore()

    def run(job: Job) -> JobResult:
        def report(stage: JobStage, ratio: float | None) -> None:
            job.stage = stage
            job.progress = ratio

        return run_job(job, get_backend(), hf_token=cfg.hf_token, on_progress=report)

    worker = JobWorker(store, run, cfg.job_ttl)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 이전 실행이 남긴 업로드 잔여물 정리 — 잡 상태가 인메모리라 재시작 후엔 고아 파일
        shutil.rmtree(cfg.work_dir, ignore_errors=True)
        worker.start()
        yield
        worker.shutdown()

    app = FastAPI(title="soriham-stt", version=soriham_stt.__version__, lifespan=lifespan)

    @app.post("/jobs", response_model=JobCreateResponse)
    async def create_job(
        file: UploadFile | None = None,
        path: Annotated[str | None, Form()] = None,
        model: Annotated[str | None, Form()] = None,
        language: Annotated[str | None, Form()] = None,
        diarize: Annotated[bool, Form()] = True,
    ) -> JobCreateResponse:
        if (file is None) == (path is None):
            raise HTTPException(422, "file과 path 중 정확히 하나를 지정해야 합니다")

        job_id = new_job_id()
        cleanup_dir: Path | None = None
        if path is not None:
            audio_path = _validate_shared_path(path, cfg)
        else:
            assert file is not None
            cleanup_dir = cfg.work_dir / job_id
            cleanup_dir.mkdir(parents=True, exist_ok=True)
            suffix = Path(file.filename or "audio").suffix or ".bin"
            audio_path = cleanup_dir / f"audio{suffix}"
            try:
                _copy_limited(file.file, audio_path, cfg.max_upload_bytes)
            except Exception:
                # 복사 실패(연결 끊김, 디스크 부족, 한도 초과) 시 부분 파일을 남기지 않는다
                shutil.rmtree(cleanup_dir, ignore_errors=True)
                raise

        job = Job(
            id=job_id,
            audio_path=audio_path,
            params=JobParams(model=model or cfg.default_model, language=language, diarize=diarize),
            cleanup_dir=cleanup_dir,
        )
        worker.submit(job)
        return JobCreateResponse(job_id=job_id)

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job(job_id: str) -> JobStatusResponse:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "잡이 없습니다")
        return JobStatusResponse(
            status=job.status,
            result=job.result,
            error=job.error,
            stage=job.stage,
            progress=job.progress,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        backend = backend_holder.get("backend")
        versions = {"soriham-stt": soriham_stt.__version__}
        if backend is not None:
            versions.update(backend.versions())
        return HealthResponse(
            device=backend.device if backend is not None else detect_device(cfg.device),
            model=cfg.default_model,
            versions=versions,
        )

    return app


def _copy_limited(src, dest: Path, max_bytes: int) -> None:
    """업로드를 한도까지만 복사한다. 초과 시 413."""
    written = 0
    with dest.open("wb") as out:
        while chunk := src.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(413, "업로드 크기 한도를 초과했습니다")
            out.write(chunk)


def _validate_shared_path(path: str, cfg: Settings) -> Path:
    if cfg.shared_dir is None:
        raise HTTPException(403, "경로 입력이 허용돼 있지 않습니다")
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(cfg.shared_dir):
        raise HTTPException(403, "허용된 폴더 밖의 경로입니다")
    if not resolved.is_file():
        raise HTTPException(404, "오디오 파일이 없습니다")
    return resolved
