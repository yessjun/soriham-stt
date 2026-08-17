"""잡 상태 저장소와 단일 워커 스레드.

추론 자원(GPU/Metal/CPU)이 하나이므로 잡은 큐에 넣어 직렬로 처리한다. 상태는
인메모리로만 유지한다 — 러너가 재시작하면 진행 중이던 잡은 사라지고, 호출자는
GET /jobs/{id}의 404를 보고 재제출한다.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from soriham_stt.schemas import JobResult, JobState

logger = logging.getLogger(__name__)


@dataclass
class JobParams:
    model: str
    language: str | None
    diarize: bool


@dataclass
class Job:
    id: str
    audio_path: Path
    params: JobParams
    # 업로드로 받은 파일이면 잡 종료 시 이 디렉터리를 통째로 지운다
    cleanup_dir: Path | None = None
    status: JobState = "queued"
    result: JobResult | None = None
    error: str | None = None
    finished_at: float | None = None


def new_job_id() -> str:
    return uuid.uuid4().hex


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def sweep(self, ttl: float, now: float | None = None) -> None:
        """TTL이 지난 완료·실패 잡을 상태 저장소에서 제거한다."""
        now = time.time() if now is None else now
        with self._lock:
            expired = [
                j.id
                for j in self._jobs.values()
                if j.finished_at is not None and now - j.finished_at > ttl
            ]
            for job_id in expired:
                del self._jobs[job_id]


class JobWorker(threading.Thread):
    """큐에서 잡을 꺼내 파이프라인을 직렬 실행하는 데몬 스레드."""

    def __init__(
        self,
        store: JobStore,
        run_pipeline: Callable[[Job], JobResult],
        job_ttl: float,
    ) -> None:
        super().__init__(name="stt-job-worker", daemon=True)
        self._store = store
        self._run_pipeline = run_pipeline
        self._job_ttl = job_ttl
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()

    def submit(self, job: Job) -> None:
        self._store.add(job)
        self._queue.put(job.id)

    def shutdown(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                self._store.sweep(self._job_ttl)
                continue
            job = self._store.get(job_id)
            if job is None:
                continue
            self._process(job)

    def _process(self, job: Job) -> None:
        job.status = "running"
        try:
            job.result = self._run_pipeline(job)
        except Exception as exc:  # noqa: BLE001 - 잡 실패는 격리하고 워커는 계속
            logger.exception("job %s failed", job.id)
            job.error = f"{type(exc).__name__}: {exc}"
        # 임시 파일 정리까지 끝낸 뒤에 종료 상태로 전이한다 — 호출자가 done을
        # 관측한 시점에는 정리가 보장돼야 한다
        if job.cleanup_dir is not None:
            shutil.rmtree(job.cleanup_dir, ignore_errors=True)
        job.finished_at = time.time()
        job.status = "error" if job.error is not None else "done"
