"""잡 하나의 실행 흐름: 전사 → (화자분리 → 병합은 후속 연결)."""

from __future__ import annotations

import time

from soriham_stt.backends.base import TranscribeBackend
from soriham_stt.jobs import Job
from soriham_stt.schemas import JobResult, Segment


def run_job(job: Job, backend: TranscribeBackend) -> JobResult:
    started = time.monotonic()
    raw = backend.transcribe(job.audio_path, model=job.params.model, language=job.params.language)
    segments = [
        Segment(start=s.start, end=s.end, text=s.text, speaker=None, words=s.words)
        for s in raw.segments
    ]
    meta = {
        "device": backend.device,
        "model": job.params.model,
        "diarized": False,
        "elapsed_sec": round(time.monotonic() - started, 2),
    }
    return JobResult(language=raw.language, segments=segments, meta=meta)
