"""잡 하나의 실행 흐름: 전사 → 화자분리(옵션) → 병합."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from soriham_stt import diarize as diarize_module
from soriham_stt.backends.base import TranscribeBackend
from soriham_stt.jobs import Job
from soriham_stt.merge import Turn, merge_transcript
from soriham_stt.schemas import JobResult, Segment

Diarizer = Callable[[Path, str | None], "list[Turn] | None"]


def run_job(
    job: Job,
    backend: TranscribeBackend,
    hf_token: str | None = None,
    diarizer: Diarizer | None = None,
) -> JobResult:
    started = time.monotonic()
    raw = backend.transcribe(job.audio_path, model=job.params.model, language=job.params.language)

    turns: list[Turn] | None = None
    if job.params.diarize:
        turns = (diarizer or diarize_module.diarize)(job.audio_path, hf_token)

    if turns:
        segments = merge_transcript(raw, turns)
    else:
        segments = [
            Segment(start=s.start, end=s.end, text=s.text, speaker=None, words=s.words)
            for s in raw.segments
        ]

    meta = {
        "device": backend.device,
        "model": job.params.model,
        "diarized": bool(turns),
        "elapsed_sec": round(time.monotonic() - started, 2),
    }
    return JobResult(language=raw.language, segments=segments, meta=meta)
