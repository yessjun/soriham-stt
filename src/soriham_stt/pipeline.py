"""잡 하나의 실행 흐름: 전사 → 화자분리(옵션) → 병합."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from soriham_stt import diarize as diarize_module
from soriham_stt.backends.base import TranscribeBackend
from soriham_stt.jobs import Job
from soriham_stt.merge import Turn, merge_transcript
from soriham_stt.schemas import JobResult, JobStage, Segment

Diarizer = Callable[[Path, str | None], "list[Turn] | None"]
# (단계, 진행률 0~1 또는 None)
ProgressHook = Callable[[JobStage, float | None], None]


def run_job(
    job: Job,
    backend: TranscribeBackend,
    hf_token: str | None = None,
    diarizer: Diarizer | None = None,
    on_progress: ProgressHook | None = None,
) -> JobResult:
    started = time.monotonic()
    if on_progress is not None:
        on_progress("transcribe", 0.0)
    raw = backend.transcribe(
        job.audio_path,
        model=job.params.model,
        language=job.params.language,
        on_progress=(lambda r: on_progress("transcribe", r)) if on_progress else None,
    )

    turns: list[Turn] | None = None
    if job.params.diarize:
        # 화자분리는 진행률을 낼 수 없어 단계만 알린다
        if on_progress is not None:
            on_progress("diarize", None)
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
