"""잡 하나의 실행 흐름: 전사 → 화자분리(옵션) → 병합."""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from soriham_stt import diarize as diarize_module
from soriham_stt.audio import DecodeFailed, decode_to_wav
from soriham_stt.backends.base import TranscribeBackend
from soriham_stt.jobs import Job
from soriham_stt.merge import Turn, merge_transcript
from soriham_stt.schemas import JobResult, JobStage, Segment

logger = logging.getLogger(__name__)

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
    with tempfile.TemporaryDirectory(prefix="soriham-") as tmp:
        # 전사와 화자분리가 같은 파형을 보게 한 번만 편다 (라이브러리별 디코더 차이 제거)
        decode_error: str | None = None
        try:
            audio_path = decode_to_wav(job.audio_path, Path(tmp))
        except DecodeFailed as exc:
            # 변환이 안 되면 원본 그대로 시도한다 — 전사까지 막을 이유는 없다
            logger.warning("오디오 선변환 실패, 원본으로 진행: %s", exc)
            decode_error = str(exc)
            audio_path = job.audio_path

        if on_progress is not None:
            on_progress("transcribe", 0.0)
        raw = backend.transcribe(
            audio_path,
            model=job.params.model,
            language=job.params.language,
            on_progress=(lambda r: on_progress("transcribe", r)) if on_progress else None,
        )

        turns: list[Turn] | None = None
        diarize_error: str | None = None
        if job.params.diarize:
            # 화자분리는 진행률을 낼 수 없어 단계만 알린다
            if on_progress is not None:
                on_progress("diarize", None)
            try:
                turns = (diarizer or diarize_module.diarize)(audio_path, hf_token)
            except Exception as exc:  # noqa: BLE001 - 녹취록은 살리고 사유를 남긴다
                logger.exception("화자분리 실패 — 화자 없이 계속")
                diarize_error = f"{type(exc).__name__}: {exc}"[:400]

    if turns:
        segments = merge_transcript(raw, turns)
    else:
        segments = [
            Segment(start=s.start, end=s.end, text=s.text, speaker=None, words=s.words)
            for s in raw.segments
        ]

    meta: dict[str, object] = {
        "device": backend.device,
        "model": job.params.model,
        "diarized": bool(turns),
        "elapsed_sec": round(time.monotonic() - started, 2),
    }
    if diarize_error:
        meta["diarize_error"] = diarize_error
    if decode_error:
        meta["decode_error"] = decode_error
    return JobResult(language=raw.language, segments=segments, meta=meta)
