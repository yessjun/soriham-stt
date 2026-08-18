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
from soriham_stt.noise import classify
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

    # 환청 판정은 병합 전에 원시 세그먼트에서 한다 — whisper 지표가 여기에만 있다
    noise_flags = classify(raw.segments)
    noise_spans = [
        (seg.start, seg.end)
        for seg, is_noise in zip(raw.segments, noise_flags, strict=True)
        if is_noise
    ]
    raw.segments = [
        seg
        for seg, is_noise in zip(raw.segments, noise_flags, strict=True)
        if not is_noise and seg.text.strip()
    ]

    if turns:
        segments = merge_transcript(raw, turns)
    else:
        segments = [
            Segment(start=s.start, end=s.end, text=s.text, speaker=None, words=s.words)
            for s in raw.segments
        ]
    segments = _with_noise(segments, noise_spans)

    meta: dict[str, object] = {
        "device": backend.device,
        "model": job.params.model,
        "diarized": bool(turns),
        "elapsed_sec": round(time.monotonic() - started, 2),
    }
    meta["noise_sec"] = round(sum(e - s for s, e in noise_spans), 1)
    if diarize_error:
        meta["diarize_error"] = diarize_error
    if decode_error:
        meta["decode_error"] = decode_error
    return JobResult(language=raw.language, segments=segments, meta=meta)


def _with_noise(segments: list[Segment], spans: list[tuple[float, float]]) -> list[Segment]:
    """소음으로 판정된 구간을 자리표시 세그먼트로 끼워 넣는다.

    지워버리면 녹취록에 설명 없는 구멍이 남는다. 무엇을 버렸는지는 보여야 한다.
    2초 이내로 붙은 구간은 하나로 합친다 — 잘게 쪼개진 표시가 줄줄이 서는 것보다 낫다.
    """
    if not spans:
        return segments
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if merged and start - merged[-1][1] <= 2.0:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    marks = [
        Segment(start=start, end=end, text="", speaker=None, words=[], kind="noise")
        for start, end in merged
    ]
    return sorted([*segments, *marks], key=lambda s: (s.start, s.end))
