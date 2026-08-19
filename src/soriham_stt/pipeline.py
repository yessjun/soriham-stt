"""잡 하나의 실행 흐름: 전사 → 재전사 → 화자분리(옵션) → 병합."""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from soriham_stt import diarize as diarize_module
from soriham_stt.audio import DecodeFailed, cut_wav, decode_to_wav
from soriham_stt.backends.base import RawSegment, TranscribeBackend
from soriham_stt.jobs import Job
from soriham_stt.merge import Turn, merge_transcript
from soriham_stt.noise import classify
from soriham_stt.schemas import JobResult, JobStage, Segment

logger = logging.getLogger(__name__)

Diarizer = Callable[[Path, str | None], "list[Turn] | None"]
# (단계, 진행률 0~1 또는 None)
ProgressHook = Callable[[JobStage, float | None], None]

# 이보다 짧은 구간은 잘라 다시 돌려도 건질 게 없다
MIN_RETRY_SEC = 1.0
# 재전사에 쓰는 오디오 총량의 상한(원본 길이 대비). 전체가 잡음인 녹음에서 시간이
# 배로 드는 것을 막는다. 상한을 넘으면 남은 구간은 그대로 표시만 남긴다
RETRY_BUDGET_RATIO = 0.25
# 짧은 녹음에서는 비율 상한이 너무 빡빡하다. 이 길이까지는 비율과 무관하게 허용한다
RETRY_MIN_BUDGET_SEC = 60.0
# 붙어 있는 구간을 하나로 합칠 때의 간격
SPAN_GAP_SEC = 2.0


def run_job(
    job: Job,
    backend: TranscribeBackend,
    hf_token: str | None = None,
    diarizer: Diarizer | None = None,
    on_progress: ProgressHook | None = None,
) -> JobResult:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="soriham-") as tmp:
        work_dir = Path(tmp)
        # 전사와 화자분리가 같은 파형을 보게 한 번만 편다 (라이브러리별 디코더 차이 제거)
        decode_error: str | None = None
        try:
            audio_path = decode_to_wav(job.audio_path, work_dir)
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

        # 환청 판정은 병합 전에 원시 세그먼트에서 한다 — whisper 지표가 여기에만 있다
        total_sec = max((s.end for s in raw.segments), default=0.0)
        flags = classify(raw.segments)
        spans = _merge_spans(
            [(s.start, s.end) for s, bad in zip(raw.segments, flags, strict=True) if bad]
        )
        kept = [s for s, bad in zip(raw.segments, flags, strict=True) if not bad and s.text.strip()]

        recovered, unread = _retry_unread(
            spans, audio_path, backend, job=job, work_dir=work_dir, total_sec=total_sec
        )
        raw.segments = sorted([*kept, *recovered], key=lambda s: s.start)

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
    segments = _with_unread_marks(segments, unread)

    meta: dict[str, object] = {
        "device": backend.device,
        "model": job.params.model,
        "diarized": bool(turns),
        "elapsed_sec": round(time.monotonic() - started, 2),
        "noise_sec": round(sum(e - s for s, e in unread), 1),
        "recovered_segments": len(recovered),
    }
    if diarize_error:
        meta["diarize_error"] = diarize_error
    if decode_error:
        meta["decode_error"] = decode_error
    return JobResult(language=raw.language, segments=segments, meta=meta)


def _retry_unread(
    spans: list[tuple[float, float]],
    audio_path: Path,
    backend: TranscribeBackend,
    *,
    job: Job,
    work_dir: Path,
    total_sec: float,
) -> tuple[list[RawSegment], list[tuple[float, float]]]:
    """판정에 걸린 구간을 잘라 한 번 더 돌린다.

    통째로 돌리면 앞선 환청이 그 30초 창을 오염시켜 무너지는데, 구간만 떼어내면
    오염원이 없어 제대로 읽히는 경우가 있다. 실측에서 같은 문장이 반복되던 자리가
    잘라 돌리자 알아들을 수 있는 대화로 나왔다.

    재전사 결과도 판정을 다시 통과해야 채택한다. 실패하면 그 구간은 표시로 남긴다.
    """
    recovered: list[RawSegment] = []
    unread: list[tuple[float, float]] = []
    budget = max(total_sec * RETRY_BUDGET_RATIO, RETRY_MIN_BUDGET_SEC)
    used = 0.0

    for index, (start, end) in enumerate(spans):
        duration = end - start
        if duration < MIN_RETRY_SEC or used + duration > budget:
            unread.append((start, end))
            continue
        used += duration
        try:
            clip = cut_wav(audio_path, start, duration, work_dir / f"retry-{index}.wav")
            again = backend.transcribe(clip, model=job.params.model, language=job.params.language)
        except Exception:  # noqa: BLE001 - 재전사는 부가 시도다
            logger.exception("재전사 실패: %.1f~%.1f", start, end)
            unread.append((start, end))
            continue

        flags = classify(again.segments)
        good = [
            s for s, bad in zip(again.segments, flags, strict=True) if not bad and s.text.strip()
        ]
        if not good:
            unread.append((start, end))
            continue
        recovered.extend(_shift(s, start) for s in good)

    return recovered, unread


def _shift(seg: RawSegment, offset: float) -> RawSegment:
    return RawSegment(
        start=seg.start + offset,
        end=seg.end + offset,
        text=seg.text,
        words=[(w, s + offset, e + offset) for w, s, e in seg.words],
        compression_ratio=seg.compression_ratio,
        no_speech_prob=seg.no_speech_prob,
        avg_logprob=seg.avg_logprob,
    )


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """붙어 있는 구간을 하나로 합친다 — 잘게 쪼개진 표시가 줄줄이 서는 것보다 낫다."""
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if merged and start - merged[-1][1] <= SPAN_GAP_SEC:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _with_unread_marks(segments: list[Segment], spans: list[tuple[float, float]]) -> list[Segment]:
    """끝내 받아적지 못한 구간을 자리표시 세그먼트로 끼워 넣는다.

    지워버리면 녹취록에 설명 없는 구멍이 남는다. 무엇을 버렸는지는 보여야 한다.
    """
    if not spans:
        return segments
    marks = [
        Segment(start=start, end=end, text="", speaker=None, words=[], kind="noise")
        for start, end in _merge_spans(spans)
    ]
    return sorted([*segments, *marks], key=lambda s: (s.start, s.end))
