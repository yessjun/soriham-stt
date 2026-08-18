"""faster-whisper 백엔드 (CPU int8 / CUDA float16). 선택된 경우에만 import된다."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

from soriham_stt.backends.base import RawSegment, RawTranscript


class FasterWhisperBackend:
    name = "faster-whisper"

    def __init__(self, device: str) -> None:
        self.device = device
        self.compute_type = "float16" if device == "cuda" else "int8"
        self._models: dict[str, Any] = {}

    def _get_model(self, model: str) -> Any:
        if model not in self._models:
            from faster_whisper import WhisperModel

            self._models[model] = WhisperModel(
                model, device=self.device, compute_type=self.compute_type
            )
        return self._models[model]

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        language: str | None,
        on_progress: Callable[[float], None] | None = None,
    ) -> RawTranscript:
        segments_iter, info = self._get_model(model).transcribe(
            str(audio_path), language=language, word_timestamps=True
        )
        # 세그먼트는 제너레이터로 나온다 — 소비하면서 마지막 끝 시각으로 진행률을 낸다
        total = float(getattr(info, "duration", 0.0) or 0.0)
        segments = []
        for seg in segments_iter:
            words = [(w.word.strip(), float(w.start), float(w.end)) for w in seg.words or []]
            segments.append(
                RawSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text.strip(),
                    words=words,
                )
            )
            if on_progress is not None and total > 0:
                on_progress(min(1.0, float(seg.end) / total))
        return RawTranscript(language=info.language, segments=segments)

    def versions(self) -> dict[str, str]:
        return {"faster-whisper": version("faster-whisper")}
