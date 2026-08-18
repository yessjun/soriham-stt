"""전사 백엔드 계약. 무거운 의존성을 import하지 않는다."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from soriham_stt.schemas import Word


@dataclass
class RawSegment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    # 환청 판정에 쓰는 whisper 자체 지표. 압축률이 높을수록 같은 말을 되풀이한 것이다
    compression_ratio: float | None = None
    no_speech_prob: float | None = None
    avg_logprob: float | None = None


@dataclass
class RawTranscript:
    language: str | None
    segments: list[RawSegment]


class TranscribeBackend(Protocol):
    """whisper 구현체가 만족해야 하는 인터페이스.

    모델 로드는 transcribe 첫 호출에서 lazy하게 수행하고 인스턴스에 캐시한다.
    """

    name: str
    device: str

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        language: str | None,
        on_progress: Callable[[float], None] | None = None,
    ) -> RawTranscript: ...

    def versions(self) -> dict[str, str]: ...
