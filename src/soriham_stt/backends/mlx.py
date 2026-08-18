"""Apple Silicon용 mlx-whisper 백엔드. 이 모듈은 선택된 경우에만 import된다."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

from soriham_stt.backends.base import RawSegment, RawTranscript

logger = logging.getLogger(__name__)

# whisper 모델 별칭 → mlx-community 변환 모델 레포 (레포마다 이름 규칙이 달라 표로 고정)
_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def resolve_repo(model: str) -> str:
    if "/" in model:  # HF 레포를 직접 지정한 경우
        return model
    try:
        return _MLX_REPOS[model]
    except KeyError:
        raise ValueError(f"mlx 백엔드가 모르는 모델입니다: {model}") from None


class MlxWhisperBackend:
    name = "mlx-whisper"
    device = "mlx"

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        language: str | None,
        on_progress: Callable[[float], None] | None = None,
    ) -> RawTranscript:
        import mlx_whisper  # 모델 로드는 mlx_whisper가 레포 단위로 내부 캐시

        with _progress_probe(on_progress):
            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=resolve_repo(model),
                language=language,
                word_timestamps=True,
            )
        segments = [
            RawSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=str(seg["text"]).strip(),
                words=[
                    (str(w["word"]).strip(), float(w["start"]), float(w["end"]))
                    for w in seg.get("words", [])
                ],
            )
            for seg in result["segments"]
        ]
        return RawTranscript(language=result.get("language"), segments=segments)

    def versions(self) -> dict[str, str]:
        return {"mlx-whisper": version("mlx-whisper")}


class _ProgressBar:
    """mlx_whisper 내부 진행 표시줄을 대신해 비율만 뽑아내는 대역."""

    def __init__(self, total: Any, report: Callable[[float], None]) -> None:
        self._total = float(total or 0)
        self._done = 0.0
        self._report = report

    def update(self, n: Any = 1) -> None:
        self._done += float(n or 0)
        if self._total > 0:
            self._report(min(1.0, self._done / self._total))

    def __enter__(self) -> _ProgressBar:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@contextmanager
def _progress_probe(on_progress: Callable[[float], None] | None) -> Iterator[None]:
    """전사 진행률을 얻기 위해 mlx_whisper의 tqdm을 잠깐 대역으로 바꾼다.

    mlx_whisper.transcribe에는 진행 콜백이 없고, 내부에서 `tqdm.tqdm(total=프레임 수)`을
    만들어 창마다 update한다. 그 호출만 가로챈다. 서드파티 내부 구조에 기대는 코드이므로
    어긋나면 진행률만 포기하고 전사는 그대로 진행한다.
    """
    if on_progress is None:
        yield
        return
    try:
        from mlx_whisper import transcribe as transcribe_module
    except Exception:  # noqa: BLE001 - 진행률은 부가 기능
        yield
        return

    original = getattr(transcribe_module, "tqdm", None)
    if original is None:
        yield
        return

    class _Shim:
        @staticmethod
        def tqdm(*args: Any, **kwargs: Any) -> _ProgressBar:
            return _ProgressBar(kwargs.get("total"), on_progress)

    transcribe_module.tqdm = _Shim
    try:
        yield
    finally:
        transcribe_module.tqdm = original
