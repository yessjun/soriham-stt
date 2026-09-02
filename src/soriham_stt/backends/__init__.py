"""백엔드 자동 선택. 무거운 모듈은 실제 선택된 경우에만 import한다."""

from __future__ import annotations

import importlib.util
import platform

from soriham_stt.backends.base import TranscribeBackend
from soriham_stt.config import Settings


def detect_device(forced: str | None = None) -> str:
    """실행 환경에서 백엔드 종류(mlx | cuda | cpu)를 고른다."""
    if forced:
        return forced
    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_whisper") is not None
    ):
        return "mlx"
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def select_backend(settings: Settings) -> TranscribeBackend:
    device = detect_device(settings.device)
    try:
        if device == "mlx":
            from soriham_stt.backends.mlx import MlxWhisperBackend

            return MlxWhisperBackend()
        # faster_whisper는 첫 잡의 모델 적재 때 import된다. 여기서 미리 보지 않으면
        # extra를 잘못 깐 것을 러너가 잡을 받은 뒤에야, 그것도 원시 ImportError로 안다
        if importlib.util.find_spec("faster_whisper") is None:
            raise ModuleNotFoundError(name="faster_whisper")
        from soriham_stt.backends.fwhisper import FasterWhisperBackend

        return FasterWhisperBackend(device=device)
    except ModuleNotFoundError as exc:
        # extra 이름이 곧 백엔드 종류다 (mlx | cuda | cpu)
        raise RuntimeError(
            f"{device} 백엔드 의존성이 없습니다. `uv sync --extra {device}`로 설치하세요"
        ) from exc
