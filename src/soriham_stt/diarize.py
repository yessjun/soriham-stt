"""pyannote 화자분리 래퍼. 토큰이 없으면 우아하게 생략한다."""

from __future__ import annotations

import logging
import os
import wave
from pathlib import Path
from typing import Any

from soriham_stt.merge import Turn

logger = logging.getLogger(__name__)

DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"

_pipeline_cache: dict[str, object] = {}


def diarize(audio_path: Path, hf_token: str | None) -> list[Turn] | None:
    """화자 구간을 돌려준다. 토큰이 없으면 None(화자분리 생략).

    실패는 삼키지 않고 올린다 — 조용히 화자 없는 녹취록이 나오면 몇 시간을 돌린 뒤에야
    알게 된다. 호출자가 잡아서 결과에 사유를 남긴다.
    """
    if not hf_token:
        logger.info("HF_TOKEN 없음 — 화자분리 생략")
        return None
    pipeline = _get_pipeline(hf_token)
    output = pipeline(_as_input(audio_path))
    # pyannote 4.x는 DiarizeOutput 래퍼를, 3.x는 Annotation을 그대로 돌려준다
    annotation = getattr(output, "speaker_diarization", output)
    return [
        (float(turn.start), float(turn.end), str(label))
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]


def _as_input(audio_path: Path) -> dict[str, Any] | str:
    """파형을 통째로 올려 준다. 못 읽는 형식이면 경로 그대로.

    경로를 넘기면 pyannote가 임베딩 단계에서 화자 구간마다 파일을 다시 디코딩한다.
    175분 녹음 실측에서 그 크롭이 CPU 한 코어를 붙잡고 GPU는 내내 놀았다 — 전사는
    100배속인데 전체는 9배속으로 떨어졌다. 앞단에서 이미 16kHz 모노로 펴 두므로
    메모리에 올려 넘기면 그 디코딩이 통째로 사라진다.
    """
    try:
        with wave.open(str(audio_path), "rb") as handle:
            if handle.getcomptype() != "NONE" or handle.getsampwidth() != 2:
                return str(audio_path)
            channels = handle.getnchannels()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, OSError) as exc:
        # 선변환이 실패해 원본이 그대로 왔을 때다. pyannote가 알아서 읽게 둔다
        logger.info("파형을 못 읽어 경로로 넘김: %s", exc)
        return str(audio_path)

    # numpy는 화자분리 의존성(pyannote)이 이미 끌고 온다 — 이 함수는 그때만 돈다
    import numpy as np
    import torch

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    shaped = (
        samples.reshape(1, -1)
        if channels == 1
        else np.ascontiguousarray(samples.reshape(-1, channels).T)
    )
    return {"waveform": torch.from_numpy(shaped), "sample_rate": rate}


def _get_pipeline(hf_token: str):
    model = os.environ.get("STT_DIARIZE_MODEL") or DEFAULT_DIARIZE_MODEL
    if model not in _pipeline_cache:
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(model, token=hf_token)
        _pipeline_cache[model] = pipeline.to(_best_torch_device())
    return _pipeline_cache[model]


def _best_torch_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
