"""pyannote 화자분리 래퍼. 토큰이 없으면 우아하게 생략한다."""

from __future__ import annotations

import logging
import os
from pathlib import Path

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
    output = pipeline(str(audio_path))
    # pyannote 4.x는 DiarizeOutput 래퍼를, 3.x는 Annotation을 그대로 돌려준다
    annotation = getattr(output, "speaker_diarization", output)
    return [
        (float(turn.start), float(turn.end), str(label))
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]


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
