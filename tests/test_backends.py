from __future__ import annotations

import pytest

from soriham_stt.backends import detect_device
from soriham_stt.backends.mlx import resolve_repo


def test_detect_device_forced_value_wins():
    assert detect_device("cuda") == "cuda"
    assert detect_device("cpu") == "cpu"


def test_resolve_repo_known_alias():
    assert resolve_repo("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"


def test_resolve_repo_explicit_repo_passthrough():
    assert resolve_repo("mlx-community/whisper-tiny") == "mlx-community/whisper-tiny"


def test_resolve_repo_unknown_model_rejected():
    with pytest.raises(ValueError, match="모르는 모델"):
        resolve_repo("large-v99")


def test_mlx_진행바_대역은_프레임을_비율로_바꾼다() -> None:
    """mlx_whisper 내부 tqdm 대신 끼우는 대역. mlx 없이도 검증되는 순수 로직."""
    from soriham_stt.backends.mlx import _ProgressBar

    seen: list[float] = []
    bar = _ProgressBar(total=100, report=seen.append)
    with bar:
        bar.update(25)
        bar.update(25)
        bar.update(60)  # 총량을 넘겨도 1을 넘지 않는다

    assert seen == [0.25, 0.5, 1.0]


def test_mlx_진행바는_총량을_모르면_보고하지_않는다() -> None:
    from soriham_stt.backends.mlx import _ProgressBar

    seen: list[float] = []
    bar = _ProgressBar(total=None, report=seen.append)
    bar.update(10)

    assert seen == []
