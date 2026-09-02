from __future__ import annotations

from pathlib import Path

import pytest

from soriham_stt.backends import detect_device
from soriham_stt.backends.mlx import resolve_repo


def test_detect_device_forced_value_wins():
    assert detect_device("cuda") == "cuda"
    assert detect_device("cpu") == "cpu"


def test_백엔드가_없으면_그_백엔드의_extra를_알려준다(monkeypatch) -> None:
    """cuda에서 `--extra cpu`를 안내하면 CPU 휠을 깔고 GPU를 못 쓴 채 돈다."""
    import importlib.util

    from soriham_stt.backends import select_backend
    from soriham_stt.config import Settings

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    settings = Settings(
        device="cuda",
        default_model="large-v3",
        hf_token=None,
        work_dir=Path("/tmp"),
        shared_dir=None,
        job_ttl=1.0,
        max_upload_bytes=1,
    )
    with pytest.raises(RuntimeError, match="--extra cuda"):
        select_backend(settings)


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


def test_진행_대역이_실제_모듈에_설치된다(monkeypatch) -> None:
    """`from mlx_whisper import transcribe`는 함수를 주므로 모듈을 직접 가져와야 한다.

    이걸 놓치면 대역이 함수 객체에 붙어 아무도 읽지 않고, 진행률이 0에서 멈춘다.
    """
    import sys
    import types

    from soriham_stt.backends.mlx import _progress_probe

    module = types.ModuleType("mlx_whisper.transcribe")
    module.tqdm = "원본"  # type: ignore[attr-defined]
    package = types.ModuleType("mlx_whisper")
    package.transcribe = lambda *a, **k: None  # 서브모듈을 가리는 함수
    monkeypatch.setitem(sys.modules, "mlx_whisper", package)
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", module)

    with _progress_probe(lambda ratio: None):
        assert module.tqdm != "원본"
        bar = module.tqdm.tqdm(total=10)  # type: ignore[attr-defined]
        assert hasattr(bar, "update")

    assert module.tqdm == "원본"  # 빠져나오면 되돌린다
