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
