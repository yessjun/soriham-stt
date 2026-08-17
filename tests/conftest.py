from __future__ import annotations

from pathlib import Path

import pytest

from soriham_stt.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        device="cpu",
        default_model="tiny",
        hf_token=None,
        work_dir=tmp_path / "work",
        shared_dir=tmp_path / "shared",
        job_ttl=3600.0,
        max_upload_bytes=64 * 1024 * 1024,
    )
