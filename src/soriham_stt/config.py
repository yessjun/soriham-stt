"""환경 변수 기반 러너 설정."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "large-v3-turbo"
DEFAULT_JOB_TTL = 3600.0


@dataclass(frozen=True)
class Settings:
    """러너 프로세스 전역 설정. 값은 환경 변수에서 읽는다.

    - STT_DEVICE: 백엔드 강제 선택 (mlx | cuda | cpu). 없으면 자동 선택
    - STT_MODEL: 기본 whisper 모델 이름
    - HF_TOKEN: pyannote 화자분리용 HuggingFace 토큰. 없으면 화자분리 생략
    - STT_WORK_DIR: 업로드 오디오 임시 저장 위치
    - STT_SHARED_DIR: 경로 입력을 허용할 최상위 폴더. 없으면 경로 입력 자체를 거부
    - STT_JOB_TTL: 완료·실패한 잡 상태를 유지할 시간(초)
    """

    device: str | None
    default_model: str
    hf_token: str | None
    work_dir: Path
    shared_dir: Path | None
    job_ttl: float


def load_settings(env: dict[str, str] | None = None) -> Settings:
    e = os.environ if env is None else env
    if env is None:
        _load_dotenv(e)
    shared = e.get("STT_SHARED_DIR")
    return Settings(
        device=e.get("STT_DEVICE") or None,
        default_model=e.get("STT_MODEL") or DEFAULT_MODEL,
        hf_token=e.get("HF_TOKEN") or None,
        work_dir=Path(e.get("STT_WORK_DIR") or Path(tempfile.gettempdir()) / "soriham-stt"),
        shared_dir=Path(shared).resolve() if shared else None,
        job_ttl=float(e.get("STT_JOB_TTL") or DEFAULT_JOB_TTL),
    )


def _load_dotenv(e) -> None:
    """레포 루트의 .env를 읽어 미설정 키만 채운다 (HF_TOKEN 등)."""
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in e:
            e[key] = value.strip().strip("'\"")
