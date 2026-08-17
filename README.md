# soriham-stt

소리함 — 음성 녹음 아카이브(로컬 STT, 화자분리, AI 요약, 검색) — 의 음성 변환
러너입니다. HTTP 잡 API 뒤에서 whisper 계열 STT와 pyannote 화자분리를 실행하고,
단어 타임스탬프와 화자 구간을 병합한 결과를 돌려줍니다. 로컬 프로세스로도 클라우드
GPU에서도 같은 계약으로 동작합니다.

스택: Python 3.12+, FastAPI, mlx-whisper/faster-whisper, pyannote.audio, Docker.

## HTTP 잡 API

```
POST /jobs        오디오 업로드(file, multipart) 또는 공유 폴더 경로(path) 중 하나
                  + model, language, diarize        → {"job_id": "..."}
GET  /jobs/{id}   → {"status": "queued|running|done|error",
                     "result": {"language", "segments": [{"start", "end", "text",
                                "speaker", "words": [[단어, 시작, 끝], …]}], "meta"},
                     "error": null}
GET  /health      → {"device": "mlx|cuda|cpu", "model", "versions"}
```

- `path` 입력은 `STT_SHARED_DIR` 아래의 파일만 허용합니다(미설정 시 거부). 원본은
  읽기만 하고 이동·삭제하지 않습니다. 업로드된 파일은 잡 종료 시 즉시 지웁니다.
- 잡 상태는 메모리에만 유지합니다. 러너가 재시작되면 호출자는 `GET /jobs/{id}`의
  404를 보고 재제출합니다. 완료된 잡 상태는 `STT_JOB_TTL`(기본 1시간) 뒤 정리됩니다.
- 잡은 단일 워커가 직렬로 처리합니다(추론 자원이 하나이므로).

### 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `STT_DEVICE` | 자동 선택 | 백엔드 강제 지정 (`mlx` \| `cuda` \| `cpu`) |
| `STT_MODEL` | `large-v3-turbo` | 기본 whisper 모델 |
| `HF_TOKEN` | 없음 | pyannote 화자분리용 토큰. 없으면 화자분리를 생략합니다 |
| `STT_WORK_DIR` | 시스템 임시 폴더 | 업로드 오디오 임시 저장 위치 |
| `STT_SHARED_DIR` | 없음 | `path` 입력을 허용할 최상위 폴더 |
| `STT_JOB_TTL` | `3600` | 완료된 잡 상태 유지 시간(초) |

### 실행

로컬(백엔드는 실행 환경에 맞는 extra 선택 — Apple Silicon은 `mlx`, 그 외 `cpu`,
화자분리는 `diarize` 추가):

```bash
uv sync --extra mlx --extra diarize
uv run uvicorn --factory soriham_stt.server:create_app --host 0.0.0.0 --port 8100
```

도커(CPU 백엔드 + 화자분리 포함, 모델 가중치는 `/data/hf` 볼륨에 캐시):

```bash
docker run -p 8100:8100 -v hf-cache:/data/hf -e HF_TOKEN=<hf-token> \
  ghcr.io/yessjun/soriham-stt:latest
```

## 전체 아키텍처

<!-- arch:begin -->
```
[녹음 폴더] ──스캔·감시──▶ [soriham-api: 인제스트 + PostgreSQL]
                                     │
                                     ▼
                            [soriham-api: 워커] ──HTTP 잡 API──▶ [soriham-stt: 변환 러너]
                                     │                            (whisper + 화자분리)
                                     ▼
[브라우저] ◀──▶ [soriham-console 웹 UI] ──REST──▶ [soriham-api: FastAPI]
```

| 레포지토리 | 역할 |
|---|---|
| [soriham-api](https://github.com/yessjun/soriham-api) | FastAPI 백엔드와 처리 워커 (Python, PostgreSQL) |
| [soriham-console](https://github.com/yessjun/soriham-console) | 웹 콘솔 (React, TypeScript) |
| [soriham-stt](https://github.com/yessjun/soriham-stt) | 음성 변환 러너 (whisper 계열, pyannote) |
<!-- arch:end -->
