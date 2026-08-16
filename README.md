# soriham-stt

소리함 — 음성 녹음 아카이브(로컬 STT, 화자분리, AI 요약, 검색) — 의 음성 변환
러너입니다. HTTP 잡 API 뒤에서 whisper 계열 STT와 pyannote 화자분리를 실행하고,
단어 타임스탬프와 화자 구간을 병합한 결과를 돌려줍니다. 로컬 프로세스로도 클라우드
GPU에서도 같은 계약으로 동작합니다.

아직 코드가 없는 부트스트랩 상태입니다. 예정 스택: Python 3.12+, FastAPI,
mlx-whisper/faster-whisper, pyannote.audio, Docker.

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
