# SPH-LLM-web

형상(OBJ / STL)을 웹에 올리고, 유체 블록·유입구·유출 영역을 화면이나 문장으로 지정한 뒤, 로컬 GPU 로 SPH 해석을 돌리고 결과를 3D 로 확인하는 개인용 유체 시뮬레이터입니다.

```
형상 업로드 (OBJ / STL)
  → 문장 지시 → 규칙 해석기 (모르는 말만 LLM) → 수정 지시(JSON) 검증 → 화면에 표시
  → 해석 실행 (로컬 Taichi: CUDA · Metal · CPU)
  → 해석 중 실시간 확인 · 스텝 제어 → 결과 (속도·압력·와도·단면·프로브) → Rhino 용 내보내기
```

- 화면과 LLM 연결은 공통 모듈 [Simweb-Core](https://github.com/MunkyunLEE/Simweb-Core) 를 씁니다 (FEM-LLM-web 디자인).
- 계산 수식은 [DualSPHysics](https://github.com/DualSPHysics/DualSPHysics) v5.4 를 기준으로 직접 구현하고, 같은 장면을 원본 DualSPHysics 로 돌려 교차검증합니다.
- 단위는 FEM-LLM-web 과 같은 **mm · tonne · s · N · MPa** 입니다.

## 현재 상태 (1단계)

| 항목 | 상태 |
|---|---|
| 화면 (FEM 배치), 예제 3종, OBJ·STL 업로드, 파일 단위 변환 | 됨 |
| 케이스 스키마 v1, 수정 지시 검증, 입자 수 추정, 경고 | 됨 |
| 규칙 해석기 (유체, 입자 간격, 시간, 장치, 중력, 유체 블록, 유입구, 유출 영역, 삭제, 되돌리기, 실행) | 됨 |
| LLM 연결 (OpenAI 호환 엔드포인트 + API 키) | 됨 (모의 서버로 시험, 실제 API 미확인) |
| 계산 장치 확인 (Taichi CUDA · Metal · Vulkan · CPU) + 간이 벤치마크 | 됨 (이 저장소 개발 환경에서는 CPU 만 확인) |
| 결과 형식 v1 (입자 ID 고정 이력) | 됨 |
| 해석 엔진 | 2단계 |

## 설치와 실행 (WSL Ubuntu · Mac 공통)

```bash
git clone https://github.com/MunkyunLEE/Simweb-Core.git
git clone https://github.com/MunkyunLEE/SPH-LLM-web.git
cd SPH-LLM-web
./scripts/setup.sh      # 가상환경 ~/.venvs/sphweb, 옆 폴더의 Simweb-Core 를 함께 설치, 계산 장치 확인
./scripts/run.sh        # http://localhost:8766
```

Python 3.10 ~ 3.13 이 필요합니다. 자세한 내용(사내 프록시·인증서, Windows, GPU 확인)은 [docs/INSTALL.md](docs/INSTALL.md).

## 지시 예시

```
x 0~500, y 0~300, z 0~400 mm 에 물 채워줘
(250, 150, 900) mm 에서 아래로 2 m/s 로 유입, 크기 100x100 mm
(100, 0, 580) mm 에서 아래로 1.5 m/s 로 유입, 지름 40 mm
z -300~-100, x -300~300, y -300~300 mm 유출 영역 만들어줘
입자 간격 5 mm / 해석 시간 3초, 초당 60 프레임 / CPU로 계산해 / 중력 꺼줘
fb1 삭제해줘 / 유입구 전부 삭제 / 되돌려줘 / 해석 실행
```

규칙이 모르는 문장은 LLM 으로 넘어갑니다. LLM 은 화면 오른쪽 위에 엔드포인트, 모델, API 키를 넣으면 연결됩니다 (환경 변수 `SPHWEB_LLM_URL`, `SPHWEB_LLM_MODEL`, `SPHWEB_LLM_KEY`, `SPHWEB_LLM_MODE`).

## 문서

- [docs/ROADMAP.md](docs/ROADMAP.md) 5단계 로드맵과 중간보고 시점
- [docs/VALIDATION.md](docs/VALIDATION.md) 검증 로드맵
- [docs/CASE_SCHEMA.md](docs/CASE_SCHEMA.md) 케이스 스키마와 결과 형식
- [docs/UNITS.md](docs/UNITS.md) 단위계와 GPU 단정밀도 정책
- [docs/reports/](docs/reports/) 단계별 보고서
- 화면 오른쪽 위 Theory 버튼 (`web/theory.html`)

## 시험

```bash
python -m pytest tests -q
```

## 라이선스

MIT. three.js (MIT) 는 Simweb-Core 에 포함되어 있습니다. DualSPHysics (LGPL-2.1) 코드는 포함하지 않으며, 수식은 논문과 문서를 근거로 직접 구현합니다. 코드를 옮겨 와야 하는 부분이 생기면 별도 폴더에 LGPL 로 표시합니다.
