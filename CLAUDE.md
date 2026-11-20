# SPH-LLM-web 개발 안내 (Claude Code 용)

개인용 SPH 유체 시뮬레이터. 형상(OBJ/STL) 업로드 → 문장으로 유체·유입·유출 지정(규칙 해석기 우선, 선택적으로 OpenAI 호환 LLM) → 로컬 Taichi 솔버 → 웹 뷰어.

## 사용자와 작업 방식
- 사용자는 CAE 엔지니어. 대화, 화면 문구, 결과 요약은 한국어 존댓말
- 코드에 주석을 달지 않음. 큰 파일은 통째로 다시 쓰지 말고 필요한 곳만 수정
- 단위 mm · tonne · s · N · MPa (물 밀도 0.998e-9, 중력 9810)
- 정합성 75% 이상이면 충분, 속도와 해석 중 가시성 우선
- 확인하지 못한 것은 확인하지 못했다고 보고
- 중간보고 시점(docs/ROADMAP.md)에서 멈추고 보고. 벤치마크 75% 미달, 설계 갈림길, 플랫폼 문제, 계획 변경은 즉시 보고

## 구조
- 공통 모듈 `simweb` (저장소 Simweb-Core): FastAPI 앱 팩토리(`simweb.app.create_app(plugin)`), `llm`, `geometry`, `select`, `units`, 화면 공통부(`/core/...`)
- `sphweb/server.py`: Plugin (default_spec, apply_ops, rules, llm_context, state, routes) + `/api/engine`, `/api/preview`, `/api/solve`
- `sphweb/spec.py`: 케이스 스키마 v1, `apply_ops`, `estimate`, `warnings_for`. 새 수정 지시는 여기와 `prompt.py` 의 OPS 설명에 같이 추가
- `sphweb/rules.py`: 한국어 규칙 해석기. 새 문장 패턴은 여기에 추가하고 `tests/test_rules.py` 의 CASES 에 시험 추가
- `sphweb/prompt.py`: LLM 시스템 지시문과 문맥
- `sphweb/results.py`: 결과 형식 v1 (`meta.json` + `frames/*.npz`, 입자 ID 고정), `pack` 으로 Rhino 용 단일 npz
- `sphweb/engine/backend.py`: Taichi 장치 확인(하위 프로세스) · 벤치마크. 솔버 커널은 2단계에서 `sphweb/engine/` 에 추가
- `web/`: `index.html`, `app.js`, `style.css`, `theory.html` (기능을 바꾸면 같이 갱신)
- `examples/make_examples.py` → `presets.json` 과 예제 형상. 예제를 바꾸면 `tests/test_api.py::test_examples_listed` 갱신

## 계산 코어 원칙 (2단계부터)
- 수식은 DualSPHysics v5.4 기준 (WCSPH, Wendland, δ-SPH, mDBC, Symplectic, shifting). 코드를 옮기지 않고 수식으로 구현. 부득이하게 옮기면 `sphweb/engine/lgpl/` 에 LGPL 표시
- 커널(Taichi)과 나머지(케이스, 입출력, 서버)를 분리해 백엔드를 바꿀 수 있게 유지
- GPU 내부는 ρ/ρ₀, 입자 부피로 정규화 (float32 아래넘침 방지), 입출력에서만 단위 변환
- 같은 장면을 원본 DualSPHysics 로 돌려 비교 (docs/VALIDATION.md)

## 실행
- `./scripts/setup.sh` (가상환경 `~/.venvs/sphweb`, 옆 폴더 `../Simweb-Core` 를 editable 설치), `./scripts/run.sh` → `http://localhost:8766`
- 시험: `python -m pytest tests -q` (약 1초)

## 커밋
- 단계 작업은 `stage<N>/...` 브랜치, 커밋 메시지는 한국어 한 줄 요약 + 항목
- 수정 후 관련 시험을 돌리고 결과를 함께 보고
