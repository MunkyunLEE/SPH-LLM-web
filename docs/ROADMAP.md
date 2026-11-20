# 로드맵 (2026-11-20 확정)

목표: SPH 기반 유체 해석 전반. 격자법이 약한 자유수면·대변형·FSI·범람을 빠르게, 해석 중에도 보면서 푼다. 정합성 75% 이상, 속도와 가시성 우선.

## 원칙
- 화면 디자인은 FEM-LLM-web, 결과 표시 항목은 lbm-flow-lab (속도·압력·와도·벡터·단면·유선·Q-criterion·프로브·하중)
- 공통 화면·LLM·지오메트리는 Simweb-Core. 앱끼리 같은 지오메트리를 공유하는 것이 최종 목표
- 계산은 로컬 (Taichi: WSL·Windows CUDA, Mac Metal, CPU). 개발·검토는 Linux(WSL Ubuntu) 기준
- 수식은 DualSPHysics v5.4 기준, 원본 DualSPHysics 와 교차검증
- 단위 mm · tonne · s · N · MPa
- LLM 은 API 키로 연결 (로컬 LLM 설치 불필요), 기본 규칙은 계속 추가

## 단계와 중간보고

| 단계 | 내용 | 완료 기준 | 보고 |
|---|---|---|---|
| 1. 기반 | Simweb-Core, 화면, 케이스 스키마 v1, 규칙, LLM 연결, 결과 형식 v1, 계산 장치 확인 | 화면이 FEM 과 같고, STL·OBJ 가 보이고, LLM 응답 테스트 통과 | 보고 1 — 완료 |
| 2. 3D 유체 코어 | GPU 이웃 탐색, WCSPH (Tait, Wendland, δ-SPH, 인공·층류 점성), mDBC, Symplectic, 실시간 화면, 일시정지·한 스텝, LBM 식 결과 UI | 정수압 오차 5% 이내, 댐 붕괴 75% 이상, 원본 DualSPHysics 와 비교 | 중간 2a (정수압·단위 일치), 보고 2 — 완료 |
| 3. 형상·조건·LLM | STL·OBJ → 경계 입자, 유체 채우기, 유입·유출, 조건부 정지, 스냅샷·재시작, Rhino·VTK 내보내기, 규칙 확장 | 3D 깔때기 배수 75% 이상, 화면·문장 양쪽으로 같은 장면 | 보고 3 |
| 4. FSI | 4a 강체 + 직접 만든 힌지, 4b 동적 Bar-and-Hinge (FEM-LLM-web `bhweb`) 에 유체력 전달 | 부력·쐐기 낙하·힌지 판, 공기 중 BH 진동 = BH 모달, 수중 부가질량 | 중간 4a (부력), 보고 4 |
| 5. SWE-SPH | 지형 높이장, 천수 방정식 입자, Manning 마찰, 젖음/마름, 입자 분할·병합, 범람 | 정지 호수, Stoker·Ritter, Thacker, 영국 환경청 벤치마크 | 세부 항목마다 |

## 보고 방식
- 단계 끝 5번 + 중간 2번 (2a, 4a)
- 즉시 보고: 벤치마크 75% 미달, 결정이 필요한 갈림길, 플랫폼 문제, 계획 변경
- 내용: 한 일, 벤치마크 표(정확도·속도), 스크린샷, WSL·Mac 에서 돌려 볼 명령, 결정 필요 사항
- 기록: `docs/reports/stageN.md`

## 보류
- OpenFOAM + solids4foam: 기준 해석용, 시뮬레이터가 갖춰진 뒤
- LBM 계열은 별도 프로젝트 (lbm-flow-lab)
