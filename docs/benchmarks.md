# 벤치마크 이력

| 날짜 | 벤치마크 | 조건 | 지표 | 결과 | 원본 DualSPHysics | 장치 | 명령 |
|---|---|---|---|---|---|---|---|
| 2026-11-20 | 정수압 탱크 | 400×400 mm, 수심 300, dp 20, 1.2 s, DBC | 내부 평균 \|P−ρgh\|/ρgh | 2.6 % | 6.1 % (스텝 수 5,945 동일) | CPU 2코어 | `python -m validation.hydrostatic --dsph <csv>` |
| 2026-11-20 | 정수압 탱크 | 같은 조건, mDBC (경계 4층) | 같음 | 1.1 % | — | CPU 2코어 | `python -m validation.hydrostatic --mdbc --skip-unit` |
| 2026-11-20 | 단위 일치 | 같은 탱크 SI 대 mm·tonne, 300 스텝 | 압력 상대 차이 | 2.8e-4 | — | CPU | 위 명령에 포함 |
| 2026-11-20 | 3D 댐 붕괴 (Koshizuka & Oka) | 물기둥 200×400, 폭 100, 탱크 800, dp 10, α 0.02, DBC | 선단 위치 정합도 | 89.7 % | 89.7 % (선단 차이 평균 0.03 mm) | CPU 2코어 | `python -m validation.dambreak --dsph <csv>` |
| 2026-11-20 | 3D 댐 붕괴 | 같은 조건, mDBC | 선단 위치 정합도 | 95.2 % | — | CPU 2코어 | 같음 |
| 2026-11-20 | 2D 정수압 수렴 | 폭 1000, 수심 300, dx 40·20·10, 1 s | 내부 \|P−ρgh\| | 98.9 / 98.8 / 98.9 % (Symplectic·mDBC) | — | CPU 2코어 | `python -m validation.convergence --what hydro2d` |
| 2026-11-20 | 2D 댐 붕괴 수렴 (Koshizuka & Oka) | 물기둥 1×2 m, dx 40·20·10, Verlet·DBC | 선단 위치 정합도 | 92.2 / 90.6 / 89.5 % | 92.3 / 90.6 / 89.5 % | CPU 2코어 | `python -m validation.convergence --what dam2d --dsph2d <폴더>` |
| 2026-11-20 | 3D 댐 붕괴 수렴 | dx 20·13.3·10, Symplectic·mDBC | 선단 위치 정합도 | 95.5 / 95.4 / 95.2 % | — | CPU 2코어 | `python -m validation.convergence --what dam3d` |
