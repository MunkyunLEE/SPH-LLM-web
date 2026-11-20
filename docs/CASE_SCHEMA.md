# 케이스 스키마 v1

`sphweb/spec.py` 의 `default_spec()` 이 기본값입니다. 화면의 "케이스 JSON" 으로 내보내고 "케이스 불러오기" 로 다시 읽습니다.

```json
{
  "version": 1,
  "units": "mm · tonne · s · N · MPa",
  "domain": {"min": [x, y, z], "max": [x, y, z]},
  "fluid": {"preset": "water", "name": "물 20°C", "rho0": 9.98e-10, "mu": 1.002e-9},
  "particle": {"dx": 10.0},
  "gravity": [0, 0, -9810],
  "solver": {"method": "wcsph", "t_end": 2.0, "fps": 30, "cfl": 0.4, "tol_density": 0.05, "tol_divergence": 0.1, "max_iter": 100, "viscosity": "artificial", "alpha": 0.02},
  "device": "auto",
  "bodies": [{"group": "tank", "role": "wall", "motion": "fixed"}],
  "fluid_blocks": [{"id": "fb1", "name": "물기둥", "box": {"min": [...], "max": [...]}}],
  "inlets": [{"id": "in1", "name": "제트", "center": [...], "normal": [0, 0, -1], "velocity": [0, 0, -1500], "speed": 1500, "shape": "circle", "size": [40, 40], "t_start": 0, "t_end": 2.4}],
  "outlets": [{"id": "out1", "name": "바닥 배출", "box": {...}}],
  "monitors": [],
  "stops": [{"id": "st1", "quantity": "density.error.max", "cmp": ">", "value": 0.03, "action": "pause"}],
  "plots": []
}
```

## 수정 지시 (ops)

| op | 필드 |
|---|---|
| set_fluid | preset 또는 rho0, mu |
| set_particle | dx |
| set_solver | t_end, fps, cfl, method, alpha, tol_density, tol_divergence, max_iter |
| set_device | device: auto, cuda, metal, vulkan, cpu |
| set_gravity | vector |
| set_domain | min, max 또는 fit, margin |
| add_fluid_block | name, box |
| add_inlet | name, center, normal, speed 또는 velocity, shape, size, t_start, t_end |
| add_outlet | name, box |
| set_body | group, role: wall, ignore |
| add_plot | title, x, series (quantity) |
| add_stop | quantity, cmp, value, action |
| update / remove / clear | id / id / what |
| replace_spec | spec (케이스 불러오기) |

## 형상

- OBJ: `v`, `f`(면), `l`(선), `g`/`o`(그룹). `hinge` 로 시작하는 그룹의 선은 접힘선 (FEM-LLM-web 과 같은 규칙)
- STL: 바이너리·ASCII. 떨어진 덩어리는 `body_1`, `body_2` … 그룹
- 같은 위치의 정점은 합칩니다. 파일 단위(mm, cm, m, inch)를 골라 mm 로 바꿉니다
- 형상 그룹마다 `bodies` 항목이 생기고 기본 역할은 벽입니다

## 결과 형식 v1 (`sphweb/results.py`)

```
run/
  meta.json        format, version, units, capacity(입자 수 상한), frames[{i, t, step, file, active}]
  frames/000000.npz  positions[N,3] f32, velocity[N,3] f32, pressure[N] f32, density[N] f32, active[N] bool
  walls.npz        경계 입자
```

- 입자 i 는 처음부터 끝까지 i 번째 행. 아직 유입되지 않았거나 제거된 입자는 `active` 만 꺼짐
- `results.pack(run, out.npz)` 는 Rhino·Grasshopper 용 단일 npz (positions[F,N,3] 등)
