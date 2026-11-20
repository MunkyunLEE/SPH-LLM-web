import pytest

from sphweb import rules
from sphweb import spec as S

CASES = [
    ("물로 바꿔줘", [{"op": "set_fluid", "preset": "water"}]),
    ("공기로 설정해줘", [{"op": "set_fluid", "preset": "air"}]),
    ("입자 간격 5 mm로 해줘", [{"op": "set_particle", "dx": 5.0}]),
    ("해상도 1 cm", [{"op": "set_particle", "dx": 10.0}]),
    ("해석 시간 3초, 초당 60 프레임", [{"op": "set_solver", "t_end": 3.0}, {"op": "set_solver", "fps": 60}]),
    ("x 0~500, y 0~300, z 0~400 mm 영역에 물 채워줘", [{"op": "add_fluid_block", "box": {"min": [0.0, 0.0, 0.0], "max": [500.0, 300.0, 400.0]}}]),
    ("x 0~0.5 m, y 0~0.3 m, z 0~0.4 m 에 물 채워줘", [{"op": "add_fluid_block", "box": {"min": [0.0, 0.0, 0.0], "max": [500.0, 300.0, 400.0]}}]),
    ("(250, 150, 900) mm 에서 아래로 2 m/s 로 유입, 크기 100x100 mm", [{"op": "add_inlet", "center": [250.0, 150.0, 900.0], "normal": [0, 0, -1], "speed": 2000.0, "shape": "rect", "size": [100.0, 100.0]}]),
    ("(100, 0, 580) mm 에서 아래로 1.5 m/s 로 유입, 지름 40 mm", [{"op": "add_inlet", "center": [100.0, 0.0, 580.0], "normal": [0, 0, -1], "speed": 1500.0, "shape": "circle", "size": [40.0, 40.0]}]),
    ("z -200~0, x -100~600, y -100~400 mm 유출 영역 만들어줘", [{"op": "add_outlet", "box": {"min": [-100.0, -100.0, -200.0], "max": [600.0, 400.0, 0.0]}}]),
    ("fb1 삭제해줘", [{"op": "remove", "id": "fb1"}]),
    ("유입구 전부 삭제", [{"op": "clear", "what": "inlets"}]),
    ("중력 꺼줘", [{"op": "set_gravity", "vector": [0, 0, 0]}]),
    ("CPU로 계산해", [{"op": "set_device", "device": "cpu"}]),
    ("밀도 1000 kg/m3", [{"op": "set_fluid", "rho0": pytest.approx(1e-9)}]),
    ("해석 영역 맞춰줘", [{"op": "set_domain", "fit": True}]),
    ("DBC로 바꿔줘", [{"op": "set_solver", "boundary": "dbc"}]),
    ("mDBC 경계로 해줘", [{"op": "set_solver", "boundary": "mdbc"}]),
    ("층류 점성으로 바꿔", [{"op": "set_solver", "viscosity": "laminar"}]),
    ("인공 점성 0.05로 해줘", [{"op": "set_solver", "viscosity": "artificial", "alpha": 0.05}]),
    ("(1000, 300, 40) mm 에 프로브 추가해줘", [{"op": "add_probe", "point": [1000.0, 300.0, 40.0]}]),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_rules_complete(text, expected):
    ops, reply, info = rules.parse(text)
    assert ops == expected
    assert info["complete"], info
    S.apply_ops(S.default_spec(), [o for o in ops if o["op"] not in ("remove",)])


def test_ask_when_missing():
    ops, reply, info = rules.parse("아래로 유입시켜줘")
    assert not ops and info["ask"] and not info["complete"]


def test_unknown_goes_to_llm():
    ops, reply, info = rules.parse("날개 각도를 바꿔줘")
    assert info["unknown"] and not info["complete"]


def test_run_and_undo():
    assert rules.parse("해석 실행")[2]["action"] == "solve"
    assert rules.parse("되돌려줘")[2]["undo"]
