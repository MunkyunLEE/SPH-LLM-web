import pytest

from simweb import geometry
from sphweb import spec as S

OBJ = open(__file__.replace("tests/test_spec.py", "examples/origami_2x2.obj")).read()


def test_default_and_estimate():
    s = S.default_spec()
    s, log = S.apply_ops(s, [{"op": "set_particle", "dx": 10}, {"op": "add_fluid_block", "box": {"min": [0, 0, 0], "max": [400, 600, 300]}}])
    e = S.estimate(s)
    assert e["fill"] == 40 * 60 * 30
    assert e["particle_mass"] == pytest.approx(0.998e-9 * 1000)
    assert s["fluid_blocks"][0]["id"] == "fb1"


def test_inlet_speed_and_update():
    s, _ = S.apply_ops(S.default_spec(), [{"op": "add_inlet", "center": [0, 0, 500], "normal": [0, 0, -2], "speed": 1500, "shape": "circle", "size": [40, 40], "t_end": 2}])
    it = s["inlets"][0]
    assert it["normal"] == [0, 0, -1] and it["velocity"] == [0, 0, -1500]
    s, _ = S.apply_ops(s, [{"op": "update", "id": "in1", "speed": 1000}])
    assert s["inlets"][0]["velocity"] == [0, 0, -1000]
    e = S.estimate(s)
    assert e["emitted"] > 0


def test_validation_errors():
    s = S.default_spec()
    for bad in [{"op": "set_particle", "dx": 0}, {"op": "add_fluid_block", "box": {"min": [0, 0, 0], "max": [0, 1, 1]}}, {"op": "add_inlet", "center": [0, 0, 0], "normal": [0, 0, 0], "speed": 1}, {"op": "set_device", "device": "tpu"}, {"op": "remove", "id": "fb9"}, {"op": "nope"}]:
        with pytest.raises(ValueError):
            S.apply_ops(s, [bad])


def test_bodies_from_geometry_and_role():
    g = geometry.read_obj(OBJ)
    s = S.default_spec(g)
    assert sorted(b["group"] for b in s["bodies"]) == ["panel_1", "panel_2", "panel_3", "panel_4"]
    s, _ = S.apply_ops(s, [{"op": "set_body", "group": "panel_2", "role": "ignore"}], g)
    assert [b["role"] for b in s["bodies"] if b["group"] == "panel_2"] == ["ignore"]
    with pytest.raises(ValueError):
        S.apply_ops(s, [{"op": "set_body", "group": "nope"}], g)


def test_warnings_and_fit():
    s, _ = S.apply_ops(S.default_spec(), [{"op": "add_fluid_block", "box": {"min": [0, 0, 0], "max": [5000, 100, 100]}}])
    assert any("영역 밖" in w["text"] for w in S.warnings_for(s))
    s, _ = S.apply_ops(s, [{"op": "set_domain", "fit": True, "margin": 50}])
    assert s["domain"]["max"][0] == 5050 and not any("영역 밖" in w["text"] for w in S.warnings_for(s))


def test_replace_spec_roundtrip():
    s, _ = S.apply_ops(S.default_spec(), [{"op": "set_particle", "dx": 7}, {"op": "add_outlet", "box": {"min": [0, 0, -10], "max": [1, 1, 0]}}])
    t, _ = S.apply_ops(S.default_spec(), [{"op": "replace_spec", "spec": s}])
    assert t == s


def test_particle_count_fits_block():
    sp = S.default_spec()
    sp, _ = S.apply_ops(sp, [{"op": "add_fluid_block", "box": {"min": [0, 0, 0], "max": [400, 600, 300]}}, {"op": "set_particle", "count": 72000}])
    assert sp["particle"]["dx"] == 10.0
    assert S.estimate(sp)["fill"] == 72000
    sp, _ = S.apply_ops(sp, [{"op": "set_dim", "dim": 2, "plane_y": 300}, {"op": "set_particle", "count": 1200}])
    e = S.estimate(sp)
    assert e["dim"] == 2 and abs(e["fill"] - 1200) / 1200 < 0.15
    assert e["support_radius"] == pytest.approx(2 * 2 ** 0.5 * sp["particle"]["dx"])
