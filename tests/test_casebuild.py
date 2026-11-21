import json
import os

import numpy as np

from simweb import geometry
from sphweb import casebuild
from sphweb import spec as S

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def _cube():
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) * 100
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return V, F


def test_inside_and_closest_cube():
    V, F = _cube()
    pts = casebuild.lattice([-50, -50, -50], [150, 150, 150], 10)
    ins = casebuild.inside(pts, V, F)
    want = np.all((pts > 0) & (pts < 100), 1)
    assert np.array_equal(ins, want)
    cp, d, _ = casebuild.closest(np.array([[50.0, 50.0, 5.0], [150.0, 50.0, 50.0], [120.0, 130.0, 50.0]]), V, F)
    assert np.allclose(d, [5.0, 50.0, np.hypot(20, 30)])
    assert np.allclose(cp[0], [50, 50, 0])


def _dam():
    g = geometry.load("dam_tank.obj", open(os.path.join(EX, "dam_tank.obj"), "rb").read(), "mm")
    pre = json.load(open(os.path.join(EX, "presets.json"), encoding="utf-8"))["dam_break"]
    sp, _ = S.apply_ops(S.default_spec(g), pre["ops"], g)
    sp, _ = S.apply_ops(sp, [{"op": "set_particle", "dx": 20}], g)
    return sp, g


def test_dam_tank_case():
    sp, g = _dam()
    r = casebuild.build(sp, g)
    f, b, n = r["fluid"], r["bound"], r["normals"]
    assert len(f) == 20 * 30 * 15
    assert np.allclose(f.min(0), [10, 10, 10]) and np.allclose(f.max(0), [390, 590, 290])
    floor = (b[:, 2] < 0) & (b[:, 0] > 100) & (b[:, 0] < 1500) & (b[:, 1] > 100) & (b[:, 1] < 500)
    assert floor.any()
    assert np.all(n[floor, 2] > 0) and np.allclose(n[floor, :2], 0)
    ghost = b[floor] + n[floor]
    assert np.allclose(ghost[:, 2], -b[floor, 2])
    assert np.unique(b[floor, 2]).tolist() == [-70.0, -50.0, -30.0, -10.0]
    assert np.linalg.norm(n, axis=1).min() > 0
    cav = (b[:, 0] > 0) & (b[:, 0] < 1600) & (b[:, 1] > 0) & (b[:, 1] < 600) & (b[:, 2] > 0)
    assert not cav.any()
    assert r["group_names"] == ["tank"] and set(r["group"]) == {0}


def test_fluid_without_geometry():
    sp = S.default_spec()
    sp, _ = S.apply_ops(sp, [{"op": "add_fluid_block", "box": {"min": [0, 0, 0], "max": [100, 100, 100]}}, {"op": "set_particle", "dx": 20}])
    r = casebuild.build(sp, None)
    assert len(r["fluid"]) == 125 and len(r["bound"]) == 0


def _preset(name, extra=()):
    pre = json.load(open(os.path.join(EX, "presets.json"), encoding="utf-8"))[name]
    g = geometry.load(pre["file"], open(os.path.join(EX, pre["file"]), "rb").read(), "mm")
    sp, _ = S.apply_ops(S.default_spec(g), pre["ops"] + list(extra), g)
    return sp, g


def test_still_tank_ghosts_all_in_cavity():
    sp, g = _preset("still_tank")
    r = casebuild.build(sp, g)
    gh = r["bound"] + r["normals"]
    assert np.all(((gh[:, 0] > 0) & (gh[:, 0] < 400) & (gh[:, 1] > 0) & (gh[:, 1] < 400) & (gh[:, 2] > 0)) | (gh[:, 2] > 400))


def test_obstacle_groups_and_drop_blocks():
    sp, g = _preset("dambreak_obstacle", [{"op": "set_particle", "dx": 25}])
    r = casebuild.build(sp, g)
    gh = r["bound"] + r["normals"]
    ins = casebuild.make_inside(np.asarray(g.vertices, float), g.triangles()[0])
    assert not ins(gh).any()
    assert set(r["group_names"]) == {"tank", "building"} and len(set(r["group"])) == 2
    b = r["fluid"]
    assert not ((b[:, 0] > 900) & (b[:, 0] < 1020) & (b[:, 1] > 240) & (b[:, 1] < 360)).any()
    sp, g = _preset("drop_pool")
    r = casebuild.build(sp, g)
    assert set(np.unique(r["zsurf"]).tolist()) == {-np.inf, 143.75}


def test_solid_mask_marks_wall_interior():
    sp, g = _preset("still_tank", [{"op": "set_particle", "dx": 10}])
    r = casebuild.build(sp, g)
    m = r["solid"]
    idx = lambda p: tuple(int(round((p[k] - m["origin"][k]) / m["h"])) for k in range(3))
    assert m["mask"][idx([200, 200, -15])]
    assert not m["mask"][idx([200, 200, -5])]
    assert not m["mask"][idx([200, 200, 100])]
