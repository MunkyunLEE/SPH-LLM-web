import json
import os
import struct
import time

import numpy as np
import pytest

pytest.importorskip("taichi")

from simweb import geometry
from sphweb import spec as S
from sphweb.runner import Runner

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def _spec(t_end):
    g = geometry.load("dam_tank.obj", open(os.path.join(EX, "dam_tank.obj"), "rb").read(), "mm")
    pre = json.load(open(os.path.join(EX, "presets.json"), encoding="utf-8"))["dam_break"]
    sp, _ = S.apply_ops(S.default_spec(g), pre["ops"], g)
    sp, _ = S.apply_ops(sp, [{"op": "set_particle", "dx": 40}, {"op": "set_solver", "t_end": t_end, "fps": 200}, {"op": "add_probe", "point": [200, 300, 60]}], g)
    return sp, g


def _wait(r, cond, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = r.status()
        if cond(st):
            return st
        time.sleep(0.2)
    raise TimeoutError(r.status())


def _decode(b):
    hl = struct.unpack("<I", b[:4])[0]
    head = json.loads(b[4 : 4 + hl])
    off = 4 + hl
    out = {}
    for name, n, t in head["arrays"]:
        off += (-off) % 4
        dt = np.uint8 if t == "u8" else np.float32
        out[name] = np.frombuffer(b, dt, n, off)
        off += n * np.dtype(dt).itemsize
    return head, out


def test_run_frames_and_controls(tmp_path):
    sp, g = _spec(0.03)
    r = Runner(str(tmp_path))
    r.start(sp, g, "cpu")
    st = _wait(r, lambda s: s["state"] in ("done", "error"))
    assert st["state"] == "done", st["message"]
    assert st["frames"] >= 6
    assert len(st["series"]["time"]) == st["frames"]
    assert st["series"]["density.error.max"][-1] < 0.05
    fz = st["series"]["force.tank.z"]
    weight = st["n"] * sp["fluid"]["rho0"] * 40.0 ** 3 * 9810.0
    assert fz[-1] < 0 and abs(fz[-1]) < 3 * weight
    head, a = _decode(r.frame_bytes(-1))
    assert head["n"] == st["n"] and a["pos"].size == 3 * head["n"]
    assert a["gu"].size == 3 * int(np.prod(head["grid"]["n"]))
    assert a["gocc"].sum() > 0
    assert len(st["series"][f"probe.{sp['monitors'][0]['id']}.pressure"]) == st["frames"]
    from sphweb import results

    last = results.Reader(st["dir"]).frame(st["frames"] - 1)
    p = last["positions"][last["active"].astype(bool)]
    assert p[:, 2].min() > 0 and p[:, 0].min() > 0 and p[:, 1].min() > 0

    sp2, _ = _spec(5.0)
    r.start(sp2, g, "cpu")
    _wait(r, lambda s: s["state"] == "running" and s["frames"] >= 1)
    r.control("pause")
    st = _wait(r, lambda s: s["state"] == "paused")
    n0, t0 = st["step"], st["t"]
    r.control("step")
    st = _wait(r, lambda s: s["step"] == n0 + 1)
    assert st["t"] > t0
    r.control("stop")
    st = _wait(r, lambda s: s["state"] == "stopped")
    assert st["t"] < 5.0


def test_run_2d_grid_on_plane(tmp_path):
    sp, g = _spec(0.02)
    sp, _ = S.apply_ops(sp, [{"op": "set_dim", "dim": 2, "plane_y": 300}, {"op": "set_particle", "dx": 20}, {"op": "set_solver", "scheme": "verlet"}], g)
    r = Runner(str(tmp_path))
    r.start(sp, g, "cpu")
    st = _wait(r, lambda s: s["state"] in ("done", "error"))
    assert st["state"] == "done", st["message"]
    head, a = _decode(r.frame_bytes(-1))
    assert head["grid"]["n"][1] == 1
    pos = a["pos"].reshape(-1, 3)
    assert np.allclose(pos[:, 1], head["grid"]["origin"][1])
    assert a["gocc"].sum() > 50

