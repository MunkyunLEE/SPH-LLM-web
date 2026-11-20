import os

import numpy as np
from fastapi.testclient import TestClient

from sphweb import results
from sphweb.server import app

HERE = os.path.dirname(os.path.abspath(__file__))
c = TestClient(app)


def test_pages():
    assert c.get("/").status_code == 200
    assert c.get("/app.js").status_code == 200
    assert c.get("/core/js/panel.js").status_code == 200
    assert c.get("/core/js/fields.js").status_code == 200
    assert c.get("/results.js").status_code == 200
    assert c.get("/theory.html").status_code == 200


def test_examples_listed_and_load():
    names = [e["name"] for e in c.get("/api/examples").json()]
    assert names == ["dam_break", "funnel_jet", "origami_panels"]
    s = c.post("/api/examples/dam_break").json()
    assert s["model"]["closed"] and s["estimate"]["fill"] == 40 * 60 * 30
    p = c.get("/api/preview").json()
    assert len(p["points"]) > 1000
    s = c.post("/api/examples/funnel_jet").json()
    assert s["spec"]["inlets"][0]["shape"] == "circle" and s["spec"]["outlets"]


def test_chat_rules_and_undo():
    c.post("/api/examples/dam_break?preset=0")
    s = c.post("/api/chat", json={"text": "x 0~400, y 0~600, z 0~300 mm 에 물 채워줘"}).json()
    assert s["chat"]["engine"] == "rule" and len(s["spec"]["fluid_blocks"]) == 1
    s = c.post("/api/chat", json={"text": "되돌려줘"}).json()
    assert len(s["spec"]["fluid_blocks"]) == 0


def test_upload_units():
    data = open(os.path.join(HERE, "..", "examples", "funnel.stl"), "rb").read()
    s = c.post("/api/upload", files={"file": ("funnel.stl", data)}, data={"unit": "m"}).json()
    assert np.isclose(s["model"]["bbox"][1][2], 460000.0)


def test_run_endpoints_idle():
    assert c.get("/api/run").json()["state"] in ("idle", "done", "error", "stopped")
    assert c.post("/api/run/pause").status_code == 409
    assert c.get("/api/run/frame/0").status_code in (404, 200)


def test_results_roundtrip(tmp_path):
    n = 5
    w = results.Writer(str(tmp_path / "run"), n, {"case": "t"})
    for k in range(3):
        a = np.zeros(n, bool)
        a[: k + 2] = True
        w.add(0.1 * k, 10 * k, positions=np.full((n, 3), k, np.float32), velocity=np.zeros((n, 3)), pressure=np.zeros(n), density=np.ones(n), active=a)
    r = results.Reader(str(tmp_path / "run"))
    assert len(r) == 3 and r.meta["frames"][2]["active"] == 4
    tr = r.trajectory([0, 4])
    assert np.isnan(tr[0, 1]).all() and tr[2, 0, 0] == 2
    out = results.pack(str(tmp_path / "run"), str(tmp_path / "run.npz"))
    d = np.load(out)
    assert d["positions"].shape == (3, n, 3) and d["active"].sum() == 2 + 3 + 4
