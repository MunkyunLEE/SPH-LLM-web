import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from sphweb.engine import cases, wcsph
from sphweb.engine.consts import Params, wendland, wendland_fac


@pytest.fixture(scope="module", autouse=True)
def _ti():
    wcsph.init("cpu")


def test_constants_match_dualsphysics():
    p = Params(dp=0.02, hswl=0.3, rho0=998.0, gravity=(0, 0, -9.81))
    assert p.h == pytest.approx(0.034641016, rel=1e-6)
    assert p.cs0 == pytest.approx(34.3103475174551, rel=1e-7)
    assert p.dt_min == pytest.approx(5.048187980791899e-05, rel=1e-7)
    assert p.bwen == pytest.approx(-4.187614e07, rel=1e-5)
    mm = Params(dp=20.0, hswl=300.0)
    assert mm.cs0 == pytest.approx(p.cs0 * 1000, rel=1e-9)
    assert mm.h == pytest.approx(p.h * 1000, rel=1e-9)


def test_kernel_normalization_and_gradient():
    dp = 1.0
    p = Params(dp=dp, hswl=1.0)
    g = np.arange(-6, 7) * dp
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    r = np.linalg.norm(X, axis=1)
    assert (p.vol * wendland(r, p.h)).sum() == pytest.approx(1.0, abs=5e-3)
    dr = -X
    fr = wendland_fac(r, p.h)[:, None] * dr
    m = -(p.vol * dr[:, :, None] * fr[:, None, :]).sum(0)
    assert np.allclose(m, np.eye(3), atol=1e-2)
    rr = np.linspace(1e-4, 2 * p.h, 4000)
    w = wendland(rr, p.h)
    num = np.gradient(w, rr)
    assert np.allclose(wendland_fac(rr, p.h)[50:-50] * rr[50:-50], num[50:-50], rtol=1e-3, atol=1e-6 * abs(num).max())


def test_neighbor_search_equals_brute_force():
    rng = np.random.default_rng(1)
    dp = 10.0
    pts = rng.uniform(0, 120, (900, 3))
    p = Params(dp=dp, hswl=100.0)
    s = wcsph.Solver(p, pts[300:], pts[:300])
    got = s.neighbor_counts()
    allp = np.vstack([pts[:300], pts[300:]])
    d2 = ((allp[:, None, :] - allp[None, :, :]) ** 2).sum(-1)
    want = ((d2 <= p.support ** 2) & (d2 > 1e-18)).sum(1)
    assert np.array_equal(got, want)


def test_tank_counts_match_gencase():
    f, b = cases.tank(400, 400, 400, 300, 20)
    assert len(f) == 5415 and len(b) == 2041


def test_mm_and_si_agree():
    out = {}
    for name, sc, g, rho0 in (("mm", 1.0, 9810.0, 0.998e-9), ("si", 1e-3, 9.81, 998.0)):
        dp = 25.0 * sc
        f, b = cases.tank(150 * sc, 150 * sc, 300 * sc, 150 * sc, dp)
        p = Params(dp=dp, hswl=150 * sc, rho0=rho0, gravity=(0.0, 0.0, -g))
        s = wcsph.Solver(p, f, b, pos64=True)
        s.hydrostatic(f[:, 2].max(), bound=False)
        for _ in range(40):
            s.step()
        out[name] = (s.time, s.snapshot(True))
    (tm, a), (ts, c) = out["mm"], out["si"]
    assert tm * 1.0 == pytest.approx(ts, rel=1e-5)
    assert np.abs(a["press"] * 1e6 - c["press"]).max() <= 1e-3 * np.abs(c["press"]).max()
    assert np.abs(a["pos"] * 1e-3 - c["pos"]).max() <= 1e-5 * 0.025
    assert np.isfinite(a["press"]).all()


@pytest.mark.parametrize("pos64", [True, False])
def test_hydrostatic_short_stays_stable(pos64):
    f, b = cases.tank(200, 200, 300, 150, 25)
    p = Params(dp=25.0, hswl=150.0)
    s = wcsph.Solver(p, f, b, pos64=pos64)
    s.hydrostatic(f[:, 2].max(), bound=False)
    for _ in range(100):
        s.step()
    d = s.snapshot(True)
    assert s.info()["out"] == 0
    assert np.isfinite(d["press"]).all()
    assert np.linalg.norm(d["vel"], axis=1).max() < 200.0
    assert d["pos"][:, 2].min() > 0.0
