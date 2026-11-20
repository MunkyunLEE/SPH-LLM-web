import json
import sys
import time

from sphweb.engine import cases, wcsph
from sphweb.engine.consts import Params


def bench(name, f, b, nr, p, n=128):
    s = wcsph.Solver(p, f, b, bound_normals=nr, cell_div=2)
    s.hydrostatic(f[:, 2].max(), bound=False)
    s.advance(8)
    t = time.perf_counter()
    k = 0
    while k < n:
        k += s.advance(min(64, n - k))
    ms = (time.perf_counter() - t) / n * 1000
    print(f"{name:34s} fluid={s.nf:7d} bound={s.nb:7d} {ms:7.2f} ms/step", flush=True)
    return {"nfluid": s.nf, "nbound": s.nb, "ms_step": ms}


def main(arch="cpu", out="runs/convergence/speed.json"):
    wcsph.init(arch)
    res = {}
    for dx in (40.0, 20.0, 10.0):
        f, b = cases.dam_break2d(4000, 3000, 1000, 2000, dx, layers=1, normals=False)
        res[f"dam2d|{dx:g}|ver20dbc"] = bench(f"2D dam dx {dx:g} Verlet DBC", f, b, None, Params(dp=dx, hswl=2000 - dx, rho0=1e-9, visco=0.02, dim=2, scheme="verlet"))
        f, b, nr = cases.dam_break2d(4000, 3000, 1000, 2000, dx, layers=4)
        res[f"dam2d|{dx:g}|sym20"] = bench(f"2D dam dx {dx:g} Symplectic mDBC", f, b, nr, Params(dp=dx, hswl=2000 - dx, rho0=1e-9, visco=0.02, dim=2))
    for dx in (20.0, 10.0):
        f, b, nr = cases.dam_break(800, 100, 600, 200, 400, dx, layers=4, normals=True)
        res[f"dam3d|{dx:g}|sym20"] = bench(f"3D dam dx {dx:g} Symplectic mDBC", f, b, nr, Params(dp=dx, hswl=400, visco=0.02))
        res[f"dam3d|{dx:g}|ver10"] = bench(f"3D dam dx {dx:g} Verlet c10 mDBC", f, b, nr, Params(dp=dx, hswl=400, visco=0.02, scheme="verlet", coefsound=10))
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
