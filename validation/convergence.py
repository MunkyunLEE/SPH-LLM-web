import argparse
import glob
import json
import os
import time

import numpy as np

from sphweb.engine import cases, wcsph
from sphweb.engine.consts import Params
from validation.dambreak import EXP, experiment, tip

CONFIGS = {
    "sym20": dict(scheme="symplectic", coefsound=20.0, mdbc=True),
    "ver20": dict(scheme="verlet", coefsound=20.0, mdbc=True),
    "ver10": dict(scheme="verlet", coefsound=10.0, mdbc=True),
    "ver20dbc": dict(scheme="verlet", coefsound=20.0, mdbc=False),
}


def score(ts, xs, te, xe, col):
    m = te <= ts[-1] + 1e-9
    xi = np.interp(te[m], ts, xs)
    return {"acc_position": float(1 - (np.abs(xi - xe[m]) / xe[m]).mean()), "acc_advance": float(1 - (np.abs(xi - xe[m]) / np.maximum(xe[m] - col, 1e-9)).mean())}


def dam2d(dx, cfg, arch, every=0.01, t_end=0.75):
    wcsph.init(arch)
    L, H, T, Ht = 1000.0, 2000.0, 4000.0, 3000.0
    if cfg["mdbc"]:
        f, b, nr = cases.dam_break2d(T, Ht, L, H, dx, layers=4)
    else:
        f, b = cases.dam_break2d(T, Ht, L, H, dx, layers=1, normals=False)
        nr = None
    p = Params(dp=dx, hswl=float(f[:, 2].max() - f[:, 2].min()), rho0=1e-9, visco=0.02, dim=2, scheme=cfg["scheme"], coefsound=cfg["coefsound"])
    s = wcsph.Solver(p, f, b, bound_normals=nr, cell_div=2)
    s.hydrostatic(f[:, 2].max(), bound=False)
    ts, xs = [0.0], [L + 0.5 * dx]

    def cb(s):
        d = s.snapshot(True)
        ts.append(s.time)
        xs.append(tip(d["pos"][d["kind"] == 1], L, dx))

    t0 = time.time()
    s.run(t_end, every=every, callback=cb)
    te = np.array([e[0] for e in EXP])
    xe = np.array([e[1] for e in EXP]) * 1000.0
    return {"t": ts, "x": xs, "score": score(np.array(ts), np.array(xs), te, xe, L), "steps": s.nstep, "wall": time.time() - t0, "n": s.n, "nfluid": s.nf}


def dam3d(dx, cfg, arch, every=0.005, t_end=0.34):
    wcsph.init(arch)
    col, W, T, Ht = 200.0, 100.0, 800.0, 600.0
    if cfg["mdbc"]:
        f, b, nr = cases.dam_break(T, W, Ht, col, 2 * col, dx, layers=4, normals=True)
    else:
        f, b = cases.dam_break(T, W, Ht, col, 2 * col, dx)
        nr = None
    p = Params(dp=dx, hswl=2 * col, visco=0.02, scheme=cfg["scheme"], coefsound=cfg["coefsound"])
    s = wcsph.Solver(p, f, b, bound_normals=nr, cell_div=2)
    s.hydrostatic(f[:, 2].max(), bound=False)
    ts, xs = [0.0], [col + 0.5 * dx]

    def cb(s):
        d = s.snapshot(True)
        ts.append(s.time)
        xs.append(tip(d["pos"][d["kind"] == 1], col, dx))

    t0 = time.time()
    s.run(t_end, every=every, callback=cb)
    te, xe = experiment(col)
    return {"t": ts, "x": xs, "score": score(np.array(ts), np.array(xs), te, xe, col), "steps": s.nstep, "wall": time.time() - t0, "n": s.n, "nfluid": s.nf}


def hydro2d(dx, cfg, arch, t_end=1.0):
    wcsph.init(arch)
    L, D = 1000.0, 300.0
    if cfg["mdbc"]:
        f, b, nr = cases.tank2d(L, 400.0, D, dx, layers=4)
    else:
        f, b = cases.tank2d(L, 400.0, D, dx, layers=1, normals=False)
        nr = None
    p = Params(dp=dx, hswl=D, dim=2, scheme=cfg["scheme"], coefsound=cfg["coefsound"])
    s = wcsph.Solver(p, f, b, bound_normals=nr, cell_div=2)
    s.hydrostatic(f[:, 2].max(), bound=False)
    t0 = time.time()
    s.run(t_end)
    d = s.snapshot(True)
    z, x = d["pos"][:, 2], d["pos"][:, 0]
    zt = z.max() + 0.5 * dx
    sel = (z > 2 * dx) & (z < zt - 2 * dx) & (x > 3 * dx) & (x < L - 3 * dx)
    ref = p.rho0 * 9810.0 * (zt - z[sel])
    err = float(np.abs(d["press"][sel] - ref).mean() / ref.mean())
    return {"err": err, "score": {"acc_pressure": 1 - err}, "steps": s.nstep, "wall": time.time() - t0, "n": s.n, "nfluid": s.nf, "vmax": float(np.linalg.norm(d["vel"], axis=1).max())}


def dsph2d(root):
    out = {}
    te = np.array([e[0] for e in EXP])
    xe = np.array([e[1] for e in EXP]) * 1000.0
    for d in sorted(glob.glob(os.path.join(root, "dp*"))):
        if not os.path.isdir(d):
            continue
        dp = float(os.path.basename(d)[2:]) * 1000.0
        ts, xs = [0.0], [1000.0 + 0.5 * dp]
        for f in sorted(glob.glob(os.path.join(d, "csv", "fluid_[0-9]*.csv"))):
            with open(f) as fh:
                fh.readline()
                t = float(fh.readline().split(";")[0])
            if t <= 0:
                continue
            pos = np.loadtxt(f, delimiter=";", skiprows=4, usecols=(0, 1, 2)) * 1000.0
            ts.append(t)
            xs.append(tip(pos, 1000.0, dp))
        if len(ts) < 3:
            continue
        run = open(os.path.join(d, "Run.out"), errors="ignore").read()
        steps = int(run.split("Steps of simulation..............:")[1].split()[0].replace(",", "")) if "Steps of simulation" in run else None
        wall = float(run.split("Total Runtime....................:")[1].split()[0]) if "Total Runtime" in run else None
        out[f"{dp:g}"] = {"t": ts, "x": xs, "score": score(np.array(ts), np.array(xs), te, xe, 1000.0), "steps": steps, "wall": wall}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="hydro2d,dam2d,dam3d")
    ap.add_argument("--dx2d", default="40,20,10")
    ap.add_argument("--dx3d", default="20,13.3333,10")
    ap.add_argument("--configs", default="sym20,ver20,ver10,ver20dbc")
    ap.add_argument("--configs3d", default="sym20,ver10")
    ap.add_argument("--dsph2d", default=None)
    ap.add_argument("--arch", default="cpu")
    ap.add_argument("--out", default="runs/convergence")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "result.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    jobs = []
    for w in a.what.split(","):
        dxs = a.dx3d if w == "dam3d" else a.dx2d
        cf = a.configs3d if w == "dam3d" else a.configs
        for dx in [float(v) for v in dxs.split(",")]:
            for c in cf.split(","):
                jobs.append((w, dx, c))
    fn = {"hydro2d": hydro2d, "dam2d": dam2d, "dam3d": dam3d}
    for w, dx, c in jobs:
        key = f"{w}|{dx:g}|{c}"
        if key in res:
            continue
        r = fn[w](dx, CONFIGS[c], a.arch)
        res[key] = r
        print(f"{key:28s} n={r['n']:7d} steps={r['steps']:6d} wall={r['wall']:7.1f}s {r['score']}", flush=True)
        json.dump(res, open(path, "w"), indent=1, default=float)
    if a.dsph2d:
        res["dsph2d"] = dsph2d(a.dsph2d)
        for k, v in res["dsph2d"].items():
            print(f"dsph2d|{k} steps={v['steps']} wall={v['wall']} {v['score']}", flush=True)
        json.dump(res, open(path, "w"), indent=1, default=float)
    print("saved", path)


if __name__ == "__main__":
    main()
