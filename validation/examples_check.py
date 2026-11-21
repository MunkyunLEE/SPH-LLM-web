import argparse
import json
import os
import time

import numpy as np

from simweb import geometry
from sphweb import results
from sphweb import spec as S
from sphweb.runner import Runner

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(name, extra, out):
    pre = json.load(open(os.path.join(HERE, "examples", "presets.json"), encoding="utf-8"))[name]
    g = geometry.load(pre["file"], open(os.path.join(HERE, "examples", pre["file"]), "rb").read(), "mm")
    sp, _ = S.apply_ops(S.default_spec(g), pre["ops"] + extra, g)
    r = Runner(out)
    r.start(sp, g, "cpu")
    t0 = time.time()
    while r.status()["state"] in ("building", "running"):
        time.sleep(1)
    st = r.status()
    rd = results.Reader(st["dir"])
    last = rd.frame(len(rd) - 1)
    act = last["active"].astype(bool)
    p = last["positions"][act]
    s = st["series"]
    res = {"state": st["state"], "message": st["message"], "nfluid": st["n"], "nbound": st["nbound"], "steps": st["step"], "wall": st["wall"], "out": st.get("out", 0), "dx": sp["particle"]["dx"], "dim": sp.get("dim", 3), "pmin": p.min(0).tolist(), "pmax": p.max(0).tolist(), "rho_err_mean_max": max(s["density.error.mean"]), "rho_err_max_max": max(s["density.error.max"]), "time": s["time"], "probes": {}, "forces": {}}
    for m in st["probes"]:
        res["probes"][m["name"]] = {"point": m["point"], "pressure": s[f"probe.{m['id']}.pressure"], "speed": s[f"probe.{m['id']}.speed"]}
    for gname in st["groups"]:
        res["forces"][gname] = {c: s[f"force.{gname}.{c}"] for c in "xyz"}
    print(f"[{name}] {st['state']} fluid={st['n']} bound={st['nbound']} steps={st['step']} wall={st['wall']:.0f}s rho_err mean<= {100 * res['rho_err_mean_max']:.2f}% max<= {100 * res['rho_err_max_max']:.1f}% fluid bbox {np.round(p.min(0))}..{np.round(p.max(0))} ({time.time() - t0:.0f}s)", flush=True)
    for k, v in res["probes"].items():
        print(f"   probe {k} {v['point']}: p_end={v['pressure'][-1]:.6f} MPa, p_max={max(v['pressure']):.6f}, speed_end={v['speed'][-1]:.1f} mm/s", flush=True)
    for k, v in res["forces"].items():
        print(f"   force {k}: Fx max={max(v['x']):.2f} N, Fz end={v['z'][-1]:.2f} N", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="still_tank,dambreak_obstacle,drop_pool")
    ap.add_argument("--dx", type=float, default=None)
    ap.add_argument("--t", type=float, default=None)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--out", default="runs/examples")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allres = {}
    path = os.path.join(a.out, "result.json")
    if os.path.exists(path):
        allres = json.load(open(path))
    for n in a.names.split(","):
        extra = []
        if a.dx:
            extra.append({"op": "set_particle", "dx": a.dx})
        if a.t:
            extra.append({"op": "set_solver", "t_end": a.t})
        if a.dim == 2:
            extra.append({"op": "set_dim", "dim": 2})
        key = f"{n}|{a.dx or 'default'}|{a.dim}d|{'f32' if os.environ.get('SPHWEB_FORCE_F32') else 'f64'}"
        allres[key] = run(n, extra, os.path.join(a.out, "runs"))
        json.dump(allres, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
