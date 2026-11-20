import argparse
import glob
import json
import os
import time

import numpy as np

from sphweb.engine import cases, wcsph
from sphweb.engine.consts import Params


def metrics(pos, press, dp, size, scale=1.0, rho0=0.998e-9, g=9810.0):
    z, x, y = pos[:, 2], pos[:, 0], pos[:, 1]
    zt = z.max() + 0.5 * dp
    sel = (z > 2 * dp) & (z < zt - 2 * dp) & (x > 3 * dp) & (x < size - 3 * dp) & (y > 3 * dp) & (y < size - 3 * dp)
    ref = rho0 * g * (zt - z[sel]) * scale
    err = float(np.abs(press[sel] - ref).mean() / ref.mean())
    bias = float((press[sel] - ref).mean() / ref.mean())
    bins = np.arange(dp, zt, dp)
    prof = []
    for zb in bins:
        m = np.abs(z - zb) < 0.5 * dp
        m &= (x > 3 * dp) & (x < size - 3 * dp) & (y > 3 * dp) & (y < size - 3 * dp)
        if m.any():
            prof.append([float(zb), float(press[m].mean()), float(rho0 * g * (zt - zb) * scale)])
    return {"err": err, "bias": bias, "ztop": float(zt), "profile": prof}


def bottom_force(pos, press, kind, dp, area, weight):
    m = (kind == 0) & (np.abs(pos[:, 2]) < 0.5 * dp)
    return float(press[m].sum() * dp * dp / weight)


def run_ours(dp, size, depth, t_end, every, arch, out):
    wcsph.init(arch)
    f, b = cases.tank(size, size, 400.0, depth, dp)
    p = Params(dp=dp, hswl=depth)
    s = wcsph.Solver(p, f, b)
    s.hydrostatic(f[:, 2].max(), bound=False)
    weight = p.rho0 * 9810.0 * len(f) * p.vol
    rows = []

    def cb(s):
        d = s.snapshot()
        fl = slice(s.nb, None)
        m = metrics(d["pos"][fl], d["press"][fl], dp, size)
        m["bottom"] = bottom_force(d["pos"], d["press"], d["kind"], dp, size * size, weight)
        m["time"] = s.time
        m["vmax"] = float(np.linalg.norm(d["vel"][fl], axis=1).max())
        rows.append(m)
        print(f"[ours] t={s.time:.2f} err={m['err']:.4f} bias={m['bias']:+.4f} bottom={m['bottom']:.3f} vmax={m['vmax']:.1f} mm/s", flush=True)

    t0 = time.time()
    s.run(t_end, every=every, callback=cb)
    info = s.info()
    info["seconds"] = time.time() - t0
    np.savez_compressed(os.path.join(out, "ours_final.npz"), **s.snapshot())
    return {"rows": rows, "info": info, "params": p.to_dict()}


def read_dsph(csvdir, dp, size):
    dpm, sm = dp / 1000.0, size / 1000.0
    rows = []
    files = sorted(glob.glob(os.path.join(csvdir, "fluid_[0-9]*.csv")))
    for f in files:
        k = os.path.basename(f)[6:10]
        with open(f) as fh:
            fh.readline()
            t = float(fh.readline().split(";")[0])
        a = np.loadtxt(f, delimiter=";", skiprows=4, usecols=(0, 1, 2, 4, 5, 6, 8))
        bf = os.path.join(csvdir, f"bound_{k}.csv")
        bb = np.loadtxt(bf, delimiter=";", skiprows=4, usecols=(0, 1, 2, 8))
        if t == 0:
            continue
        m = metrics(a[:, :3], a[:, 6], dpm, sm, scale=1.0, rho0=998.0, g=9.81)
        weight = 998.0 * 9.81 * len(a) * dpm ** 3
        m["bottom"] = bottom_force(bb[:, :3], bb[:, 3], np.zeros(len(bb), int), dpm, sm * sm, weight)
        m["time"] = t
        m["vmax"] = float(np.linalg.norm(a[:, 3:6], axis=1).max() * 1000)
        m["profile"] = [[z * 1000, pr * 1e-6, r * 1e-6] for z, pr, r in m["profile"]]
        m["ztop"] *= 1000
        rows.append(m)
        print(f"[dsph] t={t:.2f} err={m['err']:.4f} bias={m['bias']:+.4f} bottom={m['bottom']:.3f} vmax={m['vmax']:.1f} mm/s", flush=True)
    return {"rows": rows}


def unit_check(arch, steps=300):
    wcsph.init(arch)
    res = {}
    for name, sc, g, rho0 in (("mm", 1.0, 9810.0, 0.998e-9), ("si", 1e-3, 9.81, 998.0)):
        dp = 20.0 * sc
        f, b = cases.tank(200 * sc, 200 * sc, 400 * sc, 200 * sc, dp)
        p = Params(dp=dp, hswl=200 * sc, rho0=rho0, gravity=(0.0, 0.0, -g))
        s = wcsph.Solver(p, f, b, pos64=True)
        s.hydrostatic(f[:, 2].max(), bound=False)
        for _ in range(steps):
            s.step()
        d = s.snapshot(True)
        res[name] = {"time": s.time, "press": d["press"], "pos": d["pos"], "vel": d["vel"]}
    mm, si = res["mm"], res["si"]
    pr = np.abs(mm["press"] * 1e6 - si["press"]).max() / np.abs(si["press"]).max()
    ps = np.abs(mm["pos"] * 1e-3 - si["pos"]).max() / 0.02
    vs = np.abs(mm["vel"] * 1e-3 - si["vel"]).max() / max(np.abs(si["vel"]).max(), 1e-12)
    tt = abs(mm["time"] - si["time"]) / si["time"]
    out = {"steps": steps, "time_rel": float(tt), "press_rel": float(pr), "pos_rel_dp": float(ps), "vel_rel": float(vs)}
    print(f"[unit] mm vs SI after {steps} steps: time {tt:.2e}, press {pr:.2e}, pos/dp {ps:.2e}, vel {vs:.2e}", flush=True)
    return out


def plot(ours, dsph, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    po = np.array(ours["rows"][-1]["profile"])
    ax[0].plot(po[:, 2] * 1e3, po[:, 0], "k-", label="ρgh")
    ax[0].plot(po[:, 1] * 1e3, po[:, 0], "o", ms=4, color="#2563eb", label="SPH-LLM-web")
    if dsph and dsph["rows"]:
        pd = np.array(dsph["rows"][-1]["profile"])
        ax[0].plot(pd[:, 1] * 1e3, pd[:, 0], "s", ms=4, mfc="none", color="#dc2626", label="DualSPHysics 5.4")
    ax[0].set_xlabel("pressure [kPa]")
    ax[0].set_ylabel("z [mm]")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    t = [r["time"] for r in ours["rows"]]
    ax[1].plot(t, [100 * r["err"] for r in ours["rows"]], "o-", color="#2563eb", label="SPH-LLM-web")
    if dsph and dsph["rows"]:
        ax[1].plot([r["time"] for r in dsph["rows"]], [100 * r["err"] for r in dsph["rows"]], "s--", mfc="none", color="#dc2626", label="DualSPHysics 5.4")
    ax[1].axhline(5, color="gray", lw=1, ls=":")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("mean |P - ρgh| / mean ρgh [%]")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", type=float, default=20.0)
    ap.add_argument("--size", type=float, default=400.0)
    ap.add_argument("--depth", type=float, default=300.0)
    ap.add_argument("--t", type=float, default=1.2)
    ap.add_argument("--every", type=float, default=0.2)
    ap.add_argument("--arch", default="cpu")
    ap.add_argument("--dsph", default=None)
    ap.add_argument("--out", default="runs/hydrostatic")
    ap.add_argument("--skip-unit", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {"case": vars(a)}
    if not a.skip_unit:
        res["unit"] = unit_check(a.arch)
    res["ours"] = run_ours(a.dp, a.size, a.depth, a.t, a.every, a.arch, a.out)
    res["dsph"] = read_dsph(a.dsph, a.dp, a.size) if a.dsph else None
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    plot(res["ours"], res["dsph"], os.path.join(a.out, "hydrostatic.png"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
