import argparse
import glob
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from validation.examples_check import run
from sphweb.engine.consts import AWEN3

BOX = ((900.0, 240.0, 0.0), (1020.0, 360.0, 450.0))


def dsph_probe(path):
    rows = []
    for line in open(path, encoding="latin-1"):
        parts = line.strip().split(";")
        if len(parts) < 3:
            continue
        try:
            rows.append([float(x) for x in parts[1:]])
        except ValueError:
            continue
    a = np.array(rows)
    return a[:, 0], a[:, 1:] * 1e-6


def dsph_force(csvdir, dp, rho0=1000.0, gamma=7.0):
    h = np.sqrt(3) * dp
    bw = -5.0 * AWEN3 / h ** 5
    m = rho0 * dp ** 3
    lo, hi = np.array(BOX[0]) / 1000 - 1.5 * dp, np.array(BOX[1]) / 1000 + 1.5 * dp
    ts, fx = [], []
    for f in sorted(glob.glob(os.path.join(csvdir, "fluid_[0-9]*.csv"))):
        k = os.path.basename(f)[6:10]
        with open(f) as fh:
            fh.readline()
            t = float(fh.readline().split(";")[0])
        a = np.loadtxt(f, delimiter=";", skiprows=4, usecols=(0, 1, 2, 7, 8))
        b = np.loadtxt(os.path.join(csvdir, f"bound_{k}.csv"), delimiter=";", skiprows=4, usecols=(0, 1, 2, 7, 8))
        inb = np.all((b[:, :3] > lo) & (b[:, :3] < hi), 1)
        b = b[inb]
        tree = cKDTree(a[:, :3])
        F = np.zeros(3)
        for xb, pb, rb in zip(b[:, :3], b[:, 4], b[:, 3]):
            idx = tree.query_ball_point(xb, 2 * h)
            if not idx:
                continue
            fl = a[idx]
            dr = fl[:, :3] - xb
            r = np.linalg.norm(dr, axis=1)
            q = r / h
            fac = bw * (1 - q / 2) ** 3
            coef = m * m * (fl[:, 4] + pb) / (fl[:, 3] * rb)
            F += (coef[:, None] * fac[:, None] * dr).sum(0)
        ts.append(t)
        fx.append(F[0])
    return np.array(ts), np.array(fx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsph", required=True)
    ap.add_argument("--dx", type=float, default=25.0)
    ap.add_argument("--out", default="runs/obstacle")
    ap.add_argument("--skip-ours", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "result.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    cfgs = {
        "same": [{"op": "set_particle", "dx": a.dx}, {"op": "set_fluid", "rho0": 1e-9}, {"op": "set_solver", "boundary": "dbc", "scheme": "verlet", "alpha": 0.1, "t_end": 1.0, "fps": 50}],
        "mdbc": [{"op": "set_particle", "dx": a.dx}, {"op": "set_solver", "boundary": "mdbc", "scheme": "symplectic", "alpha": 0.02, "t_end": 1.0, "fps": 50}],
    }
    if not a.skip_ours:
        for k, extra in cfgs.items():
            if k not in res:
                res[k] = run("dambreak_obstacle", extra, os.path.join(a.out, "runs"))
                json.dump(res, open(path, "w"), indent=1)
    t, p = dsph_probe(glob.glob(os.path.join(a.dsph, "probe*Press*.csv"))[0])
    tf, fx = dsph_force(os.path.join(a.dsph, "csv"), a.dx / 1000)
    for k in ("same", "mdbc"):
        if k in res:
            tt = np.array(res[k]["time"])
            fi = np.interp(np.array(tf), tt, res[k]["forces"]["building"]["x"])
            m = (tf >= 0.3) & (tf <= 1.0)
            print(k, "building impulse 0.3-1.0 s ours %.2f dsph %.2f N*s" % (np.trapezoid(fi[m], tf[m]), np.trapezoid(fx[m], tf[m])))
    res["dsph"] = {"time": t.tolist(), "p1": p[:, 0].tolist(), "p2": p[:, 1].tolist(), "ftime": tf.tolist(), "fx": fx.tolist()}
    json.dump(res, open(path, "w"), indent=1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for _f in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
        if any(_f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = _f
            break
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    sty = {"same": ("#2563eb", "SPH-LLM-web (원본과 같은 설정: DBC·Verlet·α 0.1)"), "mdbc": ("#059669", "SPH-LLM-web mDBC·Symplectic·α 0.02")}
    for k, (c, lab) in sty.items():
        if k in res:
            r = res[k]
            names = list(r["probes"])
            ax[0].plot(r["time"], np.array(r["probes"][names[0]]["pressure"]) * 1e3, color=c, label=lab)
            ax[1].plot(r["time"], np.array(r["probes"][names[1]]["pressure"]) * 1e3, color=c, label=lab)
            ax[2].plot(r["time"], r["forces"]["building"]["x"], color=c, label=lab)
    ax[0].plot(t, p[:, 0] * 1e3, "r--", label="DualSPHysics 5.4 (MeasureTool)")
    ax[1].plot(t, p[:, 1] * 1e3, "r--", label="DualSPHysics 5.4 (MeasureTool)")
    ax[2].plot(tf, fx * 1.0, "r--", label="DualSPHysics 5.4 (입자 데이터로 계산)")
    for x, ti in zip(ax, ["기둥 앞 압력 (877.9, 300, 20) [kPa]", "기둥 앞 압력 (877.9, 300, 100) [kPa]", "기둥이 받는 힘 Fx [N]"]):
        x.set_title(ti, fontsize=10)
        x.set_xlabel("t [s]")
        x.grid(alpha=0.3)
        x.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "obstacle.png"), dpi=130)
    print("saved", a.out)


if __name__ == "__main__":
    main()
