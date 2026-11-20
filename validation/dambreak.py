import argparse
import glob
import json
import os

import numpy as np

from sphweb.engine import cases, wcsph
from sphweb.engine.consts import Params

EXP = [
    (0.092031603, 1.128), (0.190699774, 1.244), (0.266343115, 1.464), (0.32214447, 1.695), (0.368058691, 1.903),
    (0.413950339, 2.123), (0.446591422, 2.365), (0.495801354, 2.584), (0.521851016, 2.804), (0.56462754, 3.034),
    (0.594018059, 3.219), (0.636591422, 3.462), (0.669255079, 3.704), (0.70214447, 3.9), (0.751331828, 4.132),
]
EXP_L = 1.0


def experiment(col_l):
    s = col_l / (EXP_L * 1000.0)
    t = np.array([e[0] for e in EXP]) * np.sqrt(s)
    x = np.array([e[1] for e in EXP]) * 1000.0 * s
    return t, x


def tip(pos, col_l, dp):
    low = pos[:, 2] < 0.25 * col_l
    x = pos[low, 0]
    if len(x) == 0:
        return 0.0
    return float(np.sort(x)[-min(3, len(x))] + 0.5 * dp)


def score(ts, xs, col_l):
    te, xe = experiment(col_l)
    m = te <= ts[-1] + 1e-9
    xi = np.interp(te[m], ts, xs)
    rel = np.abs(xi - xe[m]) / xe[m]
    adv = np.abs(xi - xe[m]) / np.maximum(xe[m] - col_l, 1e-9)
    return {"acc_position": float(1 - rel.mean()), "acc_advance": float(1 - adv.mean()), "n": int(m.sum()), "max_rel": float(rel.max())}


def run_ours(a, mdbc):
    wcsph.init(a.arch)
    if mdbc:
        f, b, nr = cases.dam_break(a.length, a.width, a.height, a.col, 2 * a.col, a.dp, layers=4, normals=True)
    else:
        f, b = cases.dam_break(a.length, a.width, a.height, a.col, 2 * a.col, a.dp)
        nr = None
    p = Params(dp=a.dp, hswl=2 * a.col, visco=a.visco)
    s = wcsph.Solver(p, f, b, bound_normals=nr)
    s.hydrostatic(f[:, 2].max(), bound=False)
    ts, xs, frames = [0.0], [a.col + 0.5 * a.dp], []
    shots = set(round(v, 4) for v in a.shots)

    def cb(s):
        d = s.snapshot(True)
        ts.append(s.time)
        xs.append(tip(d["pos"], a.col, a.dp))
        k = round(round(s.time / a.every) * a.every, 4)
        if k in shots:
            frames.append({"time": s.time, "pos": d["pos"].astype(np.float32), "speed": np.linalg.norm(d["vel"], axis=1).astype(np.float32), "press": d["press"].astype(np.float32)})

    s.run(a.t, every=a.every, callback=cb)
    info = s.info()
    name = "mdbc" if mdbc else "dbc"
    print(f"[{name}] steps={info['steps']} wall={info['wall']:.1f}s ms/step={info['wall'] / info['steps'] * 1000:.1f} out={info['out']}", flush=True)
    return {"t": ts, "x": xs, "info": info, "score": score(np.array(ts), np.array(xs), a.col), "frames": frames}


def read_dsph(csvdir, a):
    ts, xs = [0.0], [a.col + 0.5 * a.dp]
    for f in sorted(glob.glob(os.path.join(csvdir, "fluid_[0-9]*.csv"))):
        with open(f) as fh:
            fh.readline()
            t = float(fh.readline().split(";")[0])
        if t <= 0:
            continue
        pos = np.loadtxt(f, delimiter=";", skiprows=4, usecols=(0, 1, 2)) * 1000.0
        ts.append(t)
        xs.append(tip(pos, a.col, a.dp))
    return {"t": ts, "x": xs, "score": score(np.array(ts), np.array(xs), a.col)}


def plot(res, a, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    te, xe = experiment(a.col)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(te, xe, "k^", label="Koshizuka & Oka 1996 (scaled)")
    sty = {"dbc": ("-", "#2563eb", "SPH-LLM-web DBC"), "mdbc": ("-", "#059669", "SPH-LLM-web mDBC"), "dsph": ("--", "#dc2626", "DualSPHysics 5.4 DBC")}
    for k, (ls, c, lab) in sty.items():
        if res.get(k):
            ax.plot(res[k]["t"], res[k]["x"], ls, color=c, label=f"{lab} ({100 * res[k]['score']['acc_position']:.1f}%)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("front position x [mm]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_frames(frames, a, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(frames)
    if not n:
        return
    fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 2.8), squeeze=False)
    vmax = max(float(f["speed"].max()) for f in frames)
    for ax, f in zip(axs[0], frames):
        p = f["pos"]
        mid = np.abs(p[:, 1] - a.width / 2) < a.dp
        sc = ax.scatter(p[mid, 0], p[mid, 2], c=f["speed"][mid], s=3, cmap="turbo", vmin=0, vmax=vmax)
        ax.set_xlim(0, a.length)
        ax.set_ylim(0, a.col * 2.2)
        ax.set_aspect("equal")
        ax.set_title(f"t = {f['time']:.3f} s", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(sc, ax=axs[0].tolist(), shrink=0.8, label="speed [mm/s]")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", type=float, default=10.0)
    ap.add_argument("--col", type=float, default=200.0)
    ap.add_argument("--length", type=float, default=800.0)
    ap.add_argument("--width", type=float, default=100.0)
    ap.add_argument("--height", type=float, default=600.0)
    ap.add_argument("--visco", type=float, default=0.02)
    ap.add_argument("--t", type=float, default=0.34)
    ap.add_argument("--every", type=float, default=0.005)
    ap.add_argument("--shots", type=float, nargs="*", default=[0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--arch", default="cpu")
    ap.add_argument("--dsph", default=None)
    ap.add_argument("--modes", default="dbc,mdbc")
    ap.add_argument("--out", default="runs/dambreak")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {"case": vars(a)}
    for m in a.modes.split(","):
        r = run_ours(a, m == "mdbc")
        plot_frames(r.pop("frames"), a, os.path.join(a.out, f"frames_{m}.png"), f"SPH-LLM-web {m.upper()} (mid-plane, speed)")
        res[m] = r
        print(f"[{m}] {r['score']}", flush=True)
    if a.dsph:
        res["dsph"] = read_dsph(a.dsph, a)
        print(f"[dsph] {res['dsph']['score']}", flush=True)
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    plot(res, a, os.path.join(a.out, "dambreak.png"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
