import json
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _f in ("Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Serif CJK KR", "Noto Serif CJK SC", "NanumGothic", "Malgun Gothic", "AppleGothic"):
    if any(_f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

from validation.dambreak import EXP, experiment

LAB = {"sym20": "Symplectic · c 20 · mDBC", "ver20": "Verlet · c 20 · mDBC", "ver10": "Verlet · c 10 · mDBC", "ver20dbc": "Verlet · c 20 · DBC"}
COL = {"sym20": "#2563eb", "ver20": "#059669", "ver10": "#d97706", "ver20dbc": "#7c3aed"}


def rows(res, what):
    out = {}
    for k, v in res.items():
        if not k.startswith(what + "|"):
            continue
        _, dx, c = k.split("|")
        out.setdefault(c, []).append((float(dx), v))
    for c in out:
        out[c].sort(key=lambda t: -t[0])
    return out


def main(path, prefix):
    res = json.load(open(path))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for i, (what, key, title) in enumerate([("hydro2d", "acc_pressure", "2D 정수압 (ρgh 대비)"), ("dam2d", "acc_position", "2D 댐 붕괴 선단 (Koshizuka & Oka)"), ("dam3d", "acc_position", "3D 댐 붕괴 선단 (Koshizuka & Oka)")]):
        for c, rs in rows(res, what).items():
            n = [v["nfluid"] for _, v in rs]
            a = [100 * v["score"][key] for _, v in rs]
            ax[i].plot(n, a, "o-", color=COL[c], label=LAB[c])
            for (dx, v), x, y in zip(rs, n, a):
                ax[i].annotate(f"{dx:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, color=COL[c])
        if what == "dam2d" and "dsph2d" in res:
            d = sorted(((float(k), v) for k, v in res["dsph2d"].items()), key=lambda t: -t[0])
            n = [(1000 / dx) * (2000 / dx) for dx, _ in d]
            ax[i].plot(n, [100 * v["score"]["acc_position"] for _, v in d], "s--", mfc="none", color="#dc2626", label="DualSPHysics 5.4 (Verlet · DBC)")
        ax[i].axhline(75, color="gray", ls=":", lw=1)
        ax[i].set_xscale("log")
        ax[i].set_xlabel("유체 입자 수")
        ax[i].set_ylabel("정합도 [%]")
        ax[i].set_title(title, fontsize=10)
        ax[i].grid(alpha=0.3)
        ax[i].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(prefix + "_accuracy.png", dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, what in enumerate(["dam2d", "dam3d"]):
        for c, rs in rows(res, what).items():
            ax[i].plot([v["nfluid"] for _, v in rs], [v["wall"] for _, v in rs], "o-", color=COL[c], label=LAB[c])
        if what == "dam2d" and "dsph2d" in res:
            d = sorted(((float(k), v) for k, v in res["dsph2d"].items()), key=lambda t: -t[0])
            ax[i].plot([(1000 / dx) * (2000 / dx) for dx, _ in d], [v["wall"] for _, v in d], "s--", mfc="none", color="#dc2626", label="DualSPHysics 5.4 CPU")
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        ax[i].set_xlabel("유체 입자 수")
        ax[i].set_ylabel("계산 시간 [s] (CPU 2코어)")
        ax[i].set_title("2D 댐 붕괴 0.75 s" if what == "dam2d" else "3D 댐 붕괴 0.34 s", fontsize=10)
        ax[i].grid(alpha=0.3)
        ax[i].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(prefix + "_time.png", dpi=130)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    te = np.array([e[0] for e in EXP])
    xe = np.array([e[1] for e in EXP]) * 1000
    ax.plot(te, xe / 1000, "k^", label="Koshizuka & Oka 1996")
    for dx, st in ((40, ":"), (20, "--"), (10, "-")):
        k = f"dam2d|{dx}|ver20dbc"
        if k in res:
            ax.plot(res[k]["t"], np.array(res[k]["x"]) / 1000, st, color="#2563eb", label=f"SPH-LLM-web dx {dx}")
        if "dsph2d" in res and f"{dx}" in res["dsph2d"]:
            v = res["dsph2d"][f"{dx}"]
            ax.plot(v["t"], np.array(v["x"]) / 1000, st, color="#dc2626", alpha=0.7, label=f"DualSPHysics dx {dx}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("front x/L")
    ax.set_title("2D 댐 붕괴, Verlet · DBC (같은 조건)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(prefix + "_dam2d_front.png", dpi=130)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/convergence/result.json", sys.argv[2] if len(sys.argv) > 2 else "runs/convergence/conv")
