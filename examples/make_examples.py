import json
import os

import numpy as np

from simweb.geometry import write_stl

HERE = os.path.dirname(os.path.abspath(__file__))


def tank(path, L=1600.0, W=600.0, H=600.0, t=30.0):
    o = [[-t, -t], [L + t, -t], [L + t, W + t], [-t, W + t]]
    i = [[0, 0], [L, 0], [L, W], [0, W]]
    v = [p + [-t] for p in o] + [p + [H] for p in o] + [p + [H] for p in i] + [p + [0] for p in i]
    q = [[0, 3, 2, 1]]
    q += [[k, (k + 1) % 4, 4 + (k + 1) % 4, 4 + k] for k in range(4)]
    q += [[4 + k, 4 + (k + 1) % 4, 8 + (k + 1) % 4, 8 + k] for k in range(4)]
    q += [[8 + k, 8 + (k + 1) % 4, 12 + (k + 1) % 4, 12 + k] for k in range(4)]
    q += [[12, 13, 14, 15]]
    lines = ["g tank"] + [f"v {a:g} {b:g} {c:g}" for a, b, c in v] + ["f " + " ".join(str(k + 1) for k in f) for f in q]
    open(path, "w").write("\n".join(lines) + "\n")


def offset(P, d):
    T = np.diff(P, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    N = np.c_[T[:, 1], -T[:, 0]]
    Q = P.astype(float).copy()
    Q[0] += d * N[0]
    Q[-1] += d * N[-1]
    for k in range(1, len(P) - 1):
        n = N[k - 1] + N[k]
        n /= np.linalg.norm(n)
        Q[k] += n * d / (n @ N[k])
    return Q


def funnel(path, r_top=200.0, r_out=20.0, z_neck=60.0, z_cone=260.0, z_top=460.0, thk=15.0, nseg=64):
    inner = np.array([[r_out, 0], [r_out, z_neck], [r_top, z_cone], [r_top, z_top]])
    prof = np.vstack([inner, offset(inner, thk)[::-1]])
    th = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    V = np.array([[[r * np.cos(a), r * np.sin(a), z] for a in th] for r, z in prof])
    tris = []
    n = len(prof)
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(nseg):
            j2 = (j + 1) % nseg
            tris += [[(i, j), (i2, j), (i2, j2)], [(i, j), (i2, j2), (i, j2)]]
    flat = V.reshape(-1, 3)
    idx = np.array([[a * nseg + b for a, b in t] for t in tris])
    write_stl(path, flat, idx, "funnel")


def origami(path, a=200.0, z=300.0):
    v = [[0, 0], [a, 0], [2 * a, 0], [0, a], [a, a], [2 * a, a], [0, 2 * a], [a, 2 * a], [2 * a, 2 * a]]
    lines = [f"v {x:g} {y:g} {z:g}" for x, y in v]
    lines += ["g panel_1", "f 1 2 5 4", "g panel_2", "f 2 3 6 5", "g panel_3", "f 4 5 8 7", "g panel_4", "f 5 6 9 8", "g hinge_x", "l 4 5 6", "g hinge_y", "l 2 5 8"]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    tank(os.path.join(HERE, "dam_tank.obj"))
    funnel(os.path.join(HERE, "funnel.stl"))
    origami(os.path.join(HERE, "origami_2x2.obj"))
    presets = {
        "dam_break": {
            "title": "3D 댐 붕괴 (탱크 1600×600×600)",
            "file": "dam_tank.obj",
            "note": "물기둥 400×600×300 mm (DualSPHysics CaseDambreak 과 같은 물기둥). 입자 간격 10 mm 면 유체 7만 2천 개라 GPU 권장, CPU 에서는 20 mm 로 키우면 빠르게 볼 수 있습니다",
            "prompts": ["입자 간격 10 mm", "해석 시간 1.5초", "x 0~400, y 0~600, z 0~300 mm 에 물 채워줘"],
            "ops": [{"op": "set_particle", "dx": 10}, {"op": "set_solver", "t_end": 1.5, "fps": 50}, {"op": "set_domain", "min": [-100, -100, -100], "max": [1700, 700, 800]}, {"op": "add_fluid_block", "name": "물기둥", "box": {"min": [0, 0, 0], "max": [400, 600, 300]}}],
        },
        "funnel_jet": {
            "title": "깔때기 위에서 쏘는 제트",
            "file": "funnel.stl",
            "note": "깔때기 상단 위 (100, 0, 580) mm 에서 아래로 1.5 m/s 원형 제트, 2.4 초 동안 유입",
            "prompts": ["입자 간격 5 mm", "(100, 0, 580) mm 에서 아래로 1.5 m/s 로 유입, 지름 40 mm", "z -300~-100, x -300~300, y -300~300 mm 유출 영역 만들어줘"],
            "ops": [{"op": "set_particle", "dx": 5}, {"op": "set_solver", "t_end": 3.0, "fps": 30}, {"op": "set_domain", "min": [-300, -300, -300], "max": [300, 300, 650]}, {"op": "add_inlet", "name": "제트", "center": [100, 0, 580], "normal": [0, 0, -1], "speed": 1500, "shape": "circle", "size": [40, 40], "t_end": 2.4}, {"op": "add_outlet", "name": "바닥 배출", "box": {"min": [-300, -300, -300], "max": [300, 300, -150]}}],
        },
        "origami_panels": {
            "title": "오리가미 패널 2×2 (힌지 선 포함, 4단계용)",
            "file": "origami_2x2.obj",
            "note": "FEM-LLM-web 과 같은 OBJ 규칙: 면 = 패널, hinge_ 그룹 선 = 접힘선. FSI 는 4단계에서 연결",
            "prompts": ["x 0~400, y 0~400, z 0~150 mm 에 물 채워줘", "입자 간격 8 mm"],
            "ops": [{"op": "set_particle", "dx": 8}, {"op": "set_domain", "min": [-100, -100, -50], "max": [500, 500, 500]}],
        },
    }
    json.dump(presets, open(os.path.join(HERE, "presets.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("dam_tank.obj, funnel.stl, origami_2x2.obj, presets.json")


if __name__ == "__main__":
    main()
