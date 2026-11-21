import json
import os

import numpy as np

from simweb.geometry import write_stl

HERE = os.path.dirname(os.path.abspath(__file__))


def tank_part(L, W, H, t=30.0):
    o = [[-t, -t], [L + t, -t], [L + t, W + t], [-t, W + t]]
    i = [[0, 0], [L, 0], [L, W], [0, W]]
    v = [p + [-t] for p in o] + [p + [H] for p in o] + [p + [H] for p in i] + [p + [0] for p in i]
    q = [[0, 3, 2, 1]]
    q += [[k, (k + 1) % 4, 4 + (k + 1) % 4, 4 + k] for k in range(4)]
    q += [[4 + k, 4 + (k + 1) % 4, 8 + (k + 1) % 4, 8 + k] for k in range(4)]
    q += [[8 + k, 8 + (k + 1) % 4, 12 + (k + 1) % 4, 12 + k] for k in range(4)]
    q += [[12, 13, 14, 15]]
    return v, q


def box_part(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
    q = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return v, q


def write_obj(path, parts):
    lines, off = [], 0
    for name, (v, q) in parts:
        lines.append(f"g {name}")
        lines += [f"v {a:g} {b:g} {c:g}" for a, b, c in v]
        lines += ["f " + " ".join(str(off + k + 1) for k in f) for f in q]
        off += len(v)
    open(path, "w").write("\n".join(lines) + "\n")


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
    write_obj(os.path.join(HERE, "still_tank.obj"), [("tank", tank_part(400.0, 400.0, 400.0))])
    write_obj(os.path.join(HERE, "dambreak_obstacle.obj"), [("tank", tank_part(1600.0, 670.0, 400.0)), ("building", box_part([900.0, 240.0, 0.0], [1020.0, 360.0, 450.0]))])
    write_obj(os.path.join(HERE, "drop_pool.obj"), [("tank", tank_part(800.0, 400.0, 600.0))])
    rg = 0.998e-9 * 9810.0
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
        "still_tank": {
            "title": "정수압 탱크 (이론값 ρgh 확인)",
            "file": "still_tank.obj",
            "note": f"가만히 있는 물 400×400×300 mm. 정답: 바닥 근처 프로브 (200, 200, 20) 압력 ≈ {rg * 280:.5f} MPa, 중간 (200, 200, 150) ≈ {rg * 150:.5f} MPa (ρgh, 수면 z ≈ 300). 속도는 0 근처, 밀도 오차는 1% 안이면 정상. 처음 0.2 s 는 출렁이며 자리를 잡습니다",
            "prompts": ["해석 시간 2초", "입자 2만 개로 해줘", "DBC로 바꿔줘"],
            "ops": [{"op": "set_particle", "dx": 20}, {"op": "set_solver", "t_end": 1.0, "fps": 20}, {"op": "set_domain", "min": [-100, -100, -100], "max": [500, 500, 500]}, {"op": "add_fluid_block", "name": "고인 물", "box": {"min": [0, 0, 0], "max": [400, 400, 300]}}, {"op": "add_probe", "name": "바닥", "point": [200, 200, 20]}, {"op": "add_probe", "name": "중간", "point": [200, 200, 150]}],
        },
        "dambreak_obstacle": {
            "title": "장애물 댐 붕괴 (DualSPHysics CaseDambreak)",
            "file": "dambreak_obstacle.obj",
            "note": "DualSPHysics 공식 예제 01_DamBreak 와 같은 장면: 탱크 1600×670×400, 물기둥 400×670×300, 기둥 장애물 120×120×450 (x 900~1020). 물이 약 0.35 s 에 기둥을 칩니다. 벽 하중 그래프의 building Fx 가 기둥이 받는 힘, 프로브는 원본 압력 측정점 (877.9, 300, 20). 원본 입자 간격 8.5 mm, 여기 기본 15 mm",
            "prompts": ["입자 간격 10 mm", "해석 시간 1.6초", "Verlet으로 바꿔줘"],
            "ops": [{"op": "set_particle", "dx": 15}, {"op": "set_solver", "t_end": 1.0, "fps": 50}, {"op": "set_domain", "min": [-100, -100, -100], "max": [1700, 770, 700]}, {"op": "add_fluid_block", "name": "물기둥", "box": {"min": [0, 0, 0], "max": [400, 670, 300]}}, {"op": "add_probe", "name": "기둥 앞 바닥", "point": [877.9, 300, 20]}, {"op": "add_probe", "name": "기둥 앞 중간", "point": [877.9, 300, 100]}],
        },
        "drop_pool": {
            "title": "물 덩어리 낙하 → 수조",
            "file": "drop_pool.obj",
            "note": "수심 150 mm 수조에 200 mm 정육면체 물 덩어리가 바닥에서 350 mm 높이에서 떨어집니다. 정답: 수면(150)까지 200 mm 낙하 → 약 0.2 s 에 √(2gh) ≈ 1.98 m/s 로 충돌. 바닥 프로브 (400, 200, 20) 압력은 처음 ≈ " + f"{rg * 130:.5f}" + " MPa (ρgh) 에서 충돌 때 뛰어오릅니다",
            "prompts": ["해석 시간 1.5초", "입자 5만 개로 해줘", "2D로 해석해줘"],
            "ops": [{"op": "set_particle", "dx": 12.5}, {"op": "set_solver", "t_end": 1.0, "fps": 50}, {"op": "set_domain", "min": [-100, -100, -100], "max": [900, 500, 700]}, {"op": "add_fluid_block", "name": "수조 물", "box": {"min": [0, 0, 0], "max": [800, 400, 150]}}, {"op": "add_fluid_block", "name": "떨어지는 물", "box": {"min": [300, 100, 350], "max": [500, 300, 550]}}, {"op": "add_probe", "name": "충돌 바닥", "point": [400, 200, 20]}],
        },
    }
    json.dump(presets, open(os.path.join(HERE, "presets.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("dam_tank.obj, funnel.stl, origami_2x2.obj, still_tank.obj, dambreak_obstacle.obj, drop_pool.obj, presets.json")


if __name__ == "__main__":
    main()
