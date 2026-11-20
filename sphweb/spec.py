import copy

import numpy as np

from simweb import units

VERSION = 1
LIST_KEYS = {"fluid_blocks": "fb", "inlets": "in", "outlets": "out", "plots": "pl", "stops": "st", "monitors": "mo"}
DEVICES = ("auto", "cuda", "metal", "vulkan", "cpu")
ROLES = ("wall", "ignore")
QUANTITIES = ("time", "particles.active", "density.error.mean", "density.error.max", "energy.kinetic", "energy.potential", "volume.fluid", "flow.in", "flow.out", "body.force.x", "body.force.y", "body.force.z", "probe.pressure", "probe.speed", "level")
MAX_PARTICLES = 20_000_000


def default_spec(geom=None):
    if geom is not None:
        lo, hi = geom.bbox()
        pad = 0.05 * float(np.max(hi - lo) or 1000.0)
        dom = {"min": (lo - pad).round(3).tolist(), "max": (hi + pad).round(3).tolist()}
        dx = float(np.round(max(np.max(hi - lo) / 60.0, 1e-3), 3))
        bodies = [{"group": g, "role": "wall", "motion": "fixed"} for g in geom.group_names() if not g.lower().startswith("hinge")]
    else:
        dom = {"min": [0.0, 0.0, 0.0], "max": [1000.0, 500.0, 600.0]}
        dx = 20.0
        bodies = []
    f = units.FLUIDS["water"]
    return {
        "version": VERSION,
        "units": "mm · tonne · s · N · MPa",
        "domain": dom,
        "fluid": {"preset": "water", "name": f["name"], "rho0": f["rho"], "mu": f["mu"]},
        "particle": {"dx": dx},
        "gravity": [0.0, 0.0, -units.G],
        "solver": {"method": "wcsph", "t_end": 2.0, "fps": 30, "cfl": 0.2, "viscosity": "artificial", "alpha": 0.02, "boundary": "mdbc", "ddt": 0.1},
        "device": "auto",
        "bodies": bodies,
        "fluid_blocks": [],
        "inlets": [],
        "outlets": [],
        "monitors": [],
        "stops": [],
        "plots": [],
    }


def _vec(v, n=3, name="벡터"):
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size != n or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}는 숫자 {n}개여야 합니다: {v}")
    return a


def _box(b, name="영역"):
    if not isinstance(b, dict) or "min" not in b or "max" not in b:
        raise ValueError(f"{name}은 {{'min': [x,y,z], 'max': [x,y,z]}} 형식이어야 합니다")
    lo, hi = _vec(b["min"], 3, name), _vec(b["max"], 3, name)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    if np.any(hi - lo <= 0):
        raise ValueError(f"{name}의 크기가 0인 방향이 있습니다")
    return {"min": lo.round(6).tolist(), "max": hi.round(6).tolist()}


def _pos(v, name):
    x = float(v)
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f"{name}는 0보다 커야 합니다")
    return x


def _next_id(spec, key):
    p = LIST_KEYS[key]
    used = {it["id"] for it in spec[key]}
    k = 1
    while f"{p}{k}" in used:
        k += 1
    return f"{p}{k}"


def _find(spec, iid):
    for key in LIST_KEYS:
        for i, it in enumerate(spec[key]):
            if it["id"] == iid:
                return key, i
    raise ValueError(f"'{iid}' 항목이 없습니다")


def _inlet(o, dx):
    c = _vec(o["center"], 3, "유입구 중심")
    n = _vec(o.get("normal", o.get("velocity", [0, 0, -1])), 3, "유입 방향")
    if np.linalg.norm(n) == 0:
        raise ValueError("유입 방향이 0 벡터입니다")
    n = n / np.linalg.norm(n)
    if "velocity" in o and o["velocity"] is not None:
        v = _vec(o["velocity"], 3, "유입 속도")
        sp = float(np.linalg.norm(v))
        if sp == 0:
            raise ValueError("유입 속도가 0입니다")
        if "normal" not in o:
            n = v / sp
    else:
        sp = _pos(o.get("speed", 0), "유입 속력")
        v = n * sp
    size = o.get("size")
    if size is None:
        size = [4 * dx, 4 * dx]
    size = [_pos(s, "유입구 크기") for s in (size if isinstance(size, (list, tuple)) else [size, size])]
    shape = o.get("shape", "rect")
    if shape not in ("rect", "circle"):
        raise ValueError("유입구 모양은 rect 또는 circle 입니다")
    t0 = float(o.get("t_start", 0.0))
    t1 = o.get("t_end")
    if t1 is not None and float(t1) <= t0:
        raise ValueError("유입 종료 시간이 시작 시간보다 커야 합니다")
    return {"center": c.round(6).tolist(), "normal": n.round(9).tolist(), "velocity": v.round(6).tolist(), "speed": round(sp, 6), "shape": shape, "size": size, "t_start": t0, "t_end": None if t1 is None else float(t1)}


def _plot(o):
    def q(x):
        name = x.get("quantity") if isinstance(x, dict) else x
        if name not in QUANTITIES:
            raise ValueError(f"그래프 항목 '{name}' 을(를) 모릅니다. 가능: {', '.join(QUANTITIES)}")
        return {"quantity": name, **({k: v for k, v in x.items() if k != "quantity"} if isinstance(x, dict) else {})}

    series = [q(s) for s in o.get("series", [])]
    if not series:
        raise ValueError("그래프에 표시할 항목(series)이 없습니다")
    return {"title": str(o.get("title", "그래프")), "x": q(o.get("x", {"quantity": "time"})), "series": series}


def apply_ops(spec, ops, geom=None):
    spec = copy.deepcopy(spec)
    log = []
    for o in ops:
        if not isinstance(o, dict) or "op" not in o:
            raise ValueError(f"수정 지시 형식이 잘못되었습니다: {o}")
        op = o["op"]
        dx = spec["particle"]["dx"]
        if op == "replace_spec":
            base = default_spec(geom)
            for k, v in o["spec"].items():
                if k in base:
                    base[k] = v
            spec = apply_ops(base, [], geom)[0]
            log.append("케이스를 불러왔습니다")
        elif op == "set_fluid":
            if "preset" in o and o["preset"]:
                p = units.FLUIDS.get(o["preset"])
                if p is None:
                    raise ValueError(f"유체 프리셋은 {', '.join(units.FLUIDS)} 입니다")
                spec["fluid"] = {"preset": o["preset"], "name": p["name"], "rho0": p["rho"], "mu": p["mu"]}
            for k in ("rho0", "mu"):
                if o.get(k) is not None:
                    spec["fluid"][k] = _pos(o[k], k)
                    spec["fluid"]["preset"] = "custom"
                    spec["fluid"]["name"] = "직접 입력"
            log.append(f"유체: {spec['fluid']['name']} (ρ₀ {spec['fluid']['rho0']:.4g} tonne/mm³, μ {spec['fluid']['mu']:.4g} MPa·s)")
        elif op == "set_particle":
            spec["particle"]["dx"] = _pos(o["dx"], "입자 간격")
            log.append(f"입자 간격 {spec['particle']['dx']:g} mm")
        elif op == "set_solver":
            s = spec["solver"]
            for k in ("t_end", "cfl", "alpha"):
                if o.get(k) is not None:
                    s[k] = _pos(o[k], k)
            if o.get("ddt") is not None:
                s["ddt"] = float(o["ddt"])
                if s["ddt"] < 0:
                    raise ValueError("밀도 확산 계수는 0 이상입니다")
            if o.get("fps") is not None:
                s["fps"] = int(_pos(o["fps"], "fps"))
            if o.get("method") is not None:
                if o["method"] != "wcsph":
                    raise ValueError("해석 방법은 wcsph (DualSPHysics 식) 입니다")
                s["method"] = o["method"]
            if o.get("viscosity") is not None:
                if o["viscosity"] not in ("artificial", "laminar"):
                    raise ValueError("점성은 artificial (인공 점성) 또는 laminar (층류) 입니다")
                s["viscosity"] = o["viscosity"]
            if o.get("boundary") is not None:
                if o["boundary"] not in ("dbc", "mdbc"):
                    raise ValueError("경계 조건은 dbc 또는 mdbc 입니다")
                s["boundary"] = o["boundary"]
            if s["cfl"] > 1:
                raise ValueError("CFL 계수는 1 이하여야 합니다")
            log.append(f"해석 시간 {s['t_end']:g} s, 출력 {s['fps']} 프레임/s")
        elif op == "set_device":
            if o.get("device") not in DEVICES:
                raise ValueError(f"계산 장치는 {', '.join(DEVICES)} 중 하나입니다")
            spec["device"] = o["device"]
            log.append(f"계산 장치 {o['device']}")
        elif op == "set_gravity":
            spec["gravity"] = _vec(o["vector"], 3, "중력").tolist()
            log.append(f"중력 {spec['gravity']} mm/s²")
        elif op == "set_domain":
            if o.get("fit"):
                lo, hi = _content_box(spec, geom)
                pad = float(o.get("margin", 4 * dx))
                spec["domain"] = {"min": (lo - pad).round(3).tolist(), "max": (hi + pad).round(3).tolist()}
            else:
                spec["domain"] = _box(o, "해석 영역")
            log.append(f"해석 영역 {spec['domain']['min']} ~ {spec['domain']['max']}")
        elif op == "add_fluid_block":
            it = {"id": _next_id(spec, "fluid_blocks"), "name": str(o.get("name") or "유체 블록"), "box": _box(o["box"], "유체 블록")}
            spec["fluid_blocks"].append(it)
            log.append(f"{it['id']} 유체 블록 추가")
        elif op == "add_inlet":
            it = {"id": _next_id(spec, "inlets"), "name": str(o.get("name") or "유입구"), **_inlet(o, dx)}
            spec["inlets"].append(it)
            log.append(f"{it['id']} 유입구 추가 ({it['speed']:g} mm/s)")
        elif op == "add_outlet":
            it = {"id": _next_id(spec, "outlets"), "name": str(o.get("name") or "유출 영역"), "box": _box(o["box"], "유출 영역")}
            spec["outlets"].append(it)
            log.append(f"{it['id']} 유출 영역 추가")
        elif op == "set_body":
            names = geom.group_names() if geom is not None else []
            g = o.get("group")
            if g not in names:
                raise ValueError(f"그룹 '{g}' 이(가) 없습니다. 사용 가능: {', '.join(names) or '없음'}")
            role = o.get("role", "wall")
            if role not in ROLES:
                raise ValueError(f"물체 역할은 {', '.join(ROLES)} 입니다")
            for b in spec["bodies"]:
                if b["group"] == g:
                    b["role"] = role
                    break
            else:
                spec["bodies"].append({"group": g, "role": role, "motion": "fixed"})
            log.append(f"그룹 {g}: {'벽' if role == 'wall' else '무시'}")
        elif op == "add_plot":
            it = {"id": _next_id(spec, "plots"), **_plot(o)}
            spec["plots"].append(it)
            log.append(f"{it['id']} 그래프 추가")
        elif op == "add_probe":
            it = {"id": _next_id(spec, "monitors"), "name": str(o.get("name") or f"프로브 {len(spec['monitors']) + 1}"), "point": _vec(o["point"], 3, "프로브 위치").tolist()}
            spec["monitors"].append(it)
            log.append(f"{it['id']} 프로브 ({', '.join(f'{v:g}' for v in it['point'])}) mm")
        elif op == "add_stop":
            if o.get("quantity") not in QUANTITIES:
                raise ValueError(f"정지 조건 항목 '{o.get('quantity')}' 을(를) 모릅니다")
            if o.get("cmp", ">") not in (">", "<", ">=", "<="):
                raise ValueError("비교는 >, <, >=, <= 입니다")
            it = {"id": _next_id(spec, "stops"), "quantity": o["quantity"], "cmp": o.get("cmp", ">"), "value": float(o["value"]), "action": o.get("action", "pause")}
            spec["stops"].append(it)
            log.append(f"{it['id']} 조건부 정지: {it['quantity']} {it['cmp']} {it['value']:g}")
        elif op == "update":
            key, i = _find(spec, o.get("id"))
            cur = spec[key][i]
            patch = {k: v for k, v in o.items() if k not in ("op", "id")}
            if key == "inlets":
                merged = {**cur, **patch}
                if "speed" in patch and "velocity" not in patch:
                    merged.pop("velocity", None)
                spec[key][i] = {"id": cur["id"], "name": str(merged.get("name", cur["name"])), **_inlet(merged, dx)}
            elif key in ("fluid_blocks", "outlets"):
                spec[key][i] = {**cur, **({"name": str(patch["name"])} if "name" in patch else {}), **({"box": _box(patch["box"])} if "box" in patch else {})}
            elif key == "monitors":
                spec[key][i] = {**cur, **({"name": str(patch["name"])} if "name" in patch else {}), **({"point": _vec(patch["point"], 3, "프로브 위치").tolist()} if "point" in patch else {})}
            else:
                spec[key][i] = {**cur, **patch}
            log.append(f"{o['id']} 수정")
        elif op == "remove":
            key, i = _find(spec, o.get("id"))
            spec[key].pop(i)
            log.append(f"{o['id']} 삭제")
        elif op == "clear":
            what = o.get("what", "all")
            keys = list(LIST_KEYS) if what == "all" else [what]
            for k in keys:
                if k not in LIST_KEYS:
                    raise ValueError(f"비울 수 있는 목록: {', '.join(LIST_KEYS)}, all")
                spec[k] = []
            log.append(f"{what} 비움")
        else:
            raise ValueError(f"모르는 수정 지시: {op}")
    return spec, log


def _content_box(spec, geom):
    pts = []
    if geom is not None and len(geom.vertices):
        lo, hi = geom.bbox()
        pts += [lo, hi]
    for b in spec["fluid_blocks"] + spec["outlets"]:
        pts += [np.array(b["box"]["min"]), np.array(b["box"]["max"])]
    for it in spec["inlets"]:
        pts.append(np.array(it["center"]))
    if not pts:
        return np.array(spec["domain"]["min"]), np.array(spec["domain"]["max"])
    p = np.array(pts)
    return p.min(axis=0), p.max(axis=0)


def block_count(box, dx):
    n = np.floor((np.array(box["max"]) - np.array(box["min"])) / dx + 1e-9).astype(int)
    return int(np.prod(np.maximum(n, 0)))


def estimate(spec):
    dx = spec["particle"]["dx"]
    fill = sum(block_count(b["box"], dx) for b in spec["fluid_blocks"])
    t_end = spec["solver"]["t_end"]
    emit = 0
    for it in spec["inlets"]:
        area = it["size"][0] * it["size"][1] * (np.pi / 4 if it["shape"] == "circle" else 1.0)
        dur = max((it["t_end"] if it["t_end"] is not None else t_end) - it["t_start"], 0.0)
        emit += int(area / dx**2 * it["speed"] * dur / dx)
    mass = spec["fluid"]["rho0"] * dx**3
    return {"fill": fill, "emitted": emit, "total": fill + emit, "particle_mass": mass, "support_radius": 2.0 * dx}


def warnings_for(spec, geom=None):
    out = []
    lo, hi = np.array(spec["domain"]["min"]), np.array(spec["domain"]["max"])
    for b in spec["fluid_blocks"] + spec["outlets"]:
        if np.any(np.array(b["box"]["min"]) < lo) or np.any(np.array(b["box"]["max"]) > hi):
            out.append({"level": "warn", "text": f"{b['id']} 가 해석 영역 밖으로 나가 있습니다"})
    for it in spec["inlets"]:
        if np.any(np.array(it["center"]) < lo) or np.any(np.array(it["center"]) > hi):
            out.append({"level": "warn", "text": f"{it['id']} 중심이 해석 영역 밖에 있습니다"})
        if min(it["size"]) < 2 * spec["particle"]["dx"]:
            out.append({"level": "warn", "text": f"{it['id']} 크기가 입자 간격의 2배보다 작습니다"})
    e = estimate(spec)
    if e["total"] > MAX_PARTICLES:
        out.append({"level": "err", "text": f"예상 입자 수 {e['total']:,}개가 상한 {MAX_PARTICLES:,}개를 넘습니다. 입자 간격을 키우세요"})
    if not spec["fluid_blocks"] and not spec["inlets"]:
        out.append({"level": "info", "text": "유체가 없습니다. 유체 블록이나 유입구를 추가하세요"})
    if geom is not None:
        s = geom.summary()
        if s["counts"]["faces"] and not s["closed"] and geom.kind == "stl":
            out.append({"level": "warn", "text": f"STL이 닫힌 표면이 아닙니다 (열린 엣지 {s['open_edges']}개). 벽 입자는 표면 양쪽에 깔립니다"})
    return out
