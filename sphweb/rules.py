import re

from simweb import units

NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
LEN_U = r"(?:mm|cm|m|㎜|㎝)?"
FILLER = {"해줘", "해", "줘", "좀", "주세요", "해주세요", "하자", "할게", "으로", "로", "을", "를", "에", "의", "에서", "까지", "부터", "그리고", "하고", "및", "또", "다시", "설정", "바꿔", "바꿔줘", "변경", "만들어", "만들어줘", "넣어", "넣어줘", "추가", "추가해줘", "놓고", "두고", "줘요", "요", "는", "은", "이", "가", "와", "과", "후", "다음"}

DIRS = {
    "아래": [0, 0, -1], "아래로": [0, 0, -1], "하향": [0, 0, -1], "위": [0, 0, 1], "위로": [0, 0, 1], "상향": [0, 0, 1],
    "-z": [0, 0, -1], "+z": [0, 0, 1], "z": [0, 0, 1], "-x": [-1, 0, 0], "+x": [1, 0, 0], "x": [1, 0, 0], "-y": [0, -1, 0], "+y": [0, 1, 0], "y": [0, 1, 0],
}


class P:
    def __init__(self, text):
        self.text = text
        self.used = []
        self.ops = []
        self.replies = []
        self.ask = []
        self.action = None
        self.undo = False

    def take(self, m, group=0):
        self.used.append(m.span(group))

    def leftover(self):
        t = list(self.text)
        for a, b in self.used:
            for i in range(a, b):
                t[i] = " "
        s = re.sub(r"[\s,.;:!?()~\-+·\[\]]+", " ", "".join(t)).strip()
        words = [w for w in s.split() if w not in FILLER and not re.fullmatch(r"(으로|로|을|를|에|은|는|이|가|해|줘|요)+", w)]
        return " ".join(words)


def _len(v, u):
    return units.to_internal(float(v), u or "mm", "length")[0]


def _ranges(text):
    out = {}
    spans = []
    for m in re.finditer(rf"\b([xyz])\s*[:=]?\s*({NUM})\s*({LEN_U})\s*(?:~|에서|부터|-|to|\.\.)\s*({NUM})\s*({LEN_U})\s*(?:까지|사이)?", text):
        u2 = m.group(5) or m.group(3)
        u1 = m.group(3) or u2
        out[m.group(1)] = sorted([_len(m.group(2), u1), _len(m.group(4), u2)])
        spans.append(m.span())
    return out, spans


def _point(text):
    m = re.search(rf"\(\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\)\s*({LEN_U})", text)
    if not m:
        return None, None
    u = m.group(4) or "mm"
    return [_len(m.group(i), u) for i in (1, 2, 3)], m


def _box_from(p, kind):
    r, spans = _ranges(p.text)
    if len(r) == 3:
        for s in spans:
            p.used.append(s)
        return {"min": [r["x"][0], r["y"][0], r["z"][0]], "max": [r["x"][1], r["y"][1], r["z"][1]]}
    m = re.search(rf"\(\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\)\s*(?:~|-|에서|부터|to)\s*\(\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\)\s*({LEN_U})", p.text)
    if m:
        u = m.group(7) or "mm"
        p.take(m)
        v = [_len(m.group(i), u) for i in range(1, 7)]
        return {"min": v[:3], "max": v[3:]}
    if r:
        p.ask.append(f"{kind} 범위를 x, y, z 세 방향 모두 알려 주세요 (예: x 0~500, y 0~300, z 0~400 mm)")
        for s in spans:
            p.used.append(s)
    return None


def parse(text, geom=None, spec=None):
    p = P(text)
    t = text
    if re.search(r"되돌려|되돌리기|취소해|undo", t):
        p.undo = True
        for m in re.finditer(r"되돌려|되돌리기|취소해|undo", t):
            p.take(m)
        return [], "", {"complete": True, "undo": True}

    for m in re.finditer(r"(물|공기|기름|윤활유|water|air|oil)\s*(?:로|으로)?\s*(?:바꿔|설정|변경|해|사용|으로 해)", t):
        key = {"물": "water", "water": "water", "공기": "air", "air": "air", "기름": "oil", "윤활유": "oil", "oil": "oil"}[m.group(1)]
        p.ops.append({"op": "set_fluid", "preset": key})
        p.take(m)
    for m in re.finditer(rf"밀도\s*({NUM})\s*(kg/m3|kg/m³|g/cm3|g/cm³|tonne/mm3)?", t):
        p.ops.append({"op": "set_fluid", "rho0": units.to_internal(float(m.group(1)), m.group(2) or "kg/m3", "density")[0]})
        p.take(m)
    for m in re.finditer(rf"(?<!인공\s)(?<!인공)점성(?:계수)?\s*({NUM})\s*(Pa·s|Pa\*s|Pa s|mPa·s|mPa\*s|cP|MPa·s)?", t):
        u = (m.group(2) or "Pa·s").replace("Pa s", "Pa·s")
        p.ops.append({"op": "set_fluid", "mu": units.to_internal(float(m.group(1)), u, "dynamic_viscosity")[0]})
        p.take(m)

    for m in re.finditer(rf"(?:입자\s*(?:간격|크기|지름|해상도)|해상도|dx|dp)\s*(?:을|를|은|는)?\s*({NUM})\s*({LEN_U})", t):
        p.ops.append({"op": "set_particle", "dx": _len(m.group(1), m.group(2))})
        p.take(m)

    for m in re.finditer(rf"(?:해석\s*시간|종료\s*시간|시간)\s*(?:을|를|은|는)?\s*({NUM})\s*(초|s|ms|분)|({NUM})\s*(초|s)\s*(?:동안|까지)", t):
        v, u = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        p.ops.append({"op": "set_solver", "t_end": units.to_internal(float(v), u, "time")[0]})
        p.take(m)
    for m in re.finditer(rf"(?:초당\s*({NUM})\s*(?:프레임|장)|fps\s*({NUM})|({NUM})\s*fps)", t, re.I):
        p.ops.append({"op": "set_solver", "fps": int(float(m.group(1) or m.group(2) or m.group(3)))})
        p.take(m)
    for m in re.finditer(rf"cfl\s*(?:을|를)?\s*({NUM})", t, re.I):
        p.ops.append({"op": "set_solver", "cfl": float(m.group(1))})
        p.take(m)

    for m in re.finditer(r"(mdbc|dbc)\s*(?:경계)?\s*(?:로|으로)?\s*(?:바꿔|설정|해|써|사용)?", t, re.I):
        p.ops.append({"op": "set_solver", "boundary": m.group(1).lower()})
        p.take(m)
    for m in re.finditer(r"층류\s*점성\s*(?:으로|로)?\s*(?:바꿔|설정|해|써|사용)?", t):
        p.ops.append({"op": "set_solver", "viscosity": "laminar"})
        p.take(m)
    for m in re.finditer(rf"인공\s*점성(?:\s*(?:계수|α|알파))?\s*(?:을|를|은|는)?\s*({NUM})?\s*(?:으로|로)?\s*(?:바꿔|설정|해|써|사용)?", t):
        o = {"op": "set_solver", "viscosity": "artificial"}
        if m.group(1):
            o["alpha"] = float(m.group(1))
        p.ops.append(o)
        p.take(m)
    if re.search(r"프로브|측정\s*점|모니터", t):
        pt, m = _point(t)
        if pt:
            p.ops.append({"op": "add_probe", "point": pt})
            p.take(m)
            p.take(re.search(r"(?:에|에서)?\s*(?:프로브|측정\s*점|모니터)\s*(?:추가|넣어|만들어|둬|설치)?\S*", t))
        else:
            p.ask.append("프로브 위치를 (x, y, z) mm 로 알려 주세요")

    for m in re.finditer(r"(GPU|gpu|CUDA|cuda|Metal|metal|CPU|cpu|자동)\s*(?:로|으로)\s*(?:계산|돌려|해석|설정|바꿔)?", t):
        d = {"gpu": "auto", "cuda": "cuda", "metal": "metal", "cpu": "cpu", "자동": "auto"}[m.group(1).lower()]
        p.ops.append({"op": "set_device", "device": d})
        p.take(m)

    for m in re.finditer(r"중력\s*(?:을|를)?\s*(꺼|끄|없애|제거|0)", t):
        p.ops.append({"op": "set_gravity", "vector": [0, 0, 0]})
        p.take(m)
    for m in re.finditer(r"중력\s*(?:을|를)?\s*(켜|다시|기본)", t):
        p.ops.append({"op": "set_gravity", "vector": [0, 0, -units.G]})
        p.take(m)

    if re.search(r"(채워|채우|물\s*블록|유체\s*블록|담아|가득)", t) and not re.search(r"유입|분사|쏴|뿜", t):
        b = _box_from(p, "유체 블록")
        if b:
            p.ops.append({"op": "add_fluid_block", "box": b})
            for m in re.finditer(r"(?:영역|범위|박스)?\s*(?:에|을|를)?\s*(?:물|유체)?\s*(?:을|를)?\s*(?:채워|채우|담아|가득)\S*|(?:물|유체)\s*블록\S*", t):
                p.take(m)
        elif not p.ask:
            p.ask.append("유체를 채울 범위를 알려 주세요 (예: x 0~500, y 0~300, z 0~400 mm 에 물 채워줘)")
            for m in re.finditer(r"채워\S*|채우\S*|가득\S*", t):
                p.take(m)

    DEL = re.search(r"삭제|지워|비워|제거|없애", t)
    if re.search(r"유입|분사|쏴|뿜|주입", t) and not DEL:
        c, mc = _point(t)
        speeds = []
        for m in re.finditer(rf"({NUM})\s*(m/s|mm/s|cm/s|km/h)", t):
            speeds.append(units.to_internal(float(m.group(1)), m.group(2), "velocity")[0])
            p.take(m)
        dirv = None
        for m in re.finditer(r"(아래로|아래|위로|위|하향|상향|[+-][xyz])\s*(?:방향)?(?:으로|로)?", t):
            dirv = DIRS[m.group(1)]
            p.take(m)
        size = None
        m = re.search(rf"(?:크기|단면)\s*({NUM})\s*(?:x|×|\*)\s*({NUM})\s*({LEN_U})", t)
        if m:
            size = [_len(m.group(1), m.group(3)), _len(m.group(2), m.group(3))]
            p.take(m)
        shape = "rect"
        m = re.search(rf"(?:지름|직경)\s*({NUM})\s*({LEN_U})", t)
        if m:
            d = _len(m.group(1), m.group(2))
            size, shape = [d, d], "circle"
            p.take(m)
        miss = []
        if c is None:
            miss.append("유입구 중심 좌표 (예: (0, 0, 500) mm)")
        else:
            p.take(mc)
        if not speeds:
            miss.append("유입 속도 (예: 2 m/s)")
        if dirv is None:
            miss.append("방향 (예: 아래로, -x)")
        if miss:
            p.ask.append("유입구를 만들려면 " + ", ".join(miss) + " 을(를) 알려 주세요")
        else:
            o = {"op": "add_inlet", "center": c, "normal": dirv, "speed": speeds[0], "shape": shape}
            if size:
                o["size"] = size
            m = re.search(rf"({NUM})\s*(초|s)\s*(?:동안|까지)\s*(?:만)?\s*(?:유입|분사|쏴|뿜|주입)", t)
            if m:
                o["t_end"] = float(m.group(1))
                p.take(m)
            p.ops.append(o)
        for m in re.finditer(r"유입구?\S*|분사\S*|쏴\S*|뿜\S*|주입\S*|중심|에서", t):
            p.take(m)

    if re.search(r"유출|빠져나가|나가면|소멸|배수구", t) and not DEL:
        b = _box_from(p, "유출 영역")
        if b:
            p.ops.append({"op": "add_outlet", "box": b})
        elif not p.ask:
            p.ask.append("유출 영역 범위를 x, y, z 로 알려 주세요")
        for m in re.finditer(r"유출\s*영역\S*|유출\S*|빠져나가\S*|나가면|소멸\S*|배수구", t):
            p.take(m)

    for m in re.finditer(r"((?:fb|in|out|pl|st|mo)\d+)\s*(?:을|를)?\s*(?:삭제|지워|제거|없애)\S*", t):
        p.ops.append({"op": "remove", "id": m.group(1)})
        p.take(m)
    for m in re.finditer(r"(유체\s*블록|유입구|유출\s*영역|그래프|정지\s*조건)?\s*(?:을|를)?\s*(전부|모두|전체|다)?\s*(?:삭제|지워|비워|제거)\S*", t):
        if not m.group(1) and not m.group(2):
            continue
        if re.match(r"\s*(?:fb|in|out|pl|st|mo)\d", t[max(0, m.start() - 6):m.start()] or " "):
            continue
        what = {"유체블록": "fluid_blocks", "유입구": "inlets", "유출영역": "outlets", "그래프": "plots", "정지조건": "stops"}.get((m.group(1) or "").replace(" ", ""), "all")
        p.ops.append({"op": "clear", "what": what})
        p.take(m)

    if geom is not None:
        for m in re.finditer(r"(?:그룹\s*)?(\S+?)\s*(?:을|를|은|는)?\s*(무시|빼|제외|벽으로|벽)\S*", t):
            g = m.group(1)
            if g in geom.group_names():
                p.ops.append({"op": "set_body", "group": g, "role": "ignore" if m.group(2) in ("무시", "빼", "제외") else "wall"})
                p.take(m)

    for m in re.finditer(r"(?:해석|계산|시뮬레이션)?\s*(?:을|를)?\s*(실행|시작|돌려|run)\S*", t):
        p.action = "solve"
        p.take(m)

    for m in re.finditer(r"(?:해석\s*)?영역\s*(?:을|를)?\s*(?:맞춰|자동|맞춤)\S*", t):
        p.ops.append({"op": "set_domain", "fit": True})
        p.take(m)

    left = p.leftover()
    left = " ".join(w for w in left.split() if w not in ("물", "물을", "유체", "유체를", "영역", "범위", "박스", "블록", "해석", "계산", "방향", "속도", "입자"))
    unknown = [left] if left else []
    if p.ask:
        reply = " / ".join(p.ask)
    elif p.ops or p.action:
        reply = "알겠습니다."
    else:
        reply = ""
    info = {"complete": not unknown and not p.ask and bool(p.ops or p.action), "ask": p.ask, "unknown": unknown, "action": p.action, "answered": False}
    return p.ops, reply, info
