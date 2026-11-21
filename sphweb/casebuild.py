import numpy as np


def lattice(lo, hi, dx):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    i0 = np.ceil(lo / dx - 0.5 - 1e-9).astype(int)
    i1 = np.floor(hi / dx - 0.5 + 1e-9).astype(int)
    if np.any(i1 < i0):
        return np.zeros((0, 3))
    ax = [dx * (np.arange(i0[k], i1[k] + 1) + 0.5) for k in range(3)]
    g = np.meshgrid(*ax, indexing="ij")
    return np.stack([a.ravel() for a in g], 1)


def inside(points, V, F):
    if len(points) == 0 or len(F) == 0:
        return np.zeros(len(points), bool)
    xy = np.round(points[:, :2], 6)
    cols, inv = np.unique(xy, axis=0, return_inverse=True)
    inv = inv.ravel()
    T = V[F]
    zr = (points[:, 2].min() - 1.0, points[:, 2].max() + 1.0)
    span = max(zr[1] - zr[0], 1e-9)
    hc, hz = [], []
    for t in T:
        a, b, c = t
        d = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(d) < 1e-12:
            continue
        lo, hi = t[:, :2].min(0), t[:, :2].max(0)
        m = np.all((cols >= lo - 1e-9) & (cols <= hi + 1e-9), 1)
        if not m.any():
            continue
        p = cols[m]
        l1 = ((b[0] - p[:, 0]) * (c[1] - p[:, 1]) - (c[0] - p[:, 0]) * (b[1] - p[:, 1])) / d
        l2 = ((c[0] - p[:, 0]) * (a[1] - p[:, 1]) - (a[0] - p[:, 0]) * (c[1] - p[:, 1])) / d
        l3 = 1.0 - l1 - l2
        ok = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not ok.any():
            continue
        z = l1[ok] * a[2] + l2[ok] * b[2] + l3[ok] * c[2]
        hc.append(np.nonzero(m)[0][ok])
        hz.append(z)
    if not hc:
        return np.zeros(len(points), bool)
    hc = np.concatenate(hc)
    hz = np.concatenate(hz)
    u = np.unique(np.stack([hc.astype(float), np.round(hz, 6)], 1), axis=0)
    hc, hz = u[:, 0].astype(int), u[:, 1]
    key = hc + np.clip((hz - zr[0]) / span, 0, 0.999999)
    key.sort()
    pk = inv + np.clip((points[:, 2] - zr[0]) / span, 0, 0.999999)
    above = np.searchsorted(key, inv + 1.0) - np.searchsorted(key, pk)
    return above % 2 == 1


def components(F):
    F = np.asarray(F, int)
    if not len(F):
        return np.zeros(0, int)
    par = np.arange(F.max() + 1)

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for t in F:
        r0 = find(t[0])
        for v in t[1:]:
            r = find(v)
            if r != r0:
                par[r] = r0
    roots = np.array([find(t[0]) for t in F])
    return np.unique(roots, return_inverse=True)[1].ravel()


def make_inside(V, F):
    comp = components(F)
    parts = [F[comp == c] for c in np.unique(comp)]

    def fn(points):
        out = np.zeros(len(points), bool)
        for P in parts:
            out |= inside(points, V, P)
        return out

    return fn


def internal_faces(V, F, ins, eps):
    c = V[F].mean(1)
    n = tri_normals(V, F)
    return ins(c + eps * n) & ins(c - eps * n)


def wet_distance(lo, hi, dx, V, F, seeds, max_iter=2000, ins=None):
    i0 = np.ceil(lo / dx - 0.5 - 1e-9).astype(int)
    i1 = np.floor(hi / dx - 0.5 + 1e-9).astype(int)
    shape = tuple(int(v) for v in (i1 - i0 + 1))
    pts = lattice(lo, hi, dx)
    solid = (ins(pts) if ins else inside(pts, V, F)).reshape(shape)
    D = np.full(shape, np.iinfo(np.int32).max, np.int32)
    k = np.round(seeds / dx - 0.5).astype(int) - i0
    ok = np.all((k >= 0) & (k < shape), 1)
    k = k[ok]
    front = np.zeros(shape, bool)
    front[k[:, 0], k[:, 1], k[:, 2]] = True
    front &= ~solid
    D[front] = 0
    seen = front.copy()
    it = 0
    while front.any() and it < max_iter:
        it += 1
        nxt = np.zeros(shape, bool)
        nxt[1:] |= front[:-1]
        nxt[:-1] |= front[1:]
        nxt[:, 1:] |= front[:, :-1]
        nxt[:, :-1] |= front[:, 1:]
        nxt[:, :, 1:] |= front[:, :, :-1]
        nxt[:, :, :-1] |= front[:, :, 1:]
        nxt &= ~seen & ~solid
        D[nxt] = it
        seen |= nxt
        front = nxt

    def lookup(q):
        kk = np.round(q / dx - 0.5).astype(int) - i0
        inb = np.all((kk >= 0) & (kk < shape), 1)
        out = np.full(len(q), np.iinfo(np.int32).max, np.int64)
        kk = kk[inb]
        out[inb] = D[kk[:, 0], kk[:, 1], kk[:, 2]]
        return out

    return lookup


def tri_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def closest(points, V, F, chunk=200000, avoid=None, tn=None, behind=False):
    n = len(points)
    best = np.full(n, np.inf)
    cp = np.zeros((n, 3))
    tri = np.full(n, -1)
    if n == 0:
        return cp, best, tri
    T = V[F]
    per = max(1, chunk // max(n, 1))
    for s in range(0, len(T), per):
        A, B, C = T[s : s + per, 0], T[s : s + per, 1], T[s : s + per, 2]
        q = _closest_tri(points[:, None, :], A[None], B[None], C[None])
        d = np.linalg.norm(q - points[:, None, :], axis=2)
        if avoid is not None:
            d = np.where(avoid @ tn[s : s + per].T > -0.5, np.inf, d)
        if behind:
            side = ((points[:, None, :] - A[None]) * tn[s : s + per][None]).sum(-1)
            d = np.where(side >= 0, np.inf, d)
        k = d.argmin(1)
        dk = d[np.arange(n), k]
        better = dk < best
        best[better] = dk[better]
        cp[better] = q[np.arange(n), k][better]
        tri[better] = s + k[better]
    return cp, best, tri


def _closest_tri(p, a, b, c):
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp_ = p - c
    d5 = (ab * cp_).sum(-1)
    d6 = (ac * cp_).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v = vb / den
    w = vc / den
    res = a + ab * v[..., None] + ac * w[..., None]
    sel = lambda m, x: np.where(m[..., None], x, res)
    t = d2 / np.where(np.abs(d2 - d6) < 1e-30, 1e-30, d2 - d6)
    res = sel((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * t[..., None])
    t = d1 / np.where(np.abs(d1 - d3) < 1e-30, 1e-30, d1 - d3)
    res = sel((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * t[..., None])
    t = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-30, 1e-30, (d4 - d3) + (d5 - d6))
    res = sel((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b + (c - b) * t[..., None])
    res = sel((d6 >= 0) & (d5 <= d6), np.broadcast_to(c, res.shape))
    res = sel((d3 >= 0) & (d4 <= d3), np.broadcast_to(b, res.shape))
    res = sel((d1 <= 0) & (d2 <= 0), np.broadcast_to(a, res.shape))
    return res


def build(spec, geom=None, layers=4):
    dx = float(spec["particle"]["dx"])
    lo = np.asarray(spec["domain"]["min"], float)
    hi = np.asarray(spec["domain"]["max"], float)
    dim = int(spec.get("dim", 3))
    y0 = None
    if dim == 2:
        y0 = dx * (np.round(float(spec.get("plane_y", 0.5 * (lo[1] + hi[1]))) / dx - 0.5) + 0.5)
        lo[1] = hi[1] = y0
    warn = []
    dry = np.zeros((0, 3))
    bound = np.zeros((0, 3))
    normals = np.zeros((0, 3))
    group = np.zeros(0, int)
    names = []
    V = F = None
    if geom is not None and len(geom.faces):
        walls = [b["group"] for b in spec.get("bodies", []) if b.get("role") == "wall"]
        names = walls
        F_all, tg = geom.triangles()
        tg = np.asarray(tg)
        keep = np.isin(tg, walls)
        if keep.any():
            if not geom.closed():
                warn.append("형상이 닫힌 표면이 아니어서 벽 입자를 만들지 못했습니다. 얇은 판·열린 표면은 3단계에서 지원합니다")
            else:
                V = np.asarray(geom.vertices, float)
                ins = make_inside(V, F_all)
                gmin, gmax = V.min(0), V.max(0)
                cand = lattice(np.maximum(gmin, lo), np.minimum(gmax, hi), dx)
                solid = cand[ins(cand)]
                keep &= ~internal_faces(V, F_all, ins, 0.25 * dx)
                Fw = F_all[keep]
                cp, dist, tri = closest(solid, V, Fw)
                band = dist <= layers * dx
                solid, cp, dist, tri = solid[band], cp[band], dist[band], tri[band]
                tnw = tri_normals(V, Fw)
                u = np.where(dist[:, None] < 1e-6, tnw[tri], (cp - solid) / np.maximum(dist[:, None], 1e-12))
                cp2, dist2, tri2 = closest(solid, V, Fw, avoid=u, tn=tnw)
                seeds = np.vstack([lattice(np.maximum(b["box"]["min"], lo), np.minimum(b["box"]["max"], hi), dx) for b in spec.get("fluid_blocks", [])] or [np.zeros((0, 3))])
                extra = np.zeros((0, 3))
                ecp = np.zeros((0, 3))
                etri = np.zeros(0, int)
                if len(seeds):
                    look = wet_distance(lo, hi, dx, V, F_all, seeds, ins=ins)
                    d1 = look(2 * cp - solid)
                    d2 = np.where(np.isfinite(dist2) & (dist2 <= layers * dx), look(2 * cp2 - solid), np.iinfo(np.int64).max)
                    flip = d2 < d1
                    cp[flip], tri[flip] = cp2[flip], tri2[flip]
                    pad = layers * dx
                    near = lattice(np.maximum(gmin - pad, lo), np.minimum(gmax + pad, hi), dx)
                    near = near[~ins(near)]
                    dn = look(near)
                    near, dn = near[dn > 2 * layers], dn[dn > 2 * layers]
                    if len(near) and len(solid):
                        from scipy.spatial import cKDTree

                        _, k = cKDTree(solid).query(near)
                        bcp, btri = cp[k], tri[k]
                        into = bcp - solid[k]
                        side = ((near - bcp) * into).sum(1) < 0
                        bd = np.linalg.norm(near - bcp, axis=1)
                        gh = 2 * bcp - near
                        ok = side & (bd <= layers * dx) & (bd > 0.5 * dx) & ~ins(gh) & (look(gh) + layers < dn)
                        extra, ecp, etri = near[ok], bcp[ok], btri[ok]
                        dry = near[side & (bd > 0.5 * dx) & (look(gh) + layers < dn)]
                    if len(extra):
                        warn.append(f"벽이 입자 간격에 비해 얇아 바깥쪽에 경계 입자 {len(extra):,}개를 더해 {layers}층을 맞췄습니다")
                bound = np.vstack([solid, extra])
                allcp = np.vstack([cp, ecp])
                alltri = np.concatenate([tri, etri])
                bad = np.nonzero(ins(2 * allcp - bound))[0]
                if len(bad):
                    q = _closest_tri(bound[bad][:, None, :], V[Fw[:, 0]][None], V[Fw[:, 1]][None], V[Fw[:, 2]][None])
                    d = np.linalg.norm(q - bound[bad][:, None, :], axis=2)
                    gh = 2 * q - bound[bad][:, None, :]
                    okg = ~ins(gh.reshape(-1, 3)).reshape(d.shape) & (d > 1e-9)
                    d = np.where(okg, d, np.inf)
                    k = d.argmin(1)
                    fine = np.isfinite(d[np.arange(len(bad)), k])
                    allcp[bad[fine]] = q[np.arange(len(bad)), k][fine]
                    alltri[bad[fine]] = k[fine]
                cp, ecp = allcp[: len(solid)], allcp[len(solid):]
                tri, etri = alltri[: len(solid)], alltri[len(solid):]
                normals = 2.0 * (np.vstack([cp, ecp]) - bound)
                if dim == 2:
                    normals[:, 1] = 0.0
                group = np.array([walls.index(g) for g in tg[keep][np.concatenate([tri, etri])]], int)
    fl, zs = [], []
    blocks = spec.get("fluid_blocks", [])
    for b in blocks:
        pts = lattice(np.maximum(b["box"]["min"], lo), np.minimum(b["box"]["max"], hi), dx)
        if len(pts):
            bot = pts[pts[:, 2] <= pts[:, 2].min() + 1e-6] - [0.0, 0.0, dx]
            held = np.zeros(len(bot), bool)
            if V is not None:
                held |= ins(bot)
                _, dd, _ = closest(bot, V, F_all)
                held |= dd < dx
            for o in blocks:
                if o is not b:
                    held |= np.all((bot >= np.asarray(o["box"]["min"]) - 1e-6) & (bot <= np.asarray(o["box"]["max"]) + 1e-6), 1)
            fl.append(pts)
            zs.append(np.full(len(pts), pts[:, 2].max() if held.mean() > 0.5 else -np.inf))
    if fl:
        fluid, first = np.unique(np.round(np.vstack(fl), 6), axis=0, return_index=True)
        zsurf = np.concatenate(zs)[first]
    else:
        fluid, zsurf = np.zeros((0, 3)), np.zeros(0)
    if len(fluid) and V is not None:
        keepf = ~ins(fluid)
        fluid, zsurf = fluid[keepf], zsurf[keepf]
        _, d, _ = closest(fluid, V, F_all[~internal_faces(V, F_all, ins, 0.25 * dx)])
        keepf = d >= 0.5 * dx - 1e-6
        fluid, zsurf = fluid[keepf], zsurf[keepf]
    if not len(fluid):
        warn.append("유체 입자가 없습니다. 유체 블록을 추가해 주세요")
    if spec.get("inlets"):
        warn.append("유입구는 3단계에서 계산에 연결됩니다. 이번 해석에서는 무시합니다")
    if dim == 2 and not len(fluid) and spec.get("fluid_blocks"):
        warn.append(f"2D 단면 y = {y0:g} mm 에 걸친 유체 블록이 없습니다. 단면 위치를 유체 블록 안으로 옮겨 주세요")
    mask = None
    if V is not None:
        i0 = np.ceil(lo / dx - 0.5 - 1e-9).astype(int)
        i1 = np.floor(hi / dx - 0.5 + 1e-9).astype(int)
        shape = tuple(int(v) for v in (i1 - i0 + 1))
        sol = ins(lattice(lo, hi, dx)).reshape(shape)
        pad = np.pad(sol, 1, constant_values=True)
        deep = sol.copy()
        for ax in range(3):
            for sh in (-1, 1):
                deep &= np.roll(pad, sh, axis=ax)[1:-1, 1:-1, 1:-1]
        kd = np.round(dry / dx - 0.5).astype(int) - i0
        kd = kd[np.all((kd >= 0) & (kd < shape), 1)]
        deep[kd[:, 0], kd[:, 1], kd[:, 2]] = True
        mask = {"origin": (dx * (i0 + 0.5)).tolist(), "h": dx, "mask": deep}
    return {"solid": mask, "fluid": fluid, "zsurf": zsurf, "bound": bound, "normals": normals, "group": group, "group_names": names, "warnings": warn, "dx": dx, "domain": (lo, hi), "dim": dim, "plane_y": y0}

