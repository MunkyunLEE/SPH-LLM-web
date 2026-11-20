import math
import threading
import time

import numpy as np
import taichi as ti

from .consts import Params

_INIT = {}

BOUND, FLUID, OUT = 0, 1, 2


def init(arch="cpu", **kw):
    me = threading.get_ident()
    if _INIT and _INIT["thread"] == me:
        return _INIT["arch"]
    if _INIT:
        ti.reset()
        arch = _INIT["arch"]
    ti.init(arch=getattr(ti, arch), offline_cache=True, log_level=ti.WARN, default_fp=ti.f32, **kw)
    _INIT.update(arch=arch, thread=me)
    return arch


@ti.data_oriented
class Solver:
    def __init__(self, params: Params, fluid, bound, domain=None, fluid_vel=None, pos64=None, bound_normals=None, cell_div=1, batch=64):
        self.p = params
        fluid = np.asarray(fluid, dtype=np.float64).reshape(-1, 3)
        bound = np.asarray(bound, dtype=np.float64).reshape(-1, 3)
        self.nb = len(bound)
        self.nf = len(fluid)
        self.n = self.nb + self.nf
        allp = np.vstack([bound, fluid])
        if domain is None:
            lo, hi = allp.min(0), allp.max(0)
            pad = params.support
            domain = (lo - pad, hi + pad + 0.5 * (hi - lo))
        lo, hi = np.array(domain[0], float), np.array(domain[1], float)
        self.dim = int(params.dim)
        if self.dim == 2:
            yc = float(allp[:, 1].mean())
            lo[1], hi[1] = yc - 0.5 * params.support, yc + 0.5 * params.support
        self.origin = lo
        self.cd = int(cell_div)
        self.cs = params.support / self.cd
        self.dims = np.maximum(np.ceil((hi - lo) / self.cs).astype(int), 1)
        self.ncell = int(np.prod(self.dims))
        self.nscan = self.ncell + 1
        self.blk = 1024
        self.nblk = (self.nscan + self.blk - 1) // self.blk
        if pos64 is None:
            pos64 = _INIT.get("arch") != "metal"
        self.pdt = ti.f64 if pos64 else ti.f32
        self.gx, self.gy, self.gz = (float(v) for v in params.gravity)
        self.ox, self.oy, self.oz = (float(v) for v in lo)
        self.cx, self.cy, self.cz = (int(v) for v in self.dims)
        self.inv_cs = 1.0 / self.cs
        self.h = float(params.h)
        self.kr2 = float(params.support ** 2)
        self.vol = float(params.vol)
        self.awen = float(params.awen)
        self.bwen = float(params.bwen)
        self.cs0 = float(params.cs0)
        self.bb = float(params.b)
        self.gam = float(params.gamma)
        self.eta2 = float(params.eta2)
        self.alpha = float(params.visco)
        self.alpha_b = float(params.visco * params.visco_bound)
        self.ddtkh = float(params.ddtkh)
        self.ddtgz = float(params.ddtgz)
        self.use_ddt = params.ddt > 0
        self.laminar = params.visco_type == "laminar"
        self.nu = float(params.visco)
        self.nu_b = float(params.visco * params.visco_bound)
        self.mdbc = bound_normals is not None
        self.rmin = float(params.rho_min)
        self.rmax = float(params.rho_max)
        self.verlet = params.scheme == "verlet"
        self.vsteps = int(params.verlet_steps)
        self.cfl = float(params.cfl)
        self.dtmin = float(params.dt_min)
        self.kmax = int(batch)

        n = self.n
        self.pos = ti.Vector.field(3, self.pdt, n)
        self.pos_pre = ti.Vector.field(3, self.pdt, n)
        self.vel = ti.Vector.field(3, ti.f32, n)
        self.vel_pre = ti.Vector.field(3, ti.f32, n)
        self.rho = ti.field(ti.f32, n)
        self.rho_pre = ti.field(ti.f32, n)
        self.prs = ti.field(ti.f32, n)
        self.ace = ti.Vector.field(3, ti.f32, n)
        self.ar = ti.field(ti.f32, n)
        self.kind = ti.field(ti.i32, n)
        self.cell = ti.field(ti.i32, n)
        self.order = ti.field(ti.i32, n)
        self.pid = ti.field(ti.i32, n)
        self.fwall = ti.Vector.field(3, ti.f32, n)
        self.bnor = ti.Vector.field(3, ti.f32, n)
        self.g_p = ti.Vector.field(3, self.pdt, (n, 2))
        self.g_v = ti.Vector.field(3, ti.f32, (n, 3))
        self.g_s = ti.field(ti.f32, (n, 2))
        self.g_i = ti.field(ti.i32, (n, 3))
        self.count = ti.field(ti.i32, self.nscan)
        self.start = ti.field(ti.i32, self.nscan + 1)
        self.cursor = ti.field(ti.i32, self.nscan)
        self.bsum = ti.field(ti.i32, self.nblk + 1)
        self.stat = ti.field(ti.f32, 4)
        self.nout = ti.field(ti.i32, ())
        self.fcount = ti.field(ti.i32, self.nscan)
        self.wetc = ti.field(ti.i32, self.nscan)
        self.dtv = ti.field(ti.f32, 4)
        self.ring = ti.field(ti.f32, self.kmax)
        self.dthit = ti.field(ti.i32, ())

        kind = np.concatenate([np.full(self.nb, BOUND), np.full(self.nf, FLUID)]).astype(np.int32)
        vel = np.zeros((n, 3), np.float32)
        if fluid_vel is not None:
            vel[self.nb:] = np.asarray(fluid_vel, np.float32).reshape(-1, 3)
        bn = np.zeros((n, 3), np.float32)
        if self.mdbc:
            bn[: self.nb] = np.asarray(bound_normals, np.float32).reshape(-1, 3)
        self.pos.from_numpy(allp.astype(np.float64 if pos64 else np.float32))
        self.vel.from_numpy(vel)
        self.bnor.from_numpy(bn)
        self.rho.fill(1.0)
        self.kind.from_numpy(kind)
        self.pid.from_numpy(np.arange(n, dtype=np.int32))
        self.time = 0.0
        self.nstep = 0
        self.dt_pre = float(params.dt_ini)
        self.dtv[2] = self.dt_pre
        self.dt_last = self.dt_pre
        self.vcount = 0
        self.first = True
        self.dt_warn = 0
        self.wall = 0.0

    @ti.func
    def cell_of(self, x):
        ix = int(ti.floor((x[0] - self.ox) * self.inv_cs))
        iy = int(ti.floor((x[1] - self.oy) * self.inv_cs))
        iz = int(ti.floor((x[2] - self.oz) * self.inv_cs))
        return ti.Vector([ix, iy, iz])

    @ti.kernel
    def _bin(self):
        for c in self.count:
            self.count[c] = 0
            self.fcount[c] = 0
        for i in self.pos:
            c = self.ncell
            if self.kind[i] != OUT:
                g = self.cell_of(self.pos[i])
                if 0 <= g[0] < self.cx and 0 <= g[1] < self.cy and 0 <= g[2] < self.cz:
                    c = (g[2] * self.cy + g[1]) * self.cx + g[0]
                else:
                    self.kind[i] = OUT
            self.cell[i] = c
            ti.atomic_add(self.count[c], 1)
            if self.kind[i] == FLUID:
                ti.atomic_add(self.fcount[c], 1)

    @ti.kernel
    def _scan(self):
        for b in range(self.nblk):
            s = 0
            for k in range(self.blk):
                c = b * self.blk + k
                if c < self.nscan:
                    self.start[c] = s
                    s += self.count[c]
            self.bsum[b] = s
        ti.loop_config(serialize=True)
        for b in range(self.nblk):
            if b == 0:
                self.bsum[self.nblk] = 0
            t = self.bsum[b]
            self.bsum[b] = self.bsum[self.nblk]
            self.bsum[self.nblk] += t
        for c in range(self.nscan):
            self.start[c] += self.bsum[c // self.blk]
            self.cursor[c] = 0
        self.start[self.nscan] = self.bsum[self.nblk]

    @ti.kernel
    def _scatter(self):
        for i in self.cell:
            c = self.cell[i]
            k = ti.atomic_add(self.cursor[c], 1)
            self.order[self.start[c] + k] = i

    @ti.kernel
    def _sort_within(self):
        for c in range(self.nscan):
            a, b = self.start[c], self.start[c + 1]
            for u in range(a + 1, b):
                v = self.order[u]
                kv = ti.select(self.kind[v] == FLUID, v, v + self.n)
                w = u - 1
                while w >= a and ti.select(self.kind[self.order[w]] == FLUID, self.order[w], self.order[w] + self.n) > kv:
                    self.order[w + 1] = self.order[w]
                    w -= 1
                self.order[w + 1] = v

    @ti.kernel
    def _gather_all(self):
        for s in range(self.n):
            o = self.order[s]
            self.g_p[s, 0] = self.pos[o]
            self.g_p[s, 1] = self.pos_pre[o]
            self.g_v[s, 0] = self.vel[o]
            self.g_v[s, 1] = self.vel_pre[o]
            self.g_v[s, 2] = self.bnor[o]
            self.g_s[s, 0] = self.rho[o]
            self.g_s[s, 1] = self.rho_pre[o]
            self.g_i[s, 0] = self.kind[o]
            self.g_i[s, 1] = self.pid[o]
            self.g_i[s, 2] = self.cell[o]
        for s in range(self.n):
            self.pos[s] = self.g_p[s, 0]
            self.pos_pre[s] = self.g_p[s, 1]
            self.vel[s] = self.g_v[s, 0]
            self.vel_pre[s] = self.g_v[s, 1]
            self.bnor[s] = self.g_v[s, 2]
            self.rho[s] = self.g_s[s, 0]
            self.rho_pre[s] = self.g_s[s, 1]
            self.kind[s] = self.g_i[s, 0]
            self.pid[s] = self.g_i[s, 1]
            self.cell[s] = self.g_i[s, 2]

    def _reorder(self):
        self._gather_all()

    def hydrostatic(self, zsurf, bound=True):
        z = self.pos.to_numpy()[:, 2].astype(np.float64)
        g = abs(self.p.gravity[2])
        r = np.power(1.0 + g * np.clip(zsurf - z, 0, None) / self.p.b, 1.0 / self.p.gamma)
        if not bound:
            r[self.pid.to_numpy() < self.nb] = 1.0
        self.rho.from_numpy(r.astype(np.float32))

    def neighbors(self):
        self._bin()
        self._scan()
        self._scatter()
        self._sort_within()
        self._reorder()
        self._wetcells()

    @ti.func
    def _row(self, g, dy: ti.template(), dz: ti.template()):
        a = 0
        b = 0
        y, z = g[1] + dy, g[2] + dz
        if 0 <= y < self.cy and 0 <= z < self.cz:
            x0 = ti.max(g[0] - self.cd, 0)
            x1 = ti.min(g[0] + self.cd, self.cx - 1)
            if x0 <= x1:
                base = (z * self.cy + y) * self.cx
                a = self.start[base + x0]
                b = self.start[base + x1 + 1]
        return a, b

    @ti.kernel
    def _wetcells(self):
        for c in range(self.ncell):
            gx = c % self.cx
            gy = (c // self.cx) % self.cy
            gz = c // (self.cx * self.cy)
            n = 0
            for dz in ti.static(range(-self.cd, self.cd + 1)):
                for dy in ti.static(range(-self.cd, self.cd + 1)):
                    y, z = gy + dy, gz + dz
                    if 0 <= y < self.cy and 0 <= z < self.cz:
                        base = (z * self.cy + y) * self.cx
                        for dx in ti.static(range(-self.cd, self.cd + 1)):
                            x = gx + dx
                            if 0 <= x < self.cx:
                                n += self.fcount[base + x]
            self.wetc[c] = n

    @ti.func
    def _wet(self, g):
        n = 0
        if 0 <= g[0] < self.cx and 0 <= g[1] < self.cy and 0 <= g[2] < self.cz:
            n = self.wetc[(g[2] * self.cy + g[1]) * self.cx + g[0]]
        return n

    @ti.kernel
    def _count_nb(self, out: ti.types.ndarray()):
        for i in self.pos:
            n = 0
            if self.kind[i] != OUT:
                g = self.cell_of(self.pos[i])
                for dz in ti.static(range(-self.cd, self.cd + 1)):
                    for dy in ti.static(range(-self.cd, self.cd + 1)):
                        a, b = self._row(g, dy, dz)
                        for j in range(a, b):
                            dr = ti.cast(self.pos[i] - self.pos[j], ti.f32)
                            rr2 = dr.dot(dr)
                            if rr2 <= self.kr2 and rr2 >= 1e-18:
                                n += 1
            out[i] = n

    def neighbor_counts(self):
        self.neighbors()
        out = np.zeros(self.n, np.int32)
        self._count_nb(out)
        res = np.empty_like(out)
        res[self.pid.to_numpy()] = out
        return res

    @ti.kernel
    def _eos(self):
        for i in self.rho:
            self.prs[i] = self.bb * (ti.pow(self.rho[i], self.gam) - 1.0)

    @ti.kernel
    def _interact(self):
        for k in range(4):
            self.stat[k] = 0.0
        for i in self.fwall:
            self.fwall[i] = ti.Vector([0.0, 0.0, 0.0])
        for i in self.pos:
            ki = self.kind[i]
            if ki == FLUID:
                self._one(i, ki)
            elif ki == BOUND:
                if self._wet(self.cell_of(self.pos[i])) > 0:
                    self._one_bound(i)
                else:
                    self.ar[i] = 0.0
                    self.ace[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def _one_bound(self, i):
        xi = self.pos[i]
        vi = self.vel[i]
        ri = self.rho[i]
        g = self.cell_of(xi)
        ar = 0.0
        visc = 0.0
        for dz in ti.static(range(-self.cd, self.cd + 1)):
            for dy in ti.static(range(-self.cd, self.cd + 1)):
                y, z = g[1] + dy, g[2] + dz
                if 0 <= y < self.cy and 0 <= z < self.cz:
                    base = (z * self.cy + y) * self.cx
                    for x in range(ti.max(g[0] - self.cd, 0), ti.min(g[0] + self.cd, self.cx - 1) + 1):
                        c = base + x
                        for j in range(self.start[c], self.start[c] + self.fcount[c]):
                            dr = ti.cast(xi - self.pos[j], ti.f32)
                            rr2 = dr.dot(dr)
                            if rr2 <= self.kr2 and rr2 >= 1e-18:
                                q = ti.sqrt(rr2) / self.h
                                w1 = 1.0 - 0.5 * q
                                fr = self.bwen * w1 * w1 * w1 * dr
                                dv = vi - self.vel[j]
                                ar += self.vol * dv.dot(fr) * (ri / self.rho[j])
                                visc = ti.max(visc, dr.dot(dv) / (rr2 + self.eta2))
        self.ar[i] = ar
        self.ace[i] = ti.Vector([0.0, 0.0, 0.0])
        ti.atomic_max(self.stat[1], visc)

    @ti.func
    def _one(self, i, ki):
        xi = self.pos[i]
        vi = self.vel[i]
        ri = self.rho[i]
        pi_ = self.prs[i]
        g = self.cell_of(xi)
        acc = ti.Vector([0.0, 0.0, 0.0])
        ar = 0.0
        dd = 0.0
        ddt_on = 1
        visc = 0.0
        for dz in ti.static(range(-self.cd, self.cd + 1)):
            for dy in ti.static(range(-self.cd, self.cd + 1)):
                a0, b0 = self._row(g, dy, dz)
                for j in range(a0, b0):
                    kj = self.kind[j]
                    if ki == BOUND and kj != FLUID:
                        continue
                    dr = ti.cast(xi - self.pos[j], ti.f32)
                    rr2 = dr.dot(dr)
                    if rr2 <= self.kr2 and rr2 >= 1e-18:
                        q = ti.sqrt(rr2) / self.h
                        w1 = 1.0 - 0.5 * q
                        fac = self.bwen * w1 * w1 * w1
                        fr = fac * dr
                        rj = self.rho[j]
                        dv = vi - self.vel[j]
                        ar += self.vol * dv.dot(fr) * (ri / rj)
                        dot = dr.dot(dv)
                        visc = ti.max(visc, dot / (rr2 + self.eta2))
                        if ki == FLUID:
                            prs = (pi_ + self.prs[j]) / (ri * rj)
                            da = -prs * self.vol * fr
                            if ti.static(self.laminar):
                                nu = ti.select(kj == FLUID, self.nu, self.nu_b)
                                da += self.vol * 4.0 * nu / ((rr2 + self.eta2) * (ri + rj)) * dr.dot(fr) * dv
                            else:
                                if dot < 0:
                                    a = ti.select(kj == FLUID, self.alpha, self.alpha_b)
                                    amubar = self.h * dot / (rr2 + self.eta2)
                                    robar = 0.5 * (ri + rj)
                                    pv = (-a * self.cs0 * amubar / robar) * self.vol
                                    da -= pv * fr
                            acc += da
                            if kj == BOUND:
                                ti.atomic_add(self.fwall[j], -da)
                            if ti.static(self.use_ddt):
                                if kj != FLUID:
                                    ddt_on = 0
                                else:
                                    rh = 1.0 + self.ddtgz * dr[2]
                                    drho = ti.pow(rh, 1.0 / self.gam) - 1.0
                                    vd = self.ddtkh * self.cs0 * ((rj - ri) - drho) / (rr2 + self.eta2)
                                    dd -= vd * dr.dot(fr) * self.vol / rj
        if ki == FLUID and ddt_on == 1:
            ar += dd
        self.ar[i] = ar
        self.ace[i] = acc
        ti.atomic_max(self.stat[1], visc)
        if ki == FLUID:
            ti.atomic_max(self.stat[0], acc.norm())
            ti.atomic_max(self.stat[2], vi.norm())

    @ti.kernel
    def _mdbc(self):
        for i in self.pos:
            if self.kind[i] == BOUND:
                bn = self.bnor[i]
                gp = self.pos[i] + ti.cast(bn, self.pdt)
                g = self.cell_of(gp)
                if bn.norm() > 0 and self._wet(g) > 0:
                    m = ti.Matrix.zero(ti.f32, 4, 4)
                    rv = ti.Vector.zero(ti.f32, 4)
                    for dz in ti.static(range(-self.cd, self.cd + 1)):
                        for dy in ti.static(range(-self.cd, self.cd + 1)):
                            y, z = g[1] + dy, g[2] + dz
                            if 0 <= y < self.cy and 0 <= z < self.cz:
                                base = (z * self.cy + y) * self.cx
                                for x in range(ti.max(g[0] - self.cd, 0), ti.min(g[0] + self.cd, self.cx - 1) + 1):
                                    c = base + x
                                    for j in range(self.start[c], self.start[c] + self.fcount[c]):
                                        dr = ti.cast(gp - self.pos[j], ti.f32)
                                        rr2 = dr.dot(dr)
                                        if rr2 <= self.kr2:
                                            q = ti.sqrt(rr2) / self.h
                                            w1 = 1.0 - 0.5 * q
                                            wab = self.awen * (2.0 * q + 1.0) * w1 * w1 * w1 * w1
                                            fr = self.bwen * w1 * w1 * w1 * dr
                                            vj = self.vol / self.rho[j]
                                            rv += ti.Vector([wab, fr[0], fr[1], fr[2]]) * self.vol
                                            row = ti.Vector([1.0, dr[0], dr[1], dr[2]])
                                            col = ti.Vector([wab, fr[0], fr[1], fr[2]]) * vj
                                            m += col.outer_product(row)
                    if ti.static(self.dim == 2):
                        m[2, 2] = 1.0
                    rf = 1.0
                    det = m.determinant()
                    if ti.abs(det) >= 1e-3:
                        inv = m.inverse()
                        sol = inv @ rv
                        rf = sol[0] + sol[1] * bn[0] + sol[2] * bn[1] + sol[3] * bn[2]
                    elif m[0, 0] > 0:
                        rf = rv[0] / m[0, 0]
                    self.rho[i] = rf

    @ti.kernel
    def _save_pre(self):
        for i in self.pos:
            self.pos_pre[i] = self.pos[i]
            self.vel_pre[i] = self.vel[i]
            self.rho_pre[i] = self.rho[i]

    @ti.kernel
    def _dtk(self, slot: ti.i32):
        acemax = self.stat[0]
        viscmax = self.stat[1]
        velmax = self.stat[2]
        dt1 = 1e30
        if acemax > 0:
            dt1 = ti.sqrt(self.h / acemax)
        dt2 = self.h / (ti.max(self.cs0, 10.0 * velmax) + self.h * viscmax)
        dt = self.cfl * ti.min(dt1, dt2)
        if dt < self.dtmin:
            dt = self.dtmin
            self.dthit[None] += 1
        if ti.math.isnan(dt):
            dt = -1.0
        self.dtv[slot] = dt

    @ti.kernel
    def _sym_end(self, k: ti.i32):
        self.ring[k] = self.dtv[2]
        self.dtv[2] = ti.min(self.dtv[0], self.dtv[1])

    @ti.kernel
    def _predict(self):
        dt = self.dtv[2]
        h = 0.5 * dt
        for i in self.pos:
            k = self.kind[i]
            if k == BOUND:
                self.rho[i] = ti.max(1.0, self.rho_pre[i] + h * self.ar[i])
            elif k == FLUID:
                self.rho[i] = self.rho_pre[i] + h * self.ar[i]
                self.pos[i] = self.pos_pre[i] + ti.cast(self.vel_pre[i] * h, self.pdt)
                self.vel[i] = self.vel_pre[i] + (self.ace[i] + ti.Vector([self.gx, self.gy, self.gz])) * h
                self._check(i)

    @ti.kernel
    def _correct(self):
        dt = self.dtv[2]
        h = 0.5 * dt
        for i in self.pos:
            k = self.kind[i]
            if k != OUT:
                eps = -(self.ar[i] / self.rho[i]) * dt
                rn = self.rho_pre[i] * (2.0 - eps) / (2.0 + eps)
                if k == BOUND:
                    self.rho[i] = ti.max(1.0, rn)
                else:
                    self.rho[i] = rn
                    vn = self.vel_pre[i] + (self.ace[i] + ti.Vector([self.gx, self.gy, self.gz])) * dt
                    self.pos[i] = self.pos_pre[i] + ti.cast((self.vel_pre[i] + vn) * h, self.pdt)
                    self.vel[i] = vn
                    self._check(i)

    @ti.kernel
    def _verlet(self, k: ti.i32, euler: ti.i32):
        dt = self.dtv[0]
        self.ring[k] = dt
        d2 = ti.select(euler == 1, dt, 2.0 * dt)
        g = ti.Vector([self.gx, self.gy, self.gz])
        for i in self.pos:
            kk = self.kind[i]
            if kk == BOUND:
                base = ti.select(euler == 1, self.rho[i], self.rho_pre[i])
                rn = ti.max(1.0, base + d2 * self.ar[i])
                self.rho_pre[i] = self.rho[i]
                self.rho[i] = rn
            elif kk == FLUID:
                a = self.ace[i] + g
                vb = self.vel[i]
                rb = self.rho[i]
                if euler == 0:
                    vb = self.vel_pre[i]
                    rb = self.rho_pre[i]
                self.pos[i] = self.pos[i] + ti.cast(self.vel[i] * dt + 0.5 * dt * dt * a, self.pdt)
                self.vel_pre[i] = self.vel[i]
                self.rho_pre[i] = self.rho[i]
                self.vel[i] = vb + d2 * a
                self.rho[i] = rb + d2 * self.ar[i]
                self._check(i)

    @ti.func
    def _check(self, i):
        r = self.rho[i]
        if r < self.rmin or r > self.rmax or ti.math.isnan(r):
            self.kind[i] = OUT
            self.vel[i] = ti.Vector([0.0, 0.0, 0.0])
            self.ace[i] = ti.Vector([0.0, 0.0, 0.0])
            self.nout[None] += 1

    def _one_step(self, k):
        if self.verlet:
            if self.first:
                self._save_pre()
                self.first = False
            self.neighbors()
            if self.mdbc:
                self._mdbc()
            self._eos()
            self._interact()
            self._dtk(0)
            self.vcount += 1
            euler = 1 if self.vcount >= self.vsteps else 0
            if euler:
                self.vcount = 0
            self._verlet(k, euler)
        else:
            self.neighbors()
            if self.mdbc:
                self._mdbc()
            self._save_pre()
            self._eos()
            self._interact()
            self._dtk(0)
            self._predict()
            self.neighbors()
            self._eos()
            self._interact()
            self._dtk(1)
            self._correct()
            self._sym_end(k)

    def advance(self, nsteps=1):
        t0 = time.perf_counter()
        n = max(1, min(int(nsteps), self.kmax))
        for k in range(n):
            self._one_step(k)
        dts = self.ring.to_numpy()[:n].astype(np.float64)
        if not np.all(np.isfinite(dts)) or np.any(dts <= 0):
            raise FloatingPointError("dt 계산 실패 (발산)")
        self.time += float(dts.sum())
        self.nstep += n
        self.dt_last = float(dts[-1])
        self.dt_pre = float(self.dtv[2]) if not self.verlet else self.dt_last
        self.dt_warn = int(self.dthit[None])
        self.wall += time.perf_counter() - t0
        return n

    def step(self):
        self.advance(1)
        return self.dt_last

    def steps_until(self, t_target, cap=None):
        dt = max(self.dt_last, 1e-30)
        k = int((t_target - self.time) / dt)
        return max(1, min(k, cap or self.kmax))

    @ti.kernel
    def _sample(self, pts: ti.types.ndarray(), out: ti.types.ndarray()):
        for m in range(pts.shape[0]):
            x = ti.Vector([pts[m, 0], pts[m, 1], pts[m, 2]])
            g = self.cell_of(x)
            den = 0.0
            u = ti.Vector([0.0, 0.0, 0.0])
            pr = 0.0
            for dz in ti.static(range(-self.cd, self.cd + 1)):
                for dy in ti.static(range(-self.cd, self.cd + 1)):
                    a0, b0 = self._row(g, dy, dz)
                    for j in range(a0, b0):
                        if self.kind[j] == FLUID:
                            dr = ti.cast(x - self.pos[j], ti.f32)
                            rr2 = dr.dot(dr)
                            if rr2 <= self.kr2:
                                q = ti.sqrt(rr2) / self.h
                                w1 = 1.0 - 0.5 * q
                                wv = self.awen * (2.0 * q + 1.0) * w1 * w1 * w1 * w1 * self.vol / self.rho[j]
                                den += wv
                                u += wv * self.vel[j]
                                pr += wv * self.prs[j]
            if den > 1e-6:
                u /= den
                pr /= den
            out[m, 0] = u[0]
            out[m, 1] = u[1]
            out[m, 2] = u[2]
            out[m, 3] = pr
            out[m, 4] = den

    def sample(self, points):
        pts = np.ascontiguousarray(np.asarray(points, np.float64).reshape(-1, 3))
        out = np.zeros((len(pts), 5), np.float32)
        if len(pts):
            self._eos()
            self._sample(pts, out)
        return out[:, :3], out[:, 3] * self.p.rho0, out[:, 4]

    def wall_forces(self, group=None, ngroup=1):
        inv = np.empty(self.n, np.int64)
        inv[self.pid.to_numpy()] = np.arange(self.n)
        f = self.fwall.to_numpy()[inv][: self.nb].astype(np.float64) * self.p.mass
        g = np.zeros(self.nb, int) if group is None else np.asarray(group, int)
        out = np.zeros((max(ngroup, 1), 3))
        np.add.at(out, g, f)
        return out

    def run(self, t_end, every=None, callback=None):
        nxt = every if every else math.inf
        while self.time < t_end - 1e-12:
            self.advance(self.steps_until(min(nxt, t_end)))
            if callback and self.time >= nxt - 1e-12:
                nxt += every
                if callback(self) is False:
                    break
        return self

    def snapshot(self, fluid_only=False):
        self._eos()
        inv = np.empty(self.n, np.int64)
        inv[self.pid.to_numpy()] = np.arange(self.n)
        pos = self.pos.to_numpy().astype(np.float64)[inv]
        vel = self.vel.to_numpy()[inv]
        rho = self.rho.to_numpy()[inv]
        kind = self.kind.to_numpy()[inv]
        prs = self.prs.to_numpy()[inv] * self.p.rho0
        out = {"pos": pos, "vel": vel, "rho": rho * self.p.rho0, "rho_rel": rho, "press": prs, "kind": kind}
        if fluid_only:
            out = {k: v[self.nb:] for k, v in out.items()}
        return out

    def info(self):
        return {"time": self.time, "steps": self.nstep, "dt": self.dt_last, "wall": self.wall, "n": self.n, "nfluid": self.nf, "nbound": self.nb, "out": int(self.nout[None]), "dt_min_hits": self.dt_warn, "cells": self.ncell}
