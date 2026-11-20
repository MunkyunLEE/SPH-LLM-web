import math
import time

import numpy as np
import taichi as ti

from .consts import Params

_INIT = {}

BOUND, FLUID, OUT = 0, 1, 2


def init(arch="cpu", **kw):
    if _INIT:
        return _INIT["arch"]
    a = getattr(ti, arch)
    ti.init(arch=a, offline_cache=True, log_level=ti.WARN, default_fp=ti.f32, **kw)
    _INIT["arch"] = arch
    return arch


@ti.data_oriented
class Solver:
    def __init__(self, params: Params, fluid, bound, domain=None, fluid_vel=None, pos64=None):
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
        lo, hi = np.asarray(domain[0], float), np.asarray(domain[1], float)
        self.origin = lo
        self.cs = params.support
        self.dims = np.maximum(np.ceil((hi - lo) / self.cs).astype(int), 1)
        self.ncell = int(np.prod(self.dims))
        self.blk = 1024
        self.nblk = (self.ncell + self.blk - 1) // self.blk
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
        self.rmin = float(params.rho_min)
        self.rmax = float(params.rho_max)

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
        self.count = ti.field(ti.i32, self.ncell)
        self.start = ti.field(ti.i32, self.ncell + 1)
        self.cursor = ti.field(ti.i32, self.ncell)
        self.bsum = ti.field(ti.i32, self.nblk + 1)
        self.stat = ti.field(ti.f32, 4)
        self.nout = ti.field(ti.i32, ())

        kind = np.concatenate([np.full(self.nb, BOUND), np.full(self.nf, FLUID)]).astype(np.int32)
        vel = np.zeros((n, 3), np.float32)
        if fluid_vel is not None:
            vel[self.nb:] = np.asarray(fluid_vel, np.float32).reshape(-1, 3)
        self.pos.from_numpy(allp.astype(np.float64 if pos64 else np.float32))
        self.vel.from_numpy(vel)
        self.rho.fill(1.0)
        self.kind.from_numpy(kind)
        self.time = 0.0
        self.nstep = 0
        self.dt_pre = float(params.dt_ini)
        self.dt_last = self.dt_pre
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
        for i in self.pos:
            c = -1
            if self.kind[i] != OUT:
                g = self.cell_of(self.pos[i])
                if 0 <= g[0] < self.cx and 0 <= g[1] < self.cy and 0 <= g[2] < self.cz:
                    c = (g[2] * self.cy + g[1]) * self.cx + g[0]
                else:
                    self.kind[i] = OUT
            self.cell[i] = c
            if c >= 0:
                ti.atomic_add(self.count[c], 1)

    @ti.kernel
    def _scan(self):
        for b in range(self.nblk):
            s = 0
            for k in range(self.blk):
                c = b * self.blk + k
                if c < self.ncell:
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
        for c in range(self.ncell):
            self.start[c] += self.bsum[c // self.blk]
            self.cursor[c] = 0
        self.start[self.ncell] = self.bsum[self.nblk]

    @ti.kernel
    def _scatter(self):
        for i in self.cell:
            c = self.cell[i]
            if c >= 0:
                k = ti.atomic_add(self.cursor[c], 1)
                self.order[self.start[c] + k] = i

    def hydrostatic(self, zsurf, bound=True):
        z = self.pos.to_numpy()[:, 2].astype(np.float64)
        g = abs(self.p.gravity[2])
        r = np.power(1.0 + g * np.clip(zsurf - z, 0, None) / self.p.b, 1.0 / self.p.gamma)
        if not bound:
            r[: self.nb] = 1.0
        self.rho.from_numpy(r.astype(np.float32))

    def neighbors(self):
        self._bin()
        self._scan()
        self._scatter()

    @ti.kernel
    def _count_nb(self, out: ti.types.ndarray()):
        for i in self.pos:
            n = 0
            if self.kind[i] != OUT:
                g = self.cell_of(self.pos[i])
                for dz in ti.static(range(-1, 2)):
                    for dy in ti.static(range(-1, 2)):
                        for dx in ti.static(range(-1, 2)):
                            cx, cy, cz = g[0] + dx, g[1] + dy, g[2] + dz
                            if 0 <= cx < self.cx and 0 <= cy < self.cy and 0 <= cz < self.cz:
                                c = (cz * self.cy + cy) * self.cx + cx
                                for s in range(self.start[c], self.start[c + 1]):
                                    j = self.order[s]
                                    dr = ti.cast(self.pos[i] - self.pos[j], ti.f32)
                                    rr2 = dr.dot(dr)
                                    if rr2 <= self.kr2 and rr2 >= 1e-18:
                                        n += 1
            out[i] = n

    def neighbor_counts(self):
        self.neighbors()
        out = np.zeros(self.n, np.int32)
        self._count_nb(out)
        return out

    @ti.kernel
    def _eos(self):
        for i in self.rho:
            self.prs[i] = self.bb * (ti.pow(self.rho[i], self.gam) - 1.0)

    @ti.kernel
    def _interact(self):
        for k in range(4):
            self.stat[k] = 0.0
        for i in self.pos:
            ki = self.kind[i]
            if ki != OUT:
                self._one(i, ki)

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
        for dz in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                for dx in ti.static(range(-1, 2)):
                    cx, cy, cz = g[0] + dx, g[1] + dy, g[2] + dz
                    if 0 <= cx < self.cx and 0 <= cy < self.cy and 0 <= cz < self.cz:
                        c = (cz * self.cy + cy) * self.cx + cx
                        for s in range(self.start[c], self.start[c + 1]):
                            j = self.order[s]
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
                                    acc -= prs * self.vol * fr
                                    if dot < 0:
                                        a = ti.select(kj == FLUID, self.alpha, self.alpha_b)
                                        amubar = self.h * dot / (rr2 + self.eta2)
                                        robar = 0.5 * (ri + rj)
                                        pv = (-a * self.cs0 * amubar / robar) * self.vol
                                        acc -= pv * fr
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
    def _save_pre(self):
        for i in self.pos:
            self.pos_pre[i] = self.pos[i]
            self.vel_pre[i] = self.vel[i]
            self.rho_pre[i] = self.rho[i]

    @ti.kernel
    def _predict(self, dt: ti.f32):
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
    def _correct(self, dt: ti.f32):
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

    @ti.func
    def _check(self, i):
        r = self.rho[i]
        if r < self.rmin or r > self.rmax or ti.math.isnan(r):
            self.kind[i] = OUT
            self.vel[i] = ti.Vector([0.0, 0.0, 0.0])
            self.ace[i] = ti.Vector([0.0, 0.0, 0.0])
            self.nout[None] += 1

    def _dt(self):
        s = self.stat.to_numpy()
        p = self.p
        acemax, viscmax, velmax = float(s[0]), float(s[1]), float(s[2])
        dt1 = math.sqrt(p.h / acemax) if acemax > 0 else math.inf
        dt2 = p.h / (max(p.cs0, 10.0 * velmax) + p.h * viscmax)
        dt = p.cfl * min(dt1, dt2)
        if not math.isfinite(dt):
            raise FloatingPointError(f"dt 계산 실패 (acemax={acemax}, velmax={velmax}, visc={viscmax})")
        if dt < p.dt_min:
            dt = p.dt_min
            self.dt_warn += 1
        return dt

    def step(self):
        t0 = time.perf_counter()
        dt = self.dt_pre
        self.neighbors()
        self._save_pre()
        self._eos()
        self._interact()
        dtp = self._dt()
        self._predict(dt)
        self.neighbors()
        self._eos()
        self._interact()
        dtc = self._dt()
        self._correct(dt)
        self.dt_pre = min(dtp, dtc)
        self.dt_last = dt
        self.time += dt
        self.nstep += 1
        self.wall += time.perf_counter() - t0
        return dt

    def run(self, t_end, every=None, callback=None):
        nxt = every if every else math.inf
        while self.time < t_end - 1e-12:
            self.step()
            if callback and self.time >= nxt - 1e-12:
                nxt += every
                if callback(self) is False:
                    break
        return self

    def snapshot(self, fluid_only=False):
        self._eos()
        pos = self.pos.to_numpy().astype(np.float64)
        vel = self.vel.to_numpy()
        rho = self.rho.to_numpy()
        kind = self.kind.to_numpy()
        prs = self.prs.to_numpy() * self.p.rho0
        out = {"pos": pos, "vel": vel, "rho": rho * self.p.rho0, "rho_rel": rho, "press": prs, "kind": kind}
        if fluid_only:
            out = {k: v[self.nb:] for k, v in out.items()}
        return out

    def info(self):
        return {"time": self.time, "steps": self.nstep, "dt": self.dt_last, "wall": self.wall, "n": self.n, "nfluid": self.nf, "nbound": self.nb, "out": int(self.nout[None]), "dt_min_hits": self.dt_warn, "cells": self.ncell}
