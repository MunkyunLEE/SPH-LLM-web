import copy
import json
import math
import os
import queue
import struct
import threading
import time
import traceback

import numpy as np

from . import casebuild, results

GRID_MAX = 60000
SHOW_MAX = 400000


def grid_for(lo, hi, dx):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    size = np.maximum(hi - lo, dx)
    h = max(dx, (np.prod(size) / GRID_MAX) ** (1 / 3))
    n = np.maximum(np.floor(size / h).astype(int), 2)
    origin = lo + 0.5 * (size - (n - 1) * h)
    g = np.meshgrid(*[origin[k] + h * np.arange(n[k]) for k in range(3)], indexing="ij")
    pts = np.stack([a.transpose(2, 1, 0).ravel() for a in g], 1)
    return {"n": n.tolist(), "origin": origin.tolist(), "h": float(h)}, pts


class Runner:
    def __init__(self, root):
        self.root = root
        self.lock = threading.Lock()
        self.cmd = queue.Queue()
        self.jobs = queue.Queue()
        self.thread = None
        self.arch = None
        self.st = self._idle()

    def _idle(self):
        return {"state": "idle", "message": "", "t": 0.0, "step": 0, "t_end": 0.0, "dt": 0.0, "wall": 0.0, "n": 0, "nbound": 0, "frames": 0, "series": {}, "warnings": [], "dir": None, "device": None, "probes": [], "groups": []}

    def status(self):
        with self.lock:
            return copy.deepcopy(self.st)

    def _set(self, **kw):
        with self.lock:
            self.st.update(kw)

    def start(self, spec, geom, device):
        with self.lock:
            if self.st["state"] in ("building", "running", "paused"):
                raise RuntimeError("이미 해석이 진행 중입니다. 먼저 정지해 주세요")
            self.st = self._idle()
            self.st["state"] = "building"
            self.st["message"] = "입자 배치 중…"
        while not self.cmd.empty():
            self.cmd.get_nowait()
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
        self.jobs.put((copy.deepcopy(spec), geom, device))

    def control(self, what):
        if what not in ("pause", "resume", "step", "stop"):
            raise ValueError(what)
        with self.lock:
            s = self.st["state"]
        if s not in ("running", "paused", "building"):
            raise RuntimeError("진행 중인 해석이 없습니다")
        self.cmd.put(what)

    def _worker(self):
        while True:
            job = self.jobs.get()
            try:
                self._run(*job)
            except Exception as e:
                traceback.print_exc()
                self._set(state="error", message=f"해석 오류: {e}")

    def _run(self, spec, geom, device):
        from .engine import wcsph
        from .engine.consts import Params

        if self.arch is None:
            self.arch = wcsph.init(device or "cpu")
        warn = []
        if device and device != self.arch:
            warn.append(f"계산 장치는 서버를 처음 실행할 때 정해집니다. 지금은 {self.arch} 로 계산합니다 (바꾸려면 서버를 다시 시작)")
        sol = spec["solver"]
        mdbc = sol.get("boundary", "mdbc") == "mdbc"
        case = casebuild.build(spec, geom, layers=4 if mdbc else 2)
        warn += case["warnings"]
        fluid, bound = case["fluid"], case["bound"]
        if not len(fluid):
            self._set(state="error", message="유체 입자가 없어 해석을 시작하지 못했습니다", warnings=warn)
            return
        dx = case["dx"]
        lo, hi = case["domain"]
        zs = float(fluid[:, 2].max())
        zb = float(min(fluid[:, 2].min(), bound[:, 2].min() if len(bound) else fluid[:, 2].min()))
        hswl = max(zs - zb + 0.5 * dx, 2 * dx)
        rho0 = float(spec["fluid"]["rho0"])
        lam = sol.get("viscosity") == "laminar"
        visco = float(spec["fluid"]["mu"]) / rho0 if lam else float(sol.get("alpha", 0.02))
        p = Params(dp=dx, hswl=hswl, rho0=rho0, gravity=tuple(spec["gravity"]), cfl=float(sol.get("cfl", 0.2)), visco=visco, visco_type="laminar" if lam else "artificial", ddt=float(sol.get("ddt", 0.1)))
        s = wcsph.Solver(p, fluid, bound, domain=(lo, hi), bound_normals=case["normals"] if mdbc and len(bound) else None)
        s.hydrostatic(zs, bound=False)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = os.path.join(self.root, stamp)
        ginfo, gpts = grid_for(lo, hi, dx)
        probes = [m for m in spec.get("monitors", []) if "point" in m]
        ppts = np.array([m["point"] for m in probes], float).reshape(-1, 3)
        groups = case["group_names"] or (["벽"] if len(bound) else [])
        meta = {"spec": spec, "params": p.to_dict(), "nbound": int(len(bound)), "grid": ginfo, "groups": groups, "probes": probes, "device": self.arch}
        w = results.Writer(out, len(fluid), meta)
        if len(bound):
            w.set_walls(bound, case["group"])
        t_end = float(sol["t_end"])
        every = 1.0 / float(sol["fps"])
        series = {k: [] for k in ("time", "particles.active", "density.error.mean", "density.error.max", "energy.kinetic", "energy.potential", "body.force.x", "body.force.y", "body.force.z")}
        for g in groups:
            for c in "xyz":
                series[f"force.{g}.{c}"] = []
        for m in probes:
            series[f"probe.{m['id']}.speed"] = []
            series[f"probe.{m['id']}.pressure"] = []
        self._set(state="running", message="계산 중", t_end=t_end, n=int(len(fluid)), nbound=int(len(bound)), dir=out, device=self.arch, warnings=warn, probes=probes, groups=groups, series=series, grid=ginfo, params={"h": p.h, "cs0": p.cs0, "dp": dx, "boundary": "mDBC" if mdbc else "DBC", "viscosity": "층류" if lam else f"인공 α={visco:g}"})
        gvec = np.asarray(spec["gravity"], float)
        mass = p.mass
        stops = spec.get("stops", [])

        def emit():
            d = s.snapshot(True)
            act = d["kind"] == 1
            u, pr, occ = s.sample(gpts)
            pu, pp, pw = s.sample(ppts) if len(ppts) else (np.zeros((0, 3)), np.zeros(0), np.zeros(0))
            fw = s.wall_forces(case["group"], len(groups)) if len(bound) else np.zeros((0, 3))
            v2 = (d["vel"][act] ** 2).sum(1)
            rr = np.abs(d["rho_rel"][act] - 1.0)
            vals = {
                "time": s.time,
                "particles.active": int(act.sum()),
                "density.error.mean": float(rr.mean()) if act.any() else 0.0,
                "density.error.max": float(rr.max()) if act.any() else 0.0,
                "energy.kinetic": float(0.5 * mass * v2.sum()),
                "energy.potential": float(-mass * (d["pos"][act] @ gvec).sum()),
                "body.force.x": float(fw[:, 0].sum()) if len(fw) else 0.0,
                "body.force.y": float(fw[:, 1].sum()) if len(fw) else 0.0,
                "body.force.z": float(fw[:, 2].sum()) if len(fw) else 0.0,
            }
            for gi, g in enumerate(groups):
                for c, a in zip("xyz", range(3)):
                    vals[f"force.{g}.{c}"] = float(fw[gi, a])
            for k, m in enumerate(probes):
                ok = pw[k] > 0.3
                vals[f"probe.{m['id']}.speed"] = float(np.linalg.norm(pu[k])) if ok else 0.0
                vals[f"probe.{m['id']}.pressure"] = float(pp[k]) if ok else 0.0
            w.add(s.time, s.nstep, positions=d["pos"], velocity=d["vel"], pressure=d["press"], density=d["rho"], active=act, grid_u=u.astype(np.float16), grid_p=pr.astype(np.float32), grid_occ=(occ > 0.3).astype(np.uint8), forces=fw.astype(np.float32))
            info = s.info()
            with self.lock:
                for k, v in vals.items():
                    self.st["series"].setdefault(k, []).append(v)
                self.st.update(t=s.time, step=s.nstep, dt=info["dt"], wall=info["wall"], frames=len(w.meta["frames"]), out=info["out"])
            return vals

        def stop_hit(vals):
            for c in stops:
                v = vals.get(c["quantity"])
                if v is None:
                    continue
                ok = {">": v > c["value"], "<": v < c["value"], ">=": v >= c["value"], "<=": v <= c["value"]}[c["cmp"]]
                if ok:
                    return c
            return None

        emit()
        nxt = every
        paused = False
        while s.time < t_end - 1e-12:
            try:
                while True:
                    c = self.cmd.get_nowait() if not paused else self.cmd.get(timeout=0.2)
                    if c == "stop":
                        emit()
                        self._set(state="stopped", message=f"사용자가 정지했습니다 (t = {s.time:.4f} s)")
                        return
                    if c == "pause":
                        paused = True
                        self._set(state="paused", message=f"일시정지 (t = {s.time:.4f} s)")
                    elif c == "resume":
                        paused = False
                        self._set(state="running", message="계산 중")
                    elif c == "step" and paused:
                        s.step()
                        emit()
                        self._set(message=f"한 스텝 진행 (t = {s.time:.6f} s, dt = {s.dt_last:.3e} s)")
                    if not paused:
                        break
            except queue.Empty:
                if paused:
                    continue
            s.step()
            if not math.isfinite(s.time) or s.info()["out"] > 0.5 * len(fluid):
                emit()
                self._set(state="error", message="입자 절반 이상이 밀도 범위를 벗어나 해석을 멈췄습니다 (발산)")
                return
            if s.time >= nxt - 1e-12:
                nxt += every
                vals = emit()
                hit = stop_hit(vals)
                if hit:
                    paused = True
                    self._set(state="paused", message=f"정지 조건 {hit['id']} ({hit['quantity']} {hit['cmp']} {hit['value']:g}) 에 걸려 일시정지했습니다")
            else:
                with self.lock:
                    self.st.update(t=s.time, step=s.nstep, dt=s.dt_last, wall=s.wall)
        if abs(s.time - (nxt - every)) > 1e-9:
            emit()
        self._set(state="done", message=f"해석 완료 ({s.nstep:,} 스텝, {s.wall:.1f} s)")

    def frame_bytes(self, k, field_max=SHOW_MAX):
        with self.lock:
            out = self.st["dir"]
            grid = self.st.get("grid")
        if not out:
            raise FileNotFoundError("결과가 없습니다")
        r = results.Reader(out)
        if k < 0:
            k = len(r) + k
        if not 0 <= k < len(r):
            raise IndexError("프레임 번호가 범위를 벗어났습니다")
        f = r.frame(k)
        act = f["active"].astype(bool)
        ids = np.nonzero(act)[0]
        if len(ids) > field_max:
            ids = ids[:: int(math.ceil(len(ids) / field_max))]
        arrays = [
            ("pos", f["positions"][ids].astype(np.float32).ravel()),
            ("vel", f["velocity"][ids].astype(np.float32).ravel()),
            ("p", f["pressure"][ids].astype(np.float32)),
            ("gu", f["grid_u"].astype(np.float32).ravel()),
            ("gp", f["grid_p"].astype(np.float32)),
            ("gocc", f["grid_occ"].astype(np.uint8)),
        ]
        head = {"k": k, "t": f["t"], "step": int(f["step"]), "n": int(len(ids)), "active": int(act.sum()), "grid": grid, "forces": f.get("forces", np.zeros((0, 3))).tolist(), "arrays": [[n, int(a.size), "u8" if a.dtype == np.uint8 else "f32"] for n, a in arrays]}
        hb = json.dumps(head).encode()
        parts = [struct.pack("<I", len(hb)), hb]
        off = 4 + len(hb)
        for _, a in arrays:
            pad = (-off) % 4
            parts.append(b"\0" * pad)
            off += pad
            b = a.tobytes()
            parts.append(b)
            off += len(b)
        return b"".join(parts)
