import json
import os

import numpy as np

from simweb import app as core
from simweb import llm

from . import __version__, prompt, results, rules, spec
from .runner import Runner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
llm.PREFIXES[:] = ["SPHWEB", "SIMWEB"]


def preview_points(sp, limit=150000):
    dx = sp["particle"]["dx"]
    dim = sp.get("dim", 3)
    y0 = dx * (np.round(float(sp.get("plane_y", 0.0)) / dx - 0.5) + 0.5)
    pts = []
    step = 1
    for b in sp["fluid_blocks"]:
        lo, hi = np.array(b["box"]["min"], float), np.array(b["box"]["max"], float)
        n = np.maximum(np.floor((hi - lo) / dx + 1e-9).astype(int), 0)
        if dim == 2:
            if not lo[1] <= y0 <= hi[1]:
                continue
            n[1] = 1
        if n.prod() == 0:
            continue
        step = max(step, int(np.ceil((n.prod() / max(limit // max(len(sp["fluid_blocks"]), 1), 1)) ** (1 / dim))))
        ax = [lo[k] + dx * (0.5 + np.arange(0, n[k], step)) for k in range(3)]
        if dim == 2:
            ax[1] = np.array([y0])
        g = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        pts.append(g)
    if not pts:
        return [], step
    return np.round(np.vstack(pts), 3).tolist(), step


class Plugin:
    name = "sphweb"
    title = "SPH-LLM-web"
    version = __version__
    web_dir = os.path.join(ROOT, "web")
    examples_dir = os.path.join(ROOT, "examples")
    llm_system = prompt.SYSTEM

    def default_spec(self, geom):
        return spec.default_spec(geom)

    def on_geometry(self, old, geom):
        new = spec.default_spec(geom)
        for k in ("fluid", "particle", "gravity", "solver", "device", "dim"):
            new[k] = old[k]
        return new

    def apply_ops(self, sp, ops, geom):
        return spec.apply_ops(sp, ops, geom)

    def warnings(self, sp, geom):
        return spec.warnings_for(sp, geom)

    def rules(self, text, geom, sp):
        return rules.parse(text, geom, sp)

    def llm_context(self, geom, sp, result):
        return prompt.context(geom, sp, result)

    def state(self, S):
        r = RUNNER.status()
        return {"estimate": spec.estimate(S["spec"]), "engine": ENGINE["info"], "run": {k: r[k] for k in ("state", "message", "t", "t_end", "frames", "dir")}}

    def tools(self):
        info = ENGINE["info"]
        return {"taichi": info["taichi"] if info else None, "devices": info["available"] if info else None}

    def examples(self):
        p = os.path.join(self.examples_dir, "presets.json")
        if not os.path.exists(p):
            return []
        return [{"name": k, **v} for k, v in json.load(open(p, encoding="utf-8")).items()]

    def routes(self, app, S, lock, state, fail):
        @app.get("/api/engine")
        def engine(refresh: int = 0):
            from .engine import backend

            ENGINE["info"] = backend.detect(bool(refresh))
            return ENGINE["info"]

        @app.get("/api/preview")
        def preview():
            with lock:
                pts, step = preview_points(S["spec"])
                return {"points": pts, "step": step, "dx": S["spec"]["particle"]["dx"], "estimate": spec.estimate(S["spec"])}

        @app.post("/api/solve")
        def solve():
            from .engine import backend

            with lock:
                sp, geom = S["spec"], S.get("geom")
                warns = [w for w in spec.warnings_for(sp, geom) if w.get("level") == "err"]
            if warns:
                fail(warns[0]["text"], 400)
            dev = sp.get("device", "auto")
            arch = backend.choose(dev) if dev != "cpu" else "cpu"
            if not arch:
                fail(f"계산 장치 {dev} 를 쓸 수 없습니다. 자동 또는 CPU 로 바꿔 주세요", 400)
            try:
                RUNNER.start(sp, geom, arch)
            except RuntimeError as e:
                fail(str(e), 409)
            return RUNNER.status()

        @app.get("/api/run")
        def run_status(since: int = 0):
            st = RUNNER.status()
            if since:
                st["series"] = {k: v[since:] for k, v in st["series"].items()}
                st["since"] = since
            return st

        @app.post("/api/run/{what}")
        def run_control(what: str):
            try:
                RUNNER.control(what)
            except (RuntimeError, ValueError) as e:
                fail(str(e), 409)
            return RUNNER.status()

        @app.get("/api/run/frame/{k}")
        def run_frame(k: int):
            from fastapi import Response

            try:
                return Response(RUNNER.frame_bytes(k), media_type="application/octet-stream")
            except (FileNotFoundError, IndexError) as e:
                fail(str(e), 404)

        @app.get("/api/run/pack")
        def run_pack():
            from fastapi.responses import FileResponse

            st = RUNNER.status()
            if not st["dir"]:
                fail("결과가 없습니다", 404)
            out = results.pack(st["dir"], os.path.join(st["dir"], "rhino_pack.npz"))
            return FileResponse(out, filename=os.path.basename(st["dir"]) + "_rhino.npz")


ENGINE = {"info": None}
RUNNER = Runner(os.environ.get("SPHWEB_RUNS", os.path.join(ROOT, "runs")))
plugin = Plugin()
app = core.create_app(plugin)
