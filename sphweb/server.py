import json
import os

import numpy as np

from simweb import app as core
from simweb import llm

from . import __version__, prompt, rules, spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
llm.PREFIXES[:] = ["SPHWEB", "SIMWEB"]


def preview_points(sp, limit=40000):
    dx = sp["particle"]["dx"]
    pts = []
    for b in sp["fluid_blocks"]:
        lo, hi = np.array(b["box"]["min"]), np.array(b["box"]["max"])
        n = np.maximum(np.floor((hi - lo) / dx + 1e-9).astype(int), 0)
        if n.prod() == 0:
            continue
        step = max(1, int(np.ceil((n.prod() / max(limit // max(len(sp["fluid_blocks"]), 1), 1)) ** (1 / 3))))
        ax = [lo[k] + dx * (0.5 + np.arange(0, n[k], step)) for k in range(3)]
        g = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        pts.append(g)
    if not pts:
        return []
    return np.round(np.vstack(pts), 3).tolist()


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
        for k in ("fluid", "particle", "gravity", "solver", "device"):
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
        return {"estimate": spec.estimate(S["spec"]), "engine": ENGINE["info"], "result": None}

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
                return {"points": preview_points(S["spec"]), "estimate": spec.estimate(S["spec"])}

        @app.post("/api/solve")
        def solve():
            fail("해석 엔진은 2단계에서 연결합니다. 지금은 케이스 설정과 미리보기만 됩니다", 501)


ENGINE = {"info": None}
plugin = Plugin()
app = core.create_app(plugin)
