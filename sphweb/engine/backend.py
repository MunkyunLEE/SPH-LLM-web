import json
import os
import subprocess
import sys
import time

ARCHS = ("cuda", "metal", "vulkan", "cpu")
_CACHE = {}

PROBE = r"""
import json, sys, time, contextlib, io
arch = sys.argv[1]
out = {"arch": arch}
try:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        import taichi as ti
        ti.init(arch=getattr(ti, arch), offline_cache=False, log_level=ti.ERROR)
    out["taichi"] = ti.__version__ if isinstance(ti.__version__, str) else ".".join(map(str, ti.__version__))
    got = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    out["actual"] = got
    out["ok"] = got == arch or (arch == "cpu" and got in ("x64", "arm64"))
    if out["ok"] and len(sys.argv) > 2:
        n = int(sys.argv[2])
        x = ti.field(ti.f32, shape=n)
        y = ti.field(ti.f32, shape=n)

        @ti.kernel
        def fill():
            for i in x:
                x[i] = i * 1e-6
                y[i] = 0.0

        @ti.kernel
        def work(r: ti.i32):
            for i in x:
                s = 0.0
                for k in range(-r, r + 1):
                    j = (i + k * 97) % n
                    d = x[i] - x[j]
                    s += ti.exp(-d * d)
                y[i] = s

        fill()
        work(4)
        ti.sync()
        t = time.perf_counter()
        for _ in range(5):
            work(16)
        ti.sync()
        dt = (time.perf_counter() - t) / 5
        out["bench"] = {"n": n, "pairs_per_s": n * 33 / dt, "ms": dt * 1000}
except Exception as e:
    out["ok"] = False
    out["error"] = str(e)[:300]
print("@@" + json.dumps(out))
"""


def probe(arch, bench=0, timeout=120):
    try:
        p = subprocess.run([sys.executable, "-c", PROBE, arch] + ([str(bench)] if bench else []), capture_output=True, text=True, timeout=timeout)
        line = [l for l in p.stdout.splitlines() if l.startswith("@@")]
        if not line:
            return {"arch": arch, "ok": False, "error": (p.stderr or p.stdout)[-300:]}
        return json.loads(line[-1][2:])
    except subprocess.TimeoutExpired:
        return {"arch": arch, "ok": False, "error": "시간 초과"}


def detect(refresh=False):
    if _CACHE and not refresh:
        return _CACHE["info"]
    try:
        import importlib.util

        has = importlib.util.find_spec("taichi") is not None
    except Exception:
        has = False
    t0 = time.time()
    if not has:
        info = {"taichi": None, "available": [], "best": None, "results": [], "note": "taichi 가 설치되어 있지 않습니다 (pip install taichi)"}
    else:
        order = [a for a in ARCHS if a != "metal" or sys.platform == "darwin"]
        res = [probe(a) for a in order]
        ok = [r["arch"] for r in res if r.get("ok")]
        info = {"taichi": next((r.get("taichi") for r in res if r.get("taichi")), None), "available": ok, "best": ok[0] if ok else None, "results": res}
    info["seconds"] = round(time.time() - t0, 1)
    info["platform"] = sys.platform
    _CACHE["info"] = info
    return info


def choose(device):
    info = detect()
    if device == "auto":
        return info["best"]
    return device if device in info["available"] else None


def main():
    import argparse

    ap = argparse.ArgumentParser(prog="python -m sphweb.engine.backend")
    ap.add_argument("--bench", type=int, default=2_000_000)
    a = ap.parse_args()
    info = detect()
    print(f"taichi {info['taichi']} · platform {info['platform']}")
    for r in info["results"]:
        print(f"  {r['arch']:7s} {'사용 가능' if r.get('ok') else '없음':6s} {r.get('error', '')[:120]}")
    for arch in info["available"]:
        r = probe(arch, a.bench, timeout=600)
        b = r.get("bench")
        if b:
            print(f"  [bench] {arch:7s} {b['pairs_per_s'] / 1e9:8.3f} G쌍/s  ({b['ms']:.1f} ms, n={b['n']:,})")
    print(f"선택 (auto): {info['best']}")


if __name__ == "__main__":
    main()
