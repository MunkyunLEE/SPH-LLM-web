import json
import os

import numpy as np

FORMAT = "sphweb-result"
VERSION = 1
FIELDS = {"positions": (3, np.float32), "velocity": (3, np.float32), "pressure": (1, np.float32), "density": (1, np.float32), "active": (1, np.bool_)}


class Writer:
    def __init__(self, path, capacity, meta=None):
        self.path = path
        self.capacity = int(capacity)
        os.makedirs(os.path.join(path, "frames"), exist_ok=True)
        self.meta = {"format": FORMAT, "version": VERSION, "units": "mm · tonne · s · N · MPa", "capacity": self.capacity, "frames": [], **(meta or {})}
        self._save_meta()

    def _save_meta(self):
        tmp = os.path.join(self.path, "meta.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(self.path, "meta.json"))

    def add(self, t, step, **arrays):
        k = len(self.meta["frames"])
        data = {}
        for name, (w, dt) in FIELDS.items():
            a = np.asarray(arrays[name], dtype=dt)
            shape = (self.capacity, w) if w > 1 else (self.capacity,)
            if a.shape != shape:
                raise ValueError(f"{name} 배열 크기 {a.shape} 가 {shape} 와 다릅니다")
            data[name] = a
        for name, a in arrays.items():
            if name not in FIELDS:
                data[name] = np.asarray(a)
        fn = f"{k:06d}.npz"
        np.savez_compressed(os.path.join(self.path, "frames", fn), **data)
        self.meta["frames"].append({"i": k, "t": float(t), "step": int(step), "file": fn, "active": int(data["active"].sum())})
        self._save_meta()
        return k

    def set_walls(self, walls, groups=None):
        np.savez_compressed(os.path.join(self.path, "walls.npz"), positions=np.asarray(walls, np.float32), group=np.asarray(groups if groups is not None else np.zeros(len(walls)), np.int32))


class Reader:
    def __init__(self, path):
        self.path = path
        self.meta = json.load(open(os.path.join(path, "meta.json"), encoding="utf-8"))
        if self.meta.get("format") != FORMAT:
            raise ValueError("sphweb 결과 폴더가 아닙니다")

    def __len__(self):
        return len(self.meta["frames"])

    def times(self):
        return np.array([f["t"] for f in self.meta["frames"]])

    def frame(self, i):
        f = self.meta["frames"][i]
        with np.load(os.path.join(self.path, "frames", f["file"])) as d:
            return {k: d[k] for k in d.files} | {"t": f["t"], "step": f["step"]}

    def walls(self):
        p = os.path.join(self.path, "walls.npz")
        if not os.path.exists(p):
            return np.zeros((0, 3), np.float32)
        with np.load(p) as d:
            return d["positions"]

    def trajectory(self, ids):
        ids = np.asarray(ids, int)
        out = np.full((len(self), len(ids), 3), np.nan, np.float32)
        for k in range(len(self)):
            f = self.frame(k)
            ok = f["active"][ids]
            out[k][ok] = f["positions"][ids][ok]
        return out


def pack(path, out):
    r = Reader(path)
    n = len(r)
    cap = r.meta["capacity"]
    P = np.zeros((n, cap, 3), np.float32)
    V = np.zeros((n, cap, 3), np.float32)
    pr = np.zeros((n, cap), np.float32)
    rho = np.zeros((n, cap), np.float32)
    act = np.zeros((n, cap), bool)
    for k in range(n):
        f = r.frame(k)
        P[k], V[k], pr[k], rho[k], act[k] = f["positions"], f["velocity"], f["pressure"], f["density"], f["active"]
    np.savez_compressed(out, positions=P, velocity=V, pressure=pr, density=rho, active=act, time=r.times().astype(np.float32), walls=r.walls(), meta=json.dumps(r.meta, ensure_ascii=False))
    return out
