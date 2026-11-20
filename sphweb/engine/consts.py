import math
from dataclasses import dataclass, field, asdict

import numpy as np

AWEN3 = 21.0 / (16.0 * math.pi)
AWEN2 = 7.0 / (4.0 * math.pi)


@dataclass
class Params:
    dp: float
    hswl: float
    rho0: float = 0.998e-9
    gravity: tuple = (0.0, 0.0, -9810.0)
    coefh: float = 1.0
    coefsound: float = 20.0
    gamma: float = 7.0
    cfl: float = 0.2
    visco: float = 0.1
    visco_type: str = "artificial"
    visco_bound: float = 1.0
    ddt: float = 0.1
    rho_min: float = 0.7
    rho_max: float = 1.3
    coef_dtmin: float = 0.05
    dim: int = 3
    scheme: str = "symplectic"
    verlet_steps: int = 40
    h: float = field(init=False)
    support: float = field(init=False)
    cs0: float = field(init=False)
    b: float = field(init=False)
    eta2: float = field(init=False)
    ddtkh: float = field(init=False)
    ddtgz: float = field(init=False)
    vol: float = field(init=False)
    mass: float = field(init=False)
    awen: float = field(init=False)
    bwen: float = field(init=False)
    dt_ini: float = field(init=False)
    dt_min: float = field(init=False)

    def __post_init__(self):
        g = float(np.linalg.norm(self.gravity))
        if self.dim not in (2, 3):
            raise ValueError("dim 은 2 또는 3 입니다")
        if self.scheme not in ("symplectic", "verlet"):
            raise ValueError("scheme 은 symplectic 또는 verlet 입니다")
        self.h = self.coefh * math.sqrt(self.dim * self.dp * self.dp)
        self.support = 2.0 * self.h
        cs = self.coefsound * math.sqrt(g * self.hswl)
        self.cs0 = max(cs, 10.0 * math.sqrt(g * self.hswl))
        self.b = self.cs0 * self.cs0 / self.gamma
        self.eta2 = (0.1 * self.h) ** 2
        self.ddtkh = self.support * self.ddt
        self.ddtgz = abs(self.gravity[2]) / self.b
        self.vol = self.dp ** self.dim
        self.mass = self.rho0 * self.vol
        self.awen = AWEN3 / self.h ** 3 if self.dim == 3 else AWEN2 / self.h ** 2
        self.bwen = -5.0 * self.awen / self.h ** 2
        self.dt_ini = self.h / self.cs0
        self.dt_min = self.coef_dtmin * self.h / self.cs0

    def to_dict(self):
        return asdict(self)


def wendland(r, h, dim=3):
    q = np.asarray(r, dtype=float) / h
    a = AWEN3 / h ** 3 if dim == 3 else AWEN2 / h ** 2
    w = a * (2 * q + 1) * np.clip(1 - q / 2, 0, None) ** 4
    return np.where(q < 2, w, 0.0)


def wendland_fac(r, h, dim=3):
    q = np.asarray(r, dtype=float) / h
    a = AWEN3 / h ** 3 if dim == 3 else AWEN2 / h ** 2
    f = -5.0 * a / h ** 2 * np.clip(1 - q / 2, 0, None) ** 3
    return np.where(q < 2, f, 0.0)


def pressure(rho_rel, p):
    return p.rho0 * p.b * (np.asarray(rho_rel) ** p.gamma - 1.0)
