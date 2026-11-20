import numpy as np


def lattice(lo, hi, dp, origin=(0.0, 0.0, 0.0)):
    o = np.asarray(origin, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    i0 = np.ceil((lo - o) / dp - 1e-6).astype(int)
    i1 = np.floor((hi - o) / dp + 1e-6).astype(int)
    ax = [o[k] + dp * np.arange(i0[k], i1[k] + 1) for k in range(3)]
    g = np.meshgrid(*ax, indexing="ij")
    return np.stack([a.ravel() for a in g], 1)


def tank(length, width, height, depth, dp, layers=1, open_top=True, normals=False):
    nx, ny, nz = (int(round(v / dp)) for v in (length, width, height))
    i = np.arange(-(layers - 1), nx + layers)
    j = np.arange(-(layers - 1), ny + layers)
    k = np.arange(-(layers - 1), nz + 1 + (0 if open_top else layers - 1))
    I, J, K = np.meshgrid(i, j, k, indexing="ij")
    wall = (I <= 0) | (I >= nx) | (J <= 0) | (J >= ny) | (K <= 0)
    if not open_top:
        wall |= K >= nz
    idx = np.stack([I[wall], J[wall], K[wall]], 1).astype(float)
    bound = idx * dp
    nd = int(np.floor(depth / dp + 1e-6))
    fi, fj, fk = np.meshgrid(np.arange(1, nx), np.arange(1, ny), np.arange(1, nd + 1), indexing="ij")
    fluid = np.stack([fi.ravel(), fj.ravel(), fk.ravel()], 1) * dp
    if normals:
        return fluid.astype(float), bound.astype(float), box_normals(idx, (nx, ny, nz), dp, open_top)
    return fluid.astype(float), bound.astype(float)


def box_normals(idx, n, dp, open_top=True):
    g = idx.copy()
    for k in range(3):
        lo = idx[:, k] <= 0
        g[lo, k] = 1.0 - idx[lo, k]
        if k < 2 or not open_top:
            hi = idx[:, k] >= n[k]
            g[hi, k] = 2 * n[k] - 1.0 - idx[hi, k]
    return (g - idx) * dp


def dam_break(length, width, height, col_l, col_h, dp, layers=1, normals=False):
    r = tank(length, width, height, 0.0, dp, layers, normals=normals)
    bound = r[1]
    nx, ny = int(round(col_l / dp)), int(round(width / dp))
    nh = int(np.floor(col_h / dp + 1e-6))
    fi, fj, fk = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny), np.arange(1, nh + 1), indexing="ij")
    fluid = np.stack([fi.ravel(), fj.ravel(), fk.ravel()], 1) * dp
    if normals:
        return fluid.astype(float), bound, r[2]
    return fluid.astype(float), bound
