"""Shared helpers for the o-means vs k-means-family experiments (CPU only, faiss-cpu 1.16.0).

Rules baked in: identical data, test split and init rows for every method within a seed; no training
subsampling for any method; test MSE is exact (faiss.knn over all centroids); every result row carries
versions, seed and load1; results are appended to CSV so a killed run resumes.
"""
import csv
import math
import os
import platform
import time

import faiss
import numpy as np

# ---------------------------------------------------------------- data

SIFT_PATH = os.environ.get("SIFT_HDF5", os.path.expanduser("~/o-means-vs-kmeans/data/sift-128-euclidean.hdf5"))

# name -> (n_modes, spread, d) for Gaussian mixtures with unit-variance noise
MIXTURES = {
    "mix_easy": (200, 6.0, 64),
    "mix_hard": (500, 0.8, 64),
    "mix_hard16k": (16384, 0.8, 96),   # scaled-down analogue of the README's 65,536-mode failure case
}


def load_dataset(name, n, n_test=20000, seed=0, d=None):
    """Return (train float32 (n,d), test float32 (n_test,d)). Train and test rows are disjoint."""
    if name == "sift":
        import h5py
        with h5py.File(SIFT_PATH, "r") as f:
            ds = f["train"]
            x = np.ascontiguousarray(ds[: n + n_test], dtype="float32")
        return x[:n], x[n:n + n_test]
    if name == "uniform":
        rng = np.random.RandomState(seed)
        dd = d or 64
        x = rng.random_sample((n + n_test, dd)).astype("float32") * 4.0
        return x[:n], x[n:]
    if name in MIXTURES:
        modes, spread, dd = MIXTURES[name]
        dd = d or dd
        rng = np.random.RandomState(1000 + seed)
        centers = (rng.standard_normal((modes, dd)) * spread).astype("float32")
        lab = rng.randint(0, modes, n + n_test)
        x = centers[lab] + rng.standard_normal((n + n_test, dd)).astype("float32")
        x = np.ascontiguousarray(x, dtype="float32")
        return x[:n], x[n:]
    raise ValueError("unknown dataset " + name)


def init_rows(x, k, seed):
    """The k random rows NumpyEmeans would draw by default, so every method starts identically."""
    return np.ascontiguousarray(x[np.random.RandomState(seed).permutation(len(x))[:k]], dtype="float32")


# ---------------------------------------------------------------- metrics

def heldout_mse(x_test, centroids):
    """Mean squared distance of held-out rows to their exact nearest centroid."""
    d, _ = faiss.knn(np.ascontiguousarray(x_test, dtype="float32"), np.ascontiguousarray(centroids, dtype="float32"), 1)
    return float(d.astype("float64").mean())


def t_interval(values, conf=0.95):
    """(mean, half-width) of a t-interval; half-width is nan for fewer than 2 values."""
    v = np.asarray(values, dtype="float64")
    m = float(v.mean())
    if len(v) < 2:
        return m, float("nan")
    from scipy import stats
    h = float(stats.t.ppf(0.5 + conf / 2, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))
    return m, h


def rel_diff_pct(a, b):
    """100*(a-b)/b."""
    return 100.0 * (a - b) / b


def dead_fraction(x, centroids, thresh=0.1):
    """Share of centroids whose cluster size is below thresh * mean size (training set)."""
    _, lab = faiss.knn(np.ascontiguousarray(x, dtype="float32"), np.ascontiguousarray(centroids, dtype="float32"), 1)
    cnt = np.bincount(lab.ravel(), minlength=len(centroids))
    return float((cnt < thresh * len(x) / len(centroids)).mean())


# ---------------------------------------------------------------- baselines

def lloyd(x, k, niter=25, seed=0, init=None):
    """faiss Lloyd, no subsampling. Returns (centroids, wall_s, cpu_s)."""
    km = faiss.Kmeans(x.shape[1], k, niter=niter, nredo=1, seed=1234 + seed, verbose=False,
                      max_points_per_centroid=10 ** 9, min_points_per_centroid=1)
    t0, c0 = time.perf_counter(), time.process_time()
    km.train(x, init_centroids=init)
    return km.centroids.copy(), time.perf_counter() - t0, time.process_time() - c0


def kmeanspp_init(x, k, seed=0):
    """k-means++ seeding via scikit-learn. Returns (centers float32, seconds)."""
    from sklearn.cluster import kmeans_plusplus
    t0 = time.perf_counter()
    c, _ = kmeans_plusplus(x, k, random_state=seed)
    return np.ascontiguousarray(c, dtype="float32"), time.perf_counter() - t0


def sculley(x, k, epochs, seed=0, init=None, batch=16384):
    """Sculley-style mini-batch k-means: exact assignment, per-centre rate 1/count, no reassignment.
    Returns (centroids, wall_s, cpu_s, evals) where evals counts point-centroid distances."""
    n, d = x.shape
    c = (init if init is not None else init_rows(x, k, seed)).copy()
    counts = np.zeros(k, dtype="float64")
    rng = np.random.RandomState(seed)
    t0, c0 = time.perf_counter(), time.process_time()
    evals = 0
    steps = int(math.ceil(epochs * n / batch))
    perm = rng.permutation(n)
    pos = 0
    for _ in range(steps):
        if pos + batch > n:
            perm, pos = rng.permutation(n), 0
        xb = x[perm[pos:pos + batch]]
        pos += batch
        _, lab = faiss.knn(xb, c, 1)
        lab = lab.ravel()
        evals += len(xb) * k
        cnt_b = np.bincount(lab, minlength=k).astype("float64")
        sums = np.zeros((k, d), dtype="float64")
        np.add.at(sums, lab, xb) if len(xb) < 2000 else _scatter_add(sums, lab, xb)
        touched = cnt_b > 0
        counts[touched] += cnt_b[touched]
        eta = (cnt_b[touched] / counts[touched])[:, None]
        mean_b = sums[touched] / cnt_b[touched][:, None]
        c[touched] = ((1 - eta) * c[touched] + eta * mean_b).astype("float32")
    return c, time.perf_counter() - t0, time.process_time() - c0, evals


def _scatter_add(sums, lab, xb):
    order = np.argsort(lab, kind="stable")
    ls = lab[order]
    bounds = np.flatnonzero(np.diff(ls)) + 1
    starts = np.concatenate([[0], bounds])
    sums[ls[starts]] += np.add.reduceat(xb[order].astype("float64"), starts, axis=0)


# ---------------------------------------------------------------- bookkeeping

def load1():
    try:
        return os.getloadavg()[0]
    except OSError:
        return float("nan")


def env_info(threads):
    import sklearn
    return {"faiss": faiss.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
            "machine": platform.machine(), "platform": platform.platform(), "threads": threads}


class Beat:
    """Touch a heartbeat file after every step so the guard can tell a hung job from a slow one."""

    def __init__(self, path):
        self.path = path

    def __call__(self):
        if self.path:
            with open(self.path, "a"):
                os.utime(self.path, None)


FIELDS = ["dataset", "n", "d", "k", "method", "T", "seed", "nprobe", "nlist", "mse_test", "wall_s", "cpu_s",
          "evals_frac", "miss_frac", "init_s", "load1", "note"]


def row_key(r):
    return (str(r["dataset"]), str(r["n"]), str(r["k"]), str(r["method"]), str(r["T"]), str(r["seed"]), str(r["nprobe"]))


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {row_key(r) for r in csv.DictReader(f)}


def append_row(path, row):
    new = not os.path.exists(path)
    full = {k: row.get(k, "") for k in FIELDS}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(full)


def set_threads(n):
    faiss.omp_set_num_threads(int(n))
    os.environ["OMP_NUM_THREADS"] = str(int(n))
