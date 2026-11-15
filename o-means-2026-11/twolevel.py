"""CPU ANALOGUE of o-means' two-level nearest-centroid assignment, inside e-means' NumPy trainer.

This is NOT o-means. o-means needs CUDA + Triton and cannot run on a CPU. What is reproduced here is
only the assignment idea and the cell-rebuild cadence described in the contrib README:
  - nlist cells over the k centroids, built by k-means on the centroids themselves;
  - each row scores the nlist cell centres, keeps its nprobe nearest cells, scans only those centroids;
  - cells rebuilt: first pass, re-sort into the existing cells every 5 batches (0 Lloyd iterations);
    afterwards every half pass with 2 Lloyd iterations; the first build uses 4 iterations.
NOT reproduced: split child/grandchild links, FP16 kernels, recency-weighted closing mean, the GPU
three-kernel layout. Everything else (batch law, EMA, cosine step, revival, closing mean) is e-means'.
"""
import time

import faiss
import numpy as np
from faiss.contrib.e_means import NumpyEmeans

FIRST_BUILD_ITERS = 4
REBUILD_ITERS = 2
FIRST_PASS_EVERY = 5     # e-means checks for dead centroids every 5 updates


def default_nlist(k):
    n = -(-k // 128)                      # ceil(k / 128)
    n = -(-n // 16) * 16                  # up to a multiple of 16
    return max(16, n)


def default_nprobe(k, d):
    if k > 16384:
        return 24
    return 12 if d >= 512 else 16


class CellIndex:
    """Two-level search over a (k, d) centroid table. Cell membership is frozen between rebuilds,
    but the scanned centroid VALUES are always the current ones, as in training."""

    def __init__(self, d, k, nlist=None, nprobe=None, seed=0):
        self.d, self.k = d, k
        self.nlist = nlist or default_nlist(k)
        self.nlist = min(self.nlist, k)
        if nprobe in ("all", None):
            self.nprobe = self.nlist if nprobe == "all" else min(default_nprobe(k, d), self.nlist)
        else:
            self.nprobe = min(int(nprobe), self.nlist)
        self.seed = seed
        self.centres = None
        self.cell_of = None
        self.order = None
        self.starts = None
        self.sizes = None
        self.t_build = 0.0

    def build(self, centroids, iters):
        """Rebuild cell centres by k-means on the centroids (warm start from previous centres), then re-sort."""
        t0 = time.perf_counter()
        km = faiss.Kmeans(self.d, self.nlist, niter=max(iters, 1) if iters > 0 else 1, nredo=1, seed=self.seed + 17,
                          verbose=False, max_points_per_centroid=10 ** 9, min_points_per_centroid=1)
        if iters > 0:
            km.train(np.ascontiguousarray(centroids, dtype="float32"), init_centroids=self.centres)
            self.centres = km.centroids.copy()
        elif self.centres is None:
            raise RuntimeError("first build needs iters > 0")
        self.resort(centroids, _t0=t0)

    def resort(self, centroids, _t0=None):
        """Assign the centroids to the existing cell centres (no Lloyd iterations)."""
        t0 = _t0 if _t0 is not None else time.perf_counter()
        _, cell = faiss.knn(np.ascontiguousarray(centroids, dtype="float32"), self.centres, 1)
        self.cell_of = cell.ravel()
        self.order = np.argsort(self.cell_of, kind="stable")
        self.sizes = np.bincount(self.cell_of, minlength=self.nlist)
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)])
        self.t_build += time.perf_counter() - t0

    def search(self, x, centroids):
        """Return (squared distances, labels, evals) with labels in the caller's centroid indexing."""
        m = len(x)
        probe = self.nprobe
        _, cells = faiss.knn(x, self.centres, probe)            # (m, nprobe) nearest cells
        cs = np.ascontiguousarray(centroids[self.order])        # table sorted by cell, current values
        flat_cell = cells.ravel()
        flat_row = np.repeat(np.arange(m), probe)
        srt = np.argsort(flat_cell, kind="stable")
        fc, fr = flat_cell[srt], flat_row[srt]
        bounds = np.searchsorted(fc, np.arange(self.nlist + 1))
        best_d = np.full(m, np.inf, dtype="float32")
        best_l = np.zeros(m, dtype="int64")
        for c in range(self.nlist):
            s, e = self.starts[c], self.starts[c + 1]
            if e == s:
                continue
            rows = fr[bounds[c]:bounds[c + 1]]
            if len(rows) == 0:
                continue
            dd, ii = faiss.knn(np.ascontiguousarray(x[rows]), cs[s:e], 1)
            dd, ii = dd.ravel(), ii.ravel()
            better = dd < best_d[rows]
            if better.any():
                r = rows[better]
                best_d[r] = dd[better]
                best_l[r] = self.order[s + ii[better]]
        evals = int(m * self.nlist + self.sizes[cells].sum())
        return best_d, best_l, evals


class TwoLevelEmeans(NumpyEmeans):
    """e-means with the two-level assignment above. Counters: evals, sampled, missed, t_assign, t_measure."""

    def __init__(self, *args, nlist=None, nprobe=None, miss_frac=0.0, **kw):
        super().__init__(*args, **kw)
        self._cell_args = (nlist, nprobe)
        self.cells = None
        self.miss_frac = float(miss_frac)
        self.evals = 0
        self.rows_assigned = 0
        self.sampled = 0
        self.missed = 0
        self.t_assign = 0.0
        self.t_measure = 0.0
        self._last_cell_step = -1
        self._mrng = np.random.RandomState(self.seed + 99)

    def _training_call(self, centroids):
        # during training NumpyEmeans passes its own mutable table; assign() after finalize passes a copy
        return centroids is self._centroids

    def _maybe_rebuild(self, centroids):
        batch = self.recipe["batch_size"]
        step = self._step
        if self.cells is None:
            self.cells = CellIndex(self.d, self.k, *self._cell_args, seed=self.seed)
            self.cells.build(centroids, FIRST_BUILD_ITERS)
            self._last_cell_step = step
            return
        if step == self._last_cell_step:
            return
        since = step - self._last_cell_step
        steps_per_pass = max(1, -(-self._n // batch))
        first_pass = step * batch < self._n
        if first_pass:
            if since >= FIRST_PASS_EVERY:
                self.cells.resort(centroids)
                self._last_cell_step = step
        elif since >= max(1, steps_per_pass // 2):
            self.cells.build(centroids, REBUILD_ITERS)
            self._last_cell_step = step

    def _assign_rows(self, x, centroids):
        if not self._training_call(centroids):
            return super()._assign_rows(x, centroids)
        x = np.ascontiguousarray(x, dtype="float32")
        self._maybe_rebuild(centroids)
        t0 = time.perf_counter()
        d, lab, ev = self.cells.search(x, centroids)
        self.t_assign += time.perf_counter() - t0
        self.evals += ev
        self.rows_assigned += len(x)
        if self.miss_frac > 0:
            t1 = time.perf_counter()
            m = max(1, int(self.miss_frac * len(x)))
            idx = self._mrng.choice(len(x), m, replace=False)
            _, ex = faiss.knn(x[idx], centroids, 1)
            self.sampled += m
            self.missed += int((ex.ravel() != lab[idx]).sum())
            self.t_measure += time.perf_counter() - t1
        return d, lab

    def stats(self):
        n = max(self.rows_assigned, 1)
        return {
            "evals_frac": self.evals / (n * self.k),
            "miss_frac": (self.missed / self.sampled) if self.sampled else float("nan"),
            "t_cells": self.cells.t_build if self.cells else 0.0,
            "nlist": self.cells.nlist if self.cells else 0,
            "nprobe": self.cells.nprobe if self.cells else 0,
        }


def final_miss(centroids, x_test, nlist=None, nprobe=None, seed=0):
    """Miss rate and evaluation fraction of the two-level search on FINAL centroids (fresh 4-iteration cells)."""
    d = centroids.shape[1]
    ci = CellIndex(d, len(centroids), nlist, nprobe, seed=seed)
    ci.build(centroids, FIRST_BUILD_ITERS)
    dd, lab, ev = ci.search(np.ascontiguousarray(x_test, dtype="float32"), np.ascontiguousarray(centroids, dtype="float32"))
    _, ex = faiss.knn(np.ascontiguousarray(x_test, dtype="float32"), np.ascontiguousarray(centroids, dtype="float32"), 1)
    return float((ex.ravel() != lab).mean()), ev / (len(x_test) * len(centroids)), ci.nlist, ci.nprobe
