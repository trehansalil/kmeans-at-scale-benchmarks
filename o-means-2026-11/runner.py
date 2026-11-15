"""One (method, T, seed) run -> one CSV row. Shared by e0..e3 so every method is timed the same way."""
import time

import numpy as np
from faiss.contrib.e_means import NumpyEmeans

from common import (init_rows, kmeanspp_init, lloyd, load1, sculley, heldout_mse)
from twolevel import TwoLevelEmeans, default_nlist, default_nprobe, final_miss

NPROBE_SPECS = ("def", "2x", "all")


def resolve_nprobe(spec, k, d):
    nl = min(default_nlist(k), k)
    base = min(default_nprobe(k, d), nl)
    return {"def": base, "2x": min(2 * base, nl), "all": nl}[spec]


def run_one(method, x, xt, k, T, seed, nprobe="def", miss_frac=0.0):
    """Return a row dict (without dataset/n bookkeeping fields except d,k,method,T,seed,nprobe)."""
    n, d = x.shape
    init = init_rows(x, k, seed)
    row = {"d": d, "k": k, "method": method, "T": T, "seed": seed, "nprobe": nprobe if method == "tl_emeans" else "",
           "init_s": ""}
    if method == "lloyd25":
        c, wall, cpu = lloyd(x, k, 25, seed, init)
        row.update(wall_s=wall, cpu_s=cpu)
    elif method == "lloyd25_kpp":
        ci, init_s = kmeanspp_init(x, k, seed)
        c, wall, cpu = lloyd(x, k, 25, seed, ci)
        row.update(wall_s=wall + init_s, cpu_s=cpu, init_s=init_s)
    elif method == "minibatch":
        c, wall, cpu, ev = sculley(x, k, T, seed, init)
        row.update(wall_s=wall, cpu_s=cpu, evals_frac=ev / (n * k * T))
    elif method == "emeans":
        em = NumpyEmeans(d, k, nepochs=T, seed=seed, capture_assignments=False)
        t0, c0 = time.perf_counter(), time.process_time()
        c, _ = em.train(x, init_centroids=init)
        row.update(wall_s=time.perf_counter() - t0, cpu_s=time.process_time() - c0, evals_frac=1.0,
                   note="dead_splits=%d" % em.n_splits)
    elif method == "tl_emeans":
        npb = resolve_nprobe(nprobe, k, d)
        em = TwoLevelEmeans(d, k, nepochs=T, seed=seed, capture_assignments=False, nprobe=npb, miss_frac=miss_frac)
        t0, c0 = time.perf_counter(), time.process_time()
        c, _ = em.train(x, init_centroids=init)
        wall = time.perf_counter() - t0 - em.t_measure      # time spent measuring misses is not training time
        cpu = time.process_time() - c0 - em.t_measure
        st = em.stats()
        row.update(wall_s=wall, cpu_s=cpu, evals_frac=st["evals_frac"], miss_frac=st["miss_frac"], nlist=st["nlist"],
                   nprobe=nprobe, note="nprobe=%d nlist=%d t_cells=%.2fs t_assign=%.2fs splits=%d" % (
                       st["nprobe"], st["nlist"], st["t_cells"], em.t_assign, em.n_splits))
    else:
        raise ValueError(method)
    row["mse_test"] = heldout_mse(xt, c)
    row["load1"] = round(load1(), 2)
    return row
