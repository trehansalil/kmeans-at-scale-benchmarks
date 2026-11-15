"""E1 analogue fidelity (hardware independent): per-row distance-evaluation fraction and miss rate of the
two-level search, (a) DURING training (sampled against exact search) and (b) on the FINAL e-means centroids
with freshly built cells, for nprobe in {default, 2x, all}.

    python e1_fidelity.py --dataset sift --n 400000 --k 16384 --seeds 3 --out ../results/e1_sift_k16384.csv \
        --beat ../run/e1.beat --threads 4
"""
import argparse
import sys

from faiss.contrib.e_means import NumpyEmeans

import e2_equal_epochs as e2
from common import Beat, append_row, done_keys, init_rows, load1, load_dataset, set_threads, heldout_mse
from runner import NPROBE_SPECS, resolve_nprobe
from twolevel import final_miss


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--Ts", default="1,3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beat", default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    argv2 = ["--dataset", a.dataset, "--n", str(a.n), "--k", str(a.k), "--seeds", str(a.seeds), "--Ts", a.Ts,
             "--methods", "tl_emeans", "--nprobes", ",".join(NPROBE_SPECS), "--miss", "0.02", "--out", a.out,
             "--threads", str(a.threads), "--tag", "e1"]
    if a.d:
        argv2 += ["--d", str(a.d)]
    if a.beat:
        argv2 += ["--beat", a.beat]
    e2.run(e2.build_parser().parse_args(argv2))

    # (b) final-centroid miss rate on held-out rows
    set_threads(a.threads)
    beat = Beat(a.beat)
    done = done_keys(a.out)
    for seed in range(a.seeds):
        x, xt = load_dataset(a.dataset, a.n, a.n_test, seed=seed, d=a.d)
        c = None
        for spec in NPROBE_SPECS:
            key = (a.dataset, str(a.n), str(a.k), "final_miss", "3", str(seed), spec)
            if key in done:
                continue
            if c is None:
                em = NumpyEmeans(x.shape[1], a.k, nepochs=3, seed=seed, capture_assignments=False)
                c, _ = em.train(x, init_centroids=init_rows(x, a.k, seed))
            npb = resolve_nprobe(spec, a.k, x.shape[1])
            miss, evf, nl, np_ = final_miss(c, xt, nprobe=npb, seed=seed)
            append_row(a.out, dict(dataset=a.dataset, n=a.n, d=x.shape[1], k=a.k, method="final_miss", T=3, seed=seed,
                                   nprobe=spec, nlist=nl, mse_test=heldout_mse(xt, c), evals_frac=evf, miss_frac=miss,
                                   load1=round(load1(), 2), note="nprobe=%d on final 3-epoch e-means centroids" % np_))
            beat()
            print(f"[e1] final_miss seed={seed} nprobe={spec}({np_}) miss={miss:.4f} evals_frac={evf:.3f}", flush=True)
    print("DONE e1", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
