"""E2: error (and time, evals, miss rate) at equal epochs for Lloyd@25, Sculley mini-batch, e-means dense,
and the two-level CPU analogue. Also the general driver reused by e0 (sanity), e1 (fidelity), e3 (wall-clock).

    python e2_equal_epochs.py --dataset sift --n 400000 --k 16384 --seeds 3 --Ts 1,2,3,5 \
        --methods lloyd25,minibatch,emeans,tl_emeans --nprobes def,2x,all --miss 0.02 \
        --out ../results/e2_sift_k16384.csv --beat ../run/e2.beat --threads 4
Rows are appended one by one; rerunning skips finished rows. Prints 'DONE e2' on success.
"""
import argparse
import sys
import time

from common import Beat, append_row, done_keys, env_info, load_dataset, set_threads
from runner import run_one


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--Ts", default="1,2,3,5")
    ap.add_argument("--methods", default="lloyd25,minibatch,emeans,tl_emeans")
    ap.add_argument("--nprobes", default="def,2x,all")
    ap.add_argument("--miss", type=float, default=0.0, help="fraction of rows per batch checked against exact search")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beat", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="e2")
    return ap


def run(a):
    set_threads(a.threads)
    beat = Beat(a.beat)
    beat()
    done = done_keys(a.out)
    Ts = [float(t) if "." in t else int(t) for t in a.Ts.split(",")]
    methods = a.methods.split(",")
    nprobes = a.nprobes.split(",")
    info = env_info(a.threads)
    print("env", info, flush=True)
    t_start = time.time()
    for seed in range(a.seeds):
        x, xt = load_dataset(a.dataset, a.n, a.n_test, seed=seed, d=a.d)
        print(f"[{a.tag}] seed={seed} data {x.shape} test {xt.shape}", flush=True)
        beat()
        plan = []
        for m in methods:
            if m in ("lloyd25", "lloyd25_kpp"):
                plan.append((m, 25, ""))
            elif m == "tl_emeans":
                plan += [(m, T, p) for T in Ts for p in nprobes]
            else:
                plan += [(m, T, "") for T in Ts]
        for m, T, p in plan:
            key = (a.dataset, str(a.n), str(a.k), m, str(T), str(seed), str(p if m == "tl_emeans" else ""))
            if key in done:
                continue
            r = run_one(m, x, xt, a.k, T, seed, nprobe=p or "def", miss_frac=a.miss)
            r.update(dataset=a.dataset, n=a.n, note=(r.get("note", "") + " " + str(info["threads"]) + "thr").strip())
            append_row(a.out, r)
            done.add(key)
            beat()
            print(f"[{a.tag}] {m} T={T} p={p} seed={seed} mse={r['mse_test']:.5f} wall={r['wall_s']:.1f}s "
                  f"evals={r.get('evals_frac', '')} miss={r.get('miss_frac', '')} load1={r['load1']} "
                  f"elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)
    return 0


def main(argv=None):
    a = build_parser().parse_args(argv)
    run(a)
    print("DONE", a.tag, flush=True)


if __name__ == "__main__":
    sys.exit(main())
