"""E3 CPU wall-clock: dense e-means vs the two-level analogue vs Lloyd@25, with NO miss measurement inside the
timed region (miss_frac=0), so the analogue pays only for assignment and cell rebuilds. Wall and CPU seconds
are both recorded; the host is shared, so compare CPU seconds and load1 columns before believing wall time.

    python e3_wallclock.py --dataset sift --n 400000 --k 16384 --seeds 3 --out ../results/e3_sift_k16384.csv \
        --beat ../run/e3.beat --threads 4
"""
import argparse
import sys

import e2_equal_epochs as e2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--Ts", default="1,2,3,5")
    ap.add_argument("--nprobes", default="def,2x")
    ap.add_argument("--no-lloyd", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beat", default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    methods = "emeans,tl_emeans" if a.no_lloyd else "lloyd25,emeans,tl_emeans"
    argv2 = ["--dataset", a.dataset, "--n", str(a.n), "--k", str(a.k), "--seeds", str(a.seeds), "--Ts", a.Ts,
             "--methods", methods, "--nprobes", a.nprobes, "--miss", "0.0", "--out", a.out,
             "--threads", str(a.threads), "--tag", "e3"]
    if a.d:
        argv2 += ["--d", str(a.d)]
    if a.beat:
        argv2 += ["--beat", a.beat]
    e2.run(e2.build_parser().parse_args(argv2))
    print("DONE e3", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
