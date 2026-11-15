"""E0 sanity: reproduce the existing e-means row on THIS machine before trusting anything else.

Reference (reference/e-means/experiments/RESULTS.md, E1): mix_hard, n=100k, d=64, k=500: e-means needs about
3 epochs (interval 1.5-4.5) to reach the test MSE of Lloyd@25. Also checks the analogue is exact when it
probes every cell. Prints 'SANITY PASS' or 'SANITY FAIL' and exits non-zero on FAIL.

    python e0_sanity.py --out ../results/e0_sanity.csv --beat ../run/e0.beat --threads 4 [--n 100000 --k 500 --seeds 3]
"""
import argparse
import csv
import sys

import e2_equal_epochs as e2


def summarise(path):
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault((r["method"], r["T"], r["nprobe"]), {})[int(r["seed"])] = float(r["mse_test"])
    seeds = sorted({int(r["seed"]) for r in rows})
    lloyd = {s: by[("lloyd25", "25", "")][s] for s in seeds}
    reach = []
    for s in seeds:
        t_hit = None
        for T in (1, 2, 3, 5):
            m = by.get(("emeans", str(T), ""), {}).get(s)
            if m is not None and m <= lloyd[s]:
                t_hit = T
                break
        reach.append(t_hit)
    exact_gap = []
    for T in (1, 2, 3, 5):
        a = by.get(("tl_emeans", str(T), "all"), {})
        b = by.get(("emeans", str(T), ""), {})
        for s in seeds:
            if s in a and s in b:
                exact_gap.append(abs(a[s] - b[s]) / b[s] * 100)
    return seeds, reach, exact_gap


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--beat", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    args = e2.build_parser().parse_args([
        "--dataset", "mix_hard", "--n", str(a.n), "--k", str(a.k), "--seeds", str(a.seeds), "--Ts", "1,2,3,5",
        "--methods", "lloyd25,emeans,tl_emeans", "--nprobes", "all", "--miss", "0.0", "--out", a.out,
        "--threads", str(a.threads), "--tag", "e0"] + (["--beat", a.beat] if a.beat else []))
    e2.run(args)
    seeds, reach, gap = summarise(a.out)
    print("epochs for e-means to reach Lloyd@25 test MSE per seed:", reach, "(reference: about 3, interval 1.5-4.5)")
    print("analogue with nprobe=all vs dense e-means, |MSE diff| %:", [round(g, 3) for g in gap])
    ok_reach = all(r is not None for r in reach)
    ok_exact = bool(gap) and max(gap) < 1.0
    if ok_reach and ok_exact:
        print("SANITY PASS")
        print("DONE e0")
        return 0
    print("SANITY FAIL reach_ok=%s exact_ok=%s" % (ok_reach, ok_exact))
    return 1


if __name__ == "__main__":
    sys.exit(main())
