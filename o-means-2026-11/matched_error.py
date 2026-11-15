"""Matched-error speedup from results/matched_*.csv (output of modal_matched.py).

For each o-means epoch budget: take its mean held-out error, linearly interpolate the time e-means
needs to reach that error on its own (time, error) curve, and report the ratio. Interpolated, so
treat it as computed, not measured. Prints None when o-means error is outside the e-means range.

Usage: python3 matched_error.py results/matched_sift_k16384_A10G.csv
"""
import csv
import statistics as st
import sys


def main(path):
    cur = {}
    for r in csv.reader(open(path)):
        if len(r) == 8 and r[0].endswith("_gpu"):
            cur.setdefault((r[0], float(r[1])), []).append((float(r[6]), float(r[7])))
    mean = {k: (st.mean(w for w, _ in v), st.mean(e for _, e in v)) for k, v in cur.items()}
    e_curve = sorted(v for (m, _), v in mean.items() if m == "emeans_gpu")
    for (m, t_ep), (w, err) in sorted(mean.items()):
        if m != "omeans_gpu":
            continue
        t_e = None
        for (w0, e0), (w1, e1) in zip(e_curve, e_curve[1:]):
            if e0 >= err >= e1:
                t_e = w0 + (w1 - w0) * (e0 - err) / (e0 - e1)
        ratio = None if t_e is None else round(t_e / w, 2)
        print(f"o-means T={t_ep:g} err={err:.2f} time={w:.4f}s -> e-means same err at {t_e and round(t_e, 4)}s, speedup {ratio}")


if __name__ == "__main__":
    main(sys.argv[1])
