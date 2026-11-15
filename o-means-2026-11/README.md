# o-means vs e-means vs the k-means family: independent tests

Code and raw results for the blog post "o-means vs k-means: 2.4x Faster at 16,384 Centroids" on saliltrehan.com.

Status: results are measurements by one person on rented GPUs (Modal) and one Mac mini. They are not a replication of the o-means authors' numbers.

## What was tested

- The released o-means (owlgebra-ai/faiss, contrib/o_means, MIT) against the faiss GPU e-means (Emeans, faiss-cpu 1.16.0), on SIFT and on a synthetic hard mixture, at equal epochs, 3 seeds.
- GPUs: A10G, L40S, A100 80GB PCIe, A100 SXM4. torch 2.8.0+cu128, Triton 3.4.0.
- A CPU analogue of the two-level assignment inside e-means' NumPy trainer. It is not o-means and has no FP16, no split links, no recency closing mean and no GPU kernels.

## Headline results (see the post for conditions)

| Test | Speedup of o-means over e-means (computed) | Held-out error vs e-means |
|---|---|---|
| SIFT, k=16,384, A10G, 1 to 5 epochs | 2.32x to 2.45x | -0.63% at 1 epoch, within about 0.13% after |
| Hard mixture, k=16,384, A100 PCIe, 1 to 5 epochs | 1.50x to 2.36x | +2.1% at 1 epoch, +6.3% at 5 |
| Hard mixture, k=4,096, A100 PCIe | 0.72x to 1.09x | under 0.5% |

The released code crashed in one kernel at SIFT k=4096 on an A10G and an L40S (one seed per cell, cause not isolated); see results/diag_k4096_*.txt.

Not measured: the authors' RTX 5090 numbers, SPACEV 10M, GloVe-200, DEEP, k above 16,384, search recall, matched-error timing.

## Layout

- results/ : raw CSVs and crash diagnostics (gpu2_* is the valid warmed-up SIFT run; gpu_sift_k16384_A10G.csv is a discarded cold-start run kept for the record)
- figures/ : the two charts used in the post
- common.py, twolevel.py, runner.py, e0_sanity.py, e1_fidelity.py, e2_equal_epochs.py, e3_wallclock.py : CPU experiments
- modal_gpu.py, modal_diag.py, modal_hard.py : GPU jobs, each with a hard timeout
- ../common/guard.py : single-job resource guard for the shared Mac (wall cap, RSS cap, memory floor, heartbeat)
- run_all.sh : sequential Mac driver

## Reproduce

CPU: install requirements.txt, then run run_all.sh from this folder (it uses ../common/guard.py). GPU: a Modal account, then run the modal_*.py entrypoints. The SIFT file is downloaded from ann-benchmarks at run time.

## Licence

MIT.
