"""Hard-mixture run (mix_hard16k, n=400k, d=96) of REAL o-means vs faiss GPU e-means on one A100, crash-safe.

A worker process runs the job list from a start index; if a kernel crashes, the worker prints FAIL and exits,
the parent records the failure and restarts the worker at the next job. Hard timeout, no secrets, no volumes.

Run (from experiments/):  modal run modal_hard.py
"""
import modal

TIMEOUT_S = 1200
SRC = "/tmp/claude-0/-root-linkedin-strategy/c45f1333-0fa2-4dd2-8251-933bb945ca06/scratchpad/omeans-full/o_means"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "numpy", "faiss-cpu==1.16.0")
    .add_local_dir(SRC, "/opt/o_means_src", copy=True)
)
app = modal.App("o-means-hard", image=image)

WORKER = r"""
import sys, time, torch, numpy as np
from faiss.contrib.e_means import Emeans
from faiss.contrib.o_means import Omeans
k, start, seeds, ts = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), [float(t) for t in sys.argv[4].split(',')]
X = np.load('/tmp/x.npy'); n = len(X) - 20000
xg = torch.from_numpy(X[:n]).cuda(); xe = torch.from_numpy(X[n:]).cuda(); d = xg.shape[1]
jobs = [(m, T, s) for s in range(seeds) for T in ts for m in ('emeans_gpu', 'omeans_gpu')]
def mk(m, T, s):
    return Emeans(d, k, nepochs=T, seed=s) if m == 'emeans_gpu' else Omeans(d, k, nepochs=T, seed=s, capture_assignments=False)
def mse(c):
    cn = (c * c).sum(1); b = []
    for i in range(0, len(xe), 4096):
        q = xe[i:i + 4096]; b.append(((q * q).sum(1, keepdim=True) - 2 * q @ c.T + cn).min(1).values.clamp_min(0))
    return float(torch.cat(b).double().mean())
try:
    for m in ('emeans_gpu', 'omeans_gpu'):
        for T in ts:
            mk(m, T, 99).train(xg)
    torch.cuda.synchronize()
except Exception as e:
    print('WARMUP_FAIL', repr(e)[:150].replace(',', ';'), flush=True); sys.exit(4)
for i in range(start, len(jobs)):
    m, T, s = jobs[i]
    try:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        o = mk(m, T, s); c, _ = o.train(xg); torch.cuda.synchronize(); wall = time.perf_counter() - t0
        r = getattr(o, 'recipe', None) or {}
        print(f"R,{i},{m},{T},{s},{k},{wall:.4f},{mse(c):.4f},{r.get('nlist','')},{r.get('nprobe','')},{r.get('batch_size','')}", flush=True)
    except Exception as e:
        print(f"FAIL,{i},{m},{T},{s},{k},{repr(e)[:150].replace(',', ';')}", flush=True); sys.exit(3)
print('WORKER_DONE', flush=True)
"""


@app.function(gpu="A100-80GB", timeout=TIMEOUT_S, retries=0, max_containers=1)
def run(ks: str = "4096,16384", n: int = 400000, seeds: int = 3, ts: str = "1,2,3,5"):
    import os
    import shutil
    import subprocess

    import faiss.contrib as c
    import numpy as np
    import torch

    dst = os.path.join(c.__path__[0], "o_means")
    if not os.path.exists(dst):
        shutil.copytree("/opt/o_means_src", dst)
    lines = []

    def w(s):
        lines.append(s)
        print(s, flush=True)

    cap = torch.cuda.get_device_capability(0)
    w(f"env,torch {torch.__version__};{torch.cuda.get_device_name(0)};cc {cap[0]}.{cap[1]}")
    modes, spread, dd, n_test = 16384, 0.8, 96, 20000
    rng = np.random.RandomState(1000)
    centers = (rng.standard_normal((modes, dd)) * spread).astype("float32")
    lab = rng.randint(0, modes, n + n_test)
    x = centers[lab] + rng.standard_normal((n + n_test, dd)).astype("float32")
    np.save("/tmp/x.npy", np.ascontiguousarray(x, dtype="float32"))
    open("/tmp/worker.py", "w").write(WORKER)
    njobs = seeds * len(ts.split(",")) * 2
    w("row,job,method,T,seed,k,wall_s,mse_test,nlist,nprobe,batch")
    for k in [int(v) for v in ks.split(",")]:
        idx, restarts = 0, 0
        while idx < njobs and restarts <= 6:
            p = subprocess.run(["python", "/tmp/worker.py", str(k), str(idx), str(seeds), ts], capture_output=True,
                               text=True, timeout=600)
            done = [ln for ln in p.stdout.splitlines() if ln.startswith(("R,", "FAIL,", "WARMUP_FAIL", "WORKER_DONE"))]
            for ln in done:
                w(ln)
            last = [ln for ln in done if ln.startswith(("R,", "FAIL,"))]
            if any(ln.startswith("WARMUP_FAIL") for ln in done) and not last:
                w(f"ABORT_K,{k},warmup failed")
                break
            if not last:
                w(f"ABORT_K,{k},no progress;{(p.stderr.strip().splitlines() or [''])[-1][:120]}")
                break
            idx = int(last[-1].split(",")[1]) + 1
            restarts += 1 if p.returncode != 0 else 0
    w("HARD_DONE")
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    try:
        res = run.remote()
    except Exception as e:
        res = "REMOTE_EXCEPTION " + repr(e)[:400]
        print(res, flush=True)
    with open("results/gpu_hard_A100.csv", "w") as f:
        f.write(res + "\n")
    print("WROTE results/gpu_hard_A100.csv", flush=True)
