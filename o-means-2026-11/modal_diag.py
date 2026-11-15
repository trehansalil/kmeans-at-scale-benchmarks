"""Isolate the o-means CUDA illegal-memory-access seen at SIFT k=4096 (A10, 2 epochs).

Each (GPU, epochs) trains o-means in a FRESH python process, so one crash cannot poison the next.
If a clean process fails, it is retried once with CUDA_LAUNCH_BLOCKING=1 to capture the failing kernel.
Same safety as modal_gpu.py: one container per GPU, hard timeout, no secrets, no volumes.

Run (from experiments/):  modal run modal_diag.py
"""
import modal

TIMEOUT_S = 900
SRC = "/tmp/claude-0/-root-linkedin-strategy/c45f1333-0fa2-4dd2-8251-933bb945ca06/scratchpad/omeans-full/o_means"
SIFT_URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
GPUS = ["A10G", "L40S", "A100-40GB"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "numpy", "h5py", "faiss-cpu==1.16.0")
    .add_local_dir(SRC, "/opt/o_means_src", copy=True)
)
app = modal.App("o-means-diag", image=image)

WORKER = r"""
import sys, time, torch, numpy as np
from faiss.contrib.o_means import Omeans
k, T, n = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
x = torch.from_numpy(np.load('/tmp/x.npy')[:n]).cuda()
t0 = time.perf_counter()
om = Omeans(x.shape[1], k, nepochs=T, seed=0, capture_assignments=False)
c, _ = om.train(x)
torch.cuda.synchronize()
print('OK', round(time.perf_counter() - t0, 3), om.recipe['nlist'], om.recipe['nprobe'], om.recipe['batch_size'], flush=True)
"""


def _impl(k: int = 4096, n: int = 400000, ts: str = "1,2,3,5"):
    import os
    import shutil
    import subprocess
    import urllib.request

    import faiss.contrib as c
    import h5py
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
    req = urllib.request.Request(SIFT_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, open("/tmp/sift.hdf5", "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    with h5py.File("/tmp/sift.hdf5", "r") as f:
        np.save("/tmp/x.npy", np.ascontiguousarray(f["train"][:n], dtype="float32"))
    open("/tmp/worker.py", "w").write(WORKER)

    def one(T, blocking):
        env = dict(os.environ)
        if blocking:
            env["CUDA_LAUNCH_BLOCKING"] = "1"
        p = subprocess.run(["python", "/tmp/worker.py", str(k), str(T), str(n)], capture_output=True,
                           text=True, timeout=300, env=env)
        return p.returncode, p.stdout.strip().splitlines()[-1:] or [""], p.stderr.strip().splitlines()

    w("gpu_k,T,blocking,rc,result")
    for T in [float(t) for t in ts.split(",")]:
        rc, out, err = one(T, False)
        w(f"k{k},{T},0,{rc},{out[0]}" if rc == 0 else f"k{k},{T},0,{rc},FAIL {(err[-1] if err else '')[:160]}".replace("\n", " "))
        if rc != 0:
            rc2, out2, err2 = one(T, True)
            w(f"k{k},{T},1,{rc2},{out2[0] if rc2 == 0 else 'FAIL'}")
            for ln in [e for e in err2 if "triton" in e.lower() or "o_means" in e or "line" in e][-6:]:
                w("TRACE," + ln.strip()[:200].replace(",", ";"))
    w("DIAG_DONE")
    return "\n".join(lines)


@app.function(gpu="A10G", timeout=TIMEOUT_S, retries=0, max_containers=1)
def run_a10g(k: int = 4096, ts: str = "1,2,3,5"):
    return _impl(k=k, ts=ts)


@app.function(gpu="L40S", timeout=TIMEOUT_S, retries=0, max_containers=1)
def run_l40s(k: int = 4096, ts: str = "1,2,3,5"):
    return _impl(k=k, ts=ts)


@app.function(gpu="A100-40GB", timeout=TIMEOUT_S, retries=0, max_containers=1)
def run_a100(k: int = 4096, ts: str = "1,2,3,5"):
    return _impl(k=k, ts=ts)


funcs = {"A10G": run_a10g, "L40S": run_l40s, "A100-40GB": run_a100}


@app.local_entrypoint()
def main(k: int = 4096, ts: str = "1,2,3,5"):
    for g in GPUS:
        print("GPU_START", g, flush=True)
        try:
            res = funcs[g].remote(k=k, ts=ts)
        except Exception as e:
            res = "REMOTE_EXCEPTION " + repr(e)[:400]
            print(res, flush=True)
        with open(f"results/diag_k{k}_{g}.txt", "w") as f:
            f.write(res + "\n")
        print("WROTE", g, flush=True)
