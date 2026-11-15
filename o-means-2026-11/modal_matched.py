"""Matched-error timing: real o-means vs faiss GPU e-means on a fine epoch grid, on Modal.

Question: how long does each method need to reach the SAME held-out error? e-means is run on a
finer epoch grid than o-means so its time-to-error can be read off without wide interpolation.

Safety: one function per dataset, one container at a time, hard timeout (TIMEOUT_S), retries=0.
Third-party code runs only inside the Modal container (no secrets, no volumes).

Run (from this folder):
  modal run modal_matched.py --dataset sift --gpu A10G
  modal run modal_matched.py --dataset mix_hard16k --gpu A100-80GB
"""
import modal

TIMEOUT_S = 900
SRC = "/tmp/claude-0/-root-linkedin-strategy/c45f1333-0fa2-4dd2-8251-933bb945ca06/scratchpad/omeans-full/o_means"
SIFT_URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "numpy", "h5py", "faiss-cpu==1.16.0")
    .add_local_dir(SRC, "/opt/o_means_src", copy=True)
)
app = modal.App("o-means-matched", image=image)


def _install_o_means():
    import os
    import shutil

    import faiss.contrib as c

    dst = os.path.join(c.__path__[0], "o_means")
    if not os.path.exists(dst):
        shutil.copytree("/opt/o_means_src", dst)


@app.function(gpu="A10G", timeout=TIMEOUT_S, retries=0, max_containers=1)
def run(dataset: str, k: int = 16384, n: int = 400000, seeds: int = 5, n_test: int = 20000,
        ts_e: str = "1,1.5,2,2.5,3,4,5,6,8", ts_o: str = "1,2,3,5,8"):
    import io
    import time
    import urllib.request

    import h5py
    import numpy as np
    import torch

    _install_o_means()
    from faiss.contrib.e_means import Emeans
    from faiss.contrib.o_means import Omeans

    out = io.StringIO()

    def w(s):
        out.write(s + "\n")
        print(s, flush=True)

    cap = torch.cuda.get_device_capability(0)
    w(f"env,torch {torch.__version__};cuda {torch.version.cuda};{torch.cuda.get_device_name(0)};cc {cap[0]}.{cap[1]}")

    if dataset == "sift":
        req = urllib.request.Request(SIFT_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp, open("/tmp/sift.hdf5", "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        with h5py.File("/tmp/sift.hdf5", "r") as f:
            x = np.ascontiguousarray(f["train"][: n + n_test], dtype="float32")
    elif dataset == "mix_hard16k":
        modes, spread, dd = 16384, 0.8, 96
        rng = np.random.RandomState(1000)
        centers = (rng.standard_normal((modes, dd)) * spread).astype("float32")
        lab = rng.randint(0, modes, n + n_test)
        x = np.ascontiguousarray(centers[lab] + rng.standard_normal((n + n_test, dd)).astype("float32"), dtype="float32")
    else:
        raise ValueError("unknown dataset " + dataset)
    xg = torch.from_numpy(x[:n]).cuda()
    xe = torch.from_numpy(x[n:n + n_test]).cuda()
    d = x.shape[1]

    def heldout_mse(c):
        c = c.float()
        cn = (c * c).sum(1)
        best = []
        for i in range(0, len(xe), 4096):
            q = xe[i:i + 4096]
            dist = (q * q).sum(1, keepdim=True) - 2 * q @ c.T + cn
            best.append(dist.min(1).values.clamp_min(0))
        return float(torch.cat(best).double().mean())

    plan = [("emeans_gpu", float(t), lambda T, s: Emeans(d, k, nepochs=T, seed=s)) for t in ts_e.split(",")]
    plan += [("omeans_gpu", float(t), lambda T, s: Omeans(d, k, nepochs=T, seed=s, capture_assignments=False))
             for t in ts_o.split(",")]

    # warmup at every budget so Triton compilation and allocator growth are not timed
    for name, T, mk in plan:
        try:
            mk(T, 99).train(xg)
        except Exception as e:
            w(f"warmup_error,{name},{T},{repr(e)[:300].replace(',', ';')}")
    torch.cuda.synchronize()

    w("method,T,seed,k,n,d,wall_s,mse_test")
    for seed in range(seeds):
        for name, T, mk in plan:
            try:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                c, _ = mk(T, seed).train(xg)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                w(f"{name},{T},{seed},{k},{n},{d},{wall:.4f},{heldout_mse(c):.4f}")
                del c
            except Exception as e:
                w(f"ERROR,{name},{T},{seed},{repr(e)[:300].replace(',', ';')}")
    w("MATCHED_DONE")
    return out.getvalue()


@app.local_entrypoint()
def main(dataset: str = "sift", gpu: str = "A10G", k: int = 16384, seeds: int = 5):
    fn = run if gpu == "A10G" else run.with_options(gpu=gpu)
    res = fn.remote(dataset=dataset, k=k, seeds=seeds)
    path = f"results/matched_{dataset}_k{k}_{gpu}.csv"
    with open(path, "w") as f:
        f.write(res)
    print("WROTE", path, flush=True)
