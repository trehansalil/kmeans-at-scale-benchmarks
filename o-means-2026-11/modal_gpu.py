"""GPU run of the REAL o-means code (owlgebra-ai/faiss contrib/o_means) vs faiss GPU e-means, on Modal.

Safety: one function, one container, hard timeout (TIMEOUT_S). Third-party code runs only inside the
Modal container (no secrets, no volumes). Results come back to the caller as CSV text.

Run (from experiments/):  modal run modal_gpu.py --gpu A10G --k 16384 --seeds 3
"""
import modal

TIMEOUT_S = 1200
SRC = "/tmp/claude-0/-root-linkedin-strategy/c45f1333-0fa2-4dd2-8251-933bb945ca06/scratchpad/omeans-full/o_means"
SIFT_URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "numpy", "h5py", "faiss-cpu==1.16.0")
    .add_local_dir(SRC, "/opt/o_means_src", copy=True)
)
app = modal.App("o-means-check", image=image)


def _install_o_means():
    import os
    import shutil

    import faiss.contrib as c

    dst = os.path.join(c.__path__[0], "o_means")
    if not os.path.exists(dst):
        shutil.copytree("/opt/o_means_src", dst)


@app.function(gpu="A10G", timeout=TIMEOUT_S, retries=0, max_containers=1)
def run(k: int = 16384, n: int = 400000, seeds: int = 3, ts: str = "1,2,3,5", n_test: int = 20000,
        dataset: str = "sift"):
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
    try:
        import triton

        w("triton," + triton.__version__)
    except Exception as e:
        w("triton,ERR " + repr(e))

    if dataset == "sift":
        req = urllib.request.Request(SIFT_URL, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp, open("/tmp/sift.hdf5", "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
        except Exception as e:
            raise RuntimeError("SIFT download failed: " + str(e)) from None
        with h5py.File("/tmp/sift.hdf5", "r") as f:
            x = np.ascontiguousarray(f["train"][: n + n_test], dtype="float32")
    elif dataset == "mix_hard16k":
        # same generator as experiments/common.py MIXTURES["mix_hard16k"], seed 0
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

    def mk_e(T, s):
        return Emeans(d, k, nepochs=T, seed=s)

    def mk_o(T, s):
        return Omeans(d, k, nepochs=T, seed=s, capture_assignments=False)

    # full-size warmup at every epoch budget so Triton compiles and allocator growth are not timed
    for name, mk in (("emeans_gpu", mk_e), ("omeans_gpu", mk_o)):
        for T in [float(t) for t in ts.split(",")]:
            try:
                mk(T, 99).train(xg)
            except Exception as e:
                w(f"warmup_error,{name},{T},{repr(e)[:300].replace(',', ';')}")
    torch.cuda.synchronize()

    w("method,T,seed,k,n,d,wall_s,mse_test,nlist,nprobe,gpu_free_gb")
    for seed in range(seeds):
        for T in [float(t) for t in ts.split(",")]:
            for name, mk in (("emeans_gpu", mk_e), ("omeans_gpu", mk_o)):
                try:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    m = mk(T, seed)
                    c, _ = m.train(xg)
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - t0
                    rec = getattr(m, "recipe", None) or {}
                    free = torch.cuda.mem_get_info()[0] / 2**30
                    w(f"{name},{T},{seed},{k},{n},{d},{wall:.4f},{heldout_mse(c):.4f},"
                      f"{rec.get('nlist', '')},{rec.get('nprobe', '')},{free:.1f}")
                    del c, m
                except Exception as e:
                    w(f"ERROR,{name},{T},{seed},{repr(e)[:300].replace(',', ';')}")
    w("GPU_RUN_DONE")
    return out.getvalue()


@app.local_entrypoint()
def main(gpu: str = "A10G", configs: str = "sift:16384,sift:4096,mix_hard16k:4096", n: int = 400000,
         seeds: int = 3, ts: str = "1,2,3,5"):
    """configs: comma list of dataset:k, run one container at a time (each capped by TIMEOUT_S)."""
    fn = run if gpu == "A10G" else run.with_options(gpu=gpu)
    for cfg in configs.split(","):
        ds, k = cfg.split(":")
        res = fn.remote(k=int(k), n=n, seeds=seeds, ts=ts, dataset=ds)
        path = f"results/gpu2_{ds}_k{k}_{gpu}.csv"
        with open(path, "w") as f:
            f.write(res)
        print("WROTE", path, flush=True)
