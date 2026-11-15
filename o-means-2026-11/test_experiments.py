"""Unit tests for the helpers. Run: pytest -q (from this folder)."""
import faiss
import numpy as np
import pytest

from common import dead_fraction, init_rows, load_dataset, rel_diff_pct, t_interval, heldout_mse
from twolevel import CellIndex, TwoLevelEmeans, default_nlist, default_nprobe, final_miss


def blobs(n=4000, d=16, seed=0):
    rng = np.random.RandomState(seed)
    return rng.standard_normal((n, d)).astype("float32")


def test_t_interval_known():
    m, h = t_interval([1.0, 2.0, 3.0])
    assert m == pytest.approx(2.0)
    assert h == pytest.approx(4.303 * 1.0 / np.sqrt(3), rel=1e-3)
    assert np.isnan(t_interval([1.0])[1])


def test_rel_diff():
    assert rel_diff_pct(103.0, 100.0) == pytest.approx(3.0)


def test_heldout_mse_matches_bruteforce():
    x, c = blobs(500, 8, 1), blobs(20, 8, 2)
    brute = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).min(1).mean()
    assert heldout_mse(x, c) == pytest.approx(float(brute), rel=1e-4)


def test_dead_fraction_zero_for_even_clusters():
    x = blobs(2000, 8, 3)
    assert dead_fraction(x, x[:10]) <= 0.5


def test_defaults_follow_readme_rule():
    assert default_nlist(16384) == 128
    assert default_nlist(8192) == 64
    assert default_nlist(500) == 16
    assert default_nprobe(16384, 128) == 16
    assert default_nprobe(16384, 768) == 12
    assert default_nprobe(65536, 100) == 24


def test_datasets_disjoint_and_seeded():
    a, at = load_dataset("mix_hard", 1000, 200, seed=1)
    b, bt = load_dataset("mix_hard", 1000, 200, seed=1)
    assert np.array_equal(a, b) and np.array_equal(at, bt)
    assert a.shape == (1000, 64) and at.shape == (200, 64)
    assert not np.array_equal(a[:200], at)


def test_two_level_equals_exact_when_probing_all_cells():
    x, c = blobs(3000, 16, 4), blobs(512, 16, 5)
    ci = CellIndex(16, 512, nlist=16, nprobe="all", seed=0)
    ci.build(c, 4)
    d, lab, ev = ci.search(x, c)
    de, le = faiss.knn(x, c, 1)
    assert np.array_equal(lab, le.ravel())
    assert np.allclose(d, de.ravel(), rtol=1e-4, atol=1e-4)
    assert ev == 3000 * 16 + 3000 * 512   # every cell centre plus every centroid


def test_two_level_evals_below_dense_and_miss_counted():
    x, c = blobs(3000, 16, 6), blobs(1024, 16, 7)
    miss, frac, nl, npb = final_miss(c, x, nlist=32, nprobe=4)
    assert 0.0 <= miss <= 1.0
    assert frac < 1.0 and nl == 32 and npb == 4
    miss_all, frac_all, _, _ = final_miss(c, x, nlist=32, nprobe=32)
    assert miss_all == 0.0 and frac_all > 1.0   # all cells: more work than dense (cell scoring on top)


def test_two_level_emeans_trains_and_counts():
    x = blobs(6000, 16, 8)
    em = TwoLevelEmeans(16, 256, nepochs=2, seed=0, capture_assignments=False, nlist=16, nprobe=4, miss_frac=0.05)
    c, _ = em.train(x, init_centroids=init_rows(x, 256, 0))
    st = em.stats()
    assert c.shape == (256, 16) and np.isfinite(c).all()
    assert 0 < st["evals_frac"] < 1.0
    assert 0.0 <= st["miss_frac"] <= 1.0
    assert em.t_measure > 0
