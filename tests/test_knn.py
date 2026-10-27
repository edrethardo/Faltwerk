"""(S2): codec.knn causal k-NN and Morton order against brute force (determinism, ties, padding)."""
import random
import time

from codec.knn import _icbrt, causal_knn, grid_cell, morton_order


def brute(pos, k):
    out = []
    for i, p in enumerate(pos):
        c = sorted((sum((q[a] - p[a]) ** 2 for a in range(3)), j) for j, q in enumerate(pos[:i]))
        out.append([j for _, j in c[:k]] + [-1] * max(0, k - i))
    return out


def test_icbrt():
    for v in list(range(0, 2000)) + [10 ** 9, 10 ** 12 + 7, 2 ** 40]:
        r = _icbrt(v)
        assert r ** 3 <= v < (r + 1) ** 3, v


def test_random_clustered():
    rng = random.Random(1)
    for trial in range(6):
        n = rng.choice([1, 2, 7, 300, 2000])
        pos = []
        for _ in range(n):
            c = rng.choice([(0, 0, 0), (900, 40, -300), (5, 5, 5)])
            pos.append([c[a] + rng.randint(-60, 60) for a in range(3)])
        for k in (1, 4, 8):
            assert causal_knn(pos, k) == brute(pos, k), (trial, n, k)


def test_duplicates_and_ties():
    pos = [[0, 0, 0]] * 5 + [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [2, 0, 0]] * 3
    for k in (1, 3, 6):
        assert causal_knn(pos, k) == brute(pos, k)


def test_prefix_property():
    rng = random.Random(3)
    pos = [[rng.randint(0, 1000) for _ in range(3)] for _ in range(500)]
    full = causal_knn(pos, 6)
    assert [r[:2] for r in full] == causal_knn(pos, 2)


def test_morton():
    pos = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 0], [7, 7, 7], [-5, 0, 0]]
    o = morton_order(pos)
    assert sorted(o) == list(range(len(pos)))
    assert o.index(0) < o.index(4)                       # equal keys: ties by index
    assert o[-1] == 5 and o[0] == 6
    big = [[v * 1000003, -v, v * 7] for v in range(50)]  # > 21 bit span -> coordinate shift
    assert sorted(morton_order(big)) == list(range(50))


def test_speed_smoke():
    rng = random.Random(5)
    pos = [[rng.randint(0, 20000) for _ in range(3)] for _ in range(20000)]
    t = time.time(); causal_knn(pos, 6); dt = time.time() - t
    assert dt < 30
    assert grid_cell([0, 0, 0], [20000, 20000, 20000], 20000) >= 1


def test_outliers_do_not_blow_up_and_stay_exact():
    rng = random.Random(7)
    pos = [[rng.randint(-50, 2000) for _ in range(3)] for _ in range(300)]
    pos += [[rng.choice([-1, 1]) * rng.randint(1 << 20, 1 << 22) for _ in range(3)] for _ in range(5)]
    rng.shuffle(pos)
    t = time.time(); got = causal_knn(pos, 3); dt = time.time() - t
    assert got == brute(pos, 3) and dt < 10
