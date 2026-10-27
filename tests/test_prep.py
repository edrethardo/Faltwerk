"""(P2): encode-time K reduction with tolerance and raw (unnormalised) quaternion knots in codec.prep."""
import numpy as np

from codec.prep import _quant_floats, _reduce_k, _raw_quat_scale, hat_basis, q8, quant_sn


def test_reduce_k_lossless_keeps_k_when_knots_bend():
    C = np.array([[[0], [10], [0], [10], [0]]], np.int64)          # zig-zag: no smaller K reproduces it
    kk, rows = _reduce_k(C, (1, 2, 3, 5), tol=0)
    assert kk == [5] and rows == [[0, 10, 0, 10, 0]]


def test_reduce_k_exact_line_goes_to_k2_and_keeps_end_knots():
    C = np.array([[[0, 4], [2, 3], [4, 2], [6, 1], [8, 0]]], np.int64)
    kk, rows = _reduce_k(C, (1, 2, 3, 5), tol=0)
    assert kk == [2] and rows == [[0, 4, 8, 0]]


def test_reduce_k_tolerance_allows_small_deviation_only():
    C = np.array([[[0], [11], [20], [30], [40]],                   # 1 step off a line
                  [[0], [14], [20], [30], [40]]], np.int64)        # 4 steps off
    assert _reduce_k(C, (1, 2, 3, 5), tol=0)[0] == [5, 5]
    kk, rows = _reduce_k(C, (1, 2, 3, 5), tol=1)
    assert kk == [2, 5]
    assert rows[0] == [0, 40]
    # the reduced curve evaluated at the 5 knot positions stays within tol of the trained knots
    rec = hat_basis(5, 2) @ np.array([[0.0], [40.0]])
    assert np.abs(np.round(rec) - C[0]).max() <= 1


def test_reduce_k_k1_needs_identical_end_knot():
    C = np.array([[[5], [5], [5], [5], [6]]], np.int64)            # within tol 1 of a constant, but last knot differs
    assert _reduce_k(C, (1, 2, 3, 5), tol=1)[0] == [2]
    C2 = np.array([[[5], [6], [5], [6], [5]]], np.int64)
    assert _reduce_k(C2, (1, 2, 3, 5), tol=1)[0] == [1]


def test_raw_quat_scale_is_per_id_max_over_sections_and_keeps_nlerp():
    rng = np.random.default_rng(0)
    q0 = rng.normal(size=(3, 5, 4)) * np.array([1, 2, 3])[:, None, None]
    q1 = rng.normal(size=(2, 5, 4)) * np.array([0.1, 9])[:, None, None]   # id 1: max in g0, id 2: max in g1
    cv = {'g0_ids': np.array([0, 1, 2]), 'g0_quats': q0, 'g1_ids': np.array([1, 2]), 'g1_quats': q1}
    mx = _raw_quat_scale(cv, 2)
    for i in range(3):
        cand = [np.linalg.norm(q0[i], axis=-1).max()] + ([np.linalg.norm(q1[i - 1], axis=-1).max()] if i else [])
        assert np.isclose(mx[i], max(cand))
    # one scalar per splat: nlerp of the scaled knots == nlerp of the raw knots
    a, b, w = q0[1, 0], q0[1, 1], 0.3
    n = lambda x: x / np.linalg.norm(x)
    s = 1 / mx[1]
    assert np.allclose(n((1 - w) * a * s + w * b * s), n((1 - w) * a + w * b))


def _dequant_sn(SN, sn_deg, sn_bits):
    q, (mn, mx) = quant_sn(SN, sn_deg, sn_bits)
    zero3 = (np.zeros(3), np.ones(3))
    off, sc = _quant_floats(np.zeros(3), 1.0, {'S': zero3, 'O': (0.0, 1.0), 'D': zero3, 'SN': (mn, mx)}, 45,
                            sn_levels=(1 << sn_bits) - 1)
    return q, np.asarray(off[14:]) + q * np.asarray(sc[14:])


def test_quant_sn_default_is_q8():
    SN = np.random.default_rng(1).normal(size=(50, 45)).astype(np.float32)
    q, _ = quant_sn(SN)
    assert np.array_equal(q, q8(SN, SN.min(0), SN.max(0)))


def test_quant_sn_drops_bands_and_dequantises_within_half_step():
    SN = np.random.default_rng(2).normal(size=(50, 45)).astype(np.float32)
    q, v = _dequant_sn(SN, sn_deg=1, sn_bits=5)
    assert q.max() <= 31
    assert (q[:, 9:] == 0).all() and np.allclose(v[:, 9:], 0)
    step = (SN[:, :9].max(0) - SN[:, :9].min(0)) / 31
    assert (np.abs(v[:, :9] - SN[:, :9]) <= step / 2 + 1e-6).all()
