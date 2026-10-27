"""(P16): .csgs version 6 / flag BSPLINE = knots are control points of a clamped cubic B-spline.
Bit-exact roundtrip (alone and combined with pred / rescale / follow / SN-VQ), golden digest of a v6 stream, version/flag
checks, B-spline weights (partition of unity, end interpolation), K reduction for B-splines (prep), adaptive per-splat K,
and: streams without the flag are byte-identical to before (the v1..v5 golden digests live in the other test files and are
untouched; one more check here compares a flag-less encode with the explicit bspline=False)."""
import hashlib
import random

import numpy as np
import pytest

from codec.csgs import (BSPLINE_DEGREE, Decoder, Encoder, FLAG_BSPLINE, FLAG_PRED, Params, VERSION_BSPLINE, bspline_weights,
                        _unpack_header)
from codec import prep
from test_csgs import GROUPS, N_SN, encode_all, make_stream, params


def bsp_params(shared=True, key_interval=3, L=8, ks=(1, 2, 3, 5, 9), pred=None):
    nq = sum(d for _, d in GROUPS) + N_SN
    return Params(L=L, ks=list(ks), groups=GROUPS, n_sn=N_SN, offsets=[0.5 * i for i in range(nq)],
                  scales=[0.001 * (i + 1) for i in range(nq)], shared_knot=shared, key_interval=key_interval,
                  pred=pred, knn_k=4 if pred and any(m & 2 for m in pred) else 0, bspline=True)


@pytest.mark.parametrize('K', [1, 2, 3, 4, 5, 6, 9, 12])
def test_weights_partition_of_unity_and_clamped_ends(K):
    for i in range(0, 401):
        u = i / 400
        w = bspline_weights(K, u)
        assert len(w) == K and abs(sum(w) - 1.0) < 1e-12 and min(w) >= -1e-15
    assert bspline_weights(K, 0.0)[0] == 1.0 and bspline_weights(K, 1.0)[K - 1] == 1.0
    assert bspline_weights(K, -3.0) == bspline_weights(K, 0.0) and bspline_weights(K, 9.0) == bspline_weights(K, 1.0)


def test_weights_degree_and_linear_case():
    assert BSPLINE_DEGREE == 3
    assert bspline_weights(1, 0.3) == [1.0]
    w = bspline_weights(2, 0.25)                           # K=2: degree 1 = the hat basis
    assert w == [0.75, 0.25]
    # K=4 without interior knots = cubic Bezier (Bernstein)
    u = 0.3
    assert np.allclose(bspline_weights(4, u), [(1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u * u * (1 - u), u ** 3], atol=1e-14)
    # K=5: only 4 of 5 weights are non-zero at any u (local support)
    assert sum(1 for x in bspline_weights(5, 0.3) if x != 0.0) <= 4


def test_weights_reproduce_polynomials():
    """a cubic B-spline with control points on the Greville abscissae reproduces linear functions exactly."""
    K, p = 6, 3
    kn = [0.0] * (p + 1) + [(i + 1) / (K - p) for i in range(K - p - 1)] + [1.0] * (p + 1)
    gre = [sum(kn[i + 1:i + p + 1]) / p for i in range(K)]
    for u in (0.0, 0.17, 0.5, 0.83, 1.0):
        assert abs(sum(w * g for w, g in zip(bspline_weights(K, u), gre)) - u) < 1e-12


@pytest.mark.parametrize('shared', [True, False])
@pytest.mark.parametrize('pred', [None, [0, 0, 0, 0, 0], [5] * 5])
def test_v6_roundtrip_bitexact(shared, pred):
    p = bsp_params(shared, pred=pred)
    secs = make_stream(p, 7, seed=61 + shared)
    bufs = encode_all(p, secs)
    dec = Decoder()
    for s, b in zip(secs, bufs):
        assert b[4] == VERSION_BSPLINE and b[5] & FLAG_BSPLINE
        out = dec.decode(b)
        assert out == s, out.first_difference(s)
        assert dec.params == p and dec.params.bspline
    assert encode_all(p, secs) == bufs


def test_v6_header_version_and_flag_checks():
    p = bsp_params()
    buf = bytearray(encode_all(p, make_stream(p, 1, seed=62))[0])
    hp, h, _, _ = _unpack_header(bytes(buf))
    assert hp.bspline
    import struct
    import zlib

    def fix(b):                                              # recompute the CRC so only the version/flag logic is tested
        b[-4:] = struct.pack('<I', zlib.crc32(bytes(b[:-4])) & 0xffffffff)
        return bytes(b)
    bad = bytearray(buf); bad[4] = 1                          # flag 64 with version 1
    with pytest.raises(ValueError, match='disagree'):
        Decoder().decode(fix(bad))
    bad = bytearray(buf); bad[5] &= ~FLAG_BSPLINE             # version 6 without the flag
    with pytest.raises(ValueError, match='disagree'):
        Decoder().decode(fix(bad))
    plain = bytearray(encode_all(params(True), make_stream(params(True), 1, seed=63))[0]); plain[5] |= FLAG_BSPLINE
    with pytest.raises(ValueError, match='disagree'):
        Decoder().decode(fix(plain))


def test_v6_golden_digest():
    """a change of the B-spline stream layout or of the model contexts must bump the version."""
    p = bsp_params(True, key_interval=2)
    h = hashlib.sha256(b''.join(encode_all(p, make_stream(p, 3, seed=21)))).hexdigest()
    assert h == GOLDEN_V6, h


def test_without_flag_nothing_changes():
    p = params(True, key_interval=2)
    q = Params(L=p.L, ks=p.ks, groups=GROUPS, n_sn=N_SN, offsets=p.offsets, scales=p.scales, shared_knot=True, key_interval=2,
               bspline=False)
    secs = make_stream(p, 3, seed=21)
    a, b = encode_all(p, secs), encode_all(q, secs)
    assert a == b and all(x[4] == 1 and not x[5] & FLAG_BSPLINE for x in a)
    # the same knots with the flag change only the header (version/flags) and the CRC, never the entropy-coded streams
    pb = bsp_params(True, key_interval=2, ks=(1, 2, 3, 5))
    c = encode_all(pb, secs)
    assert all(len(x) == len(y) for x, y in zip(a, c))


def test_v6_combines_with_pred_flag_and_keeps_pred_semantics():
    p = bsp_params(False, pred=[7] * 5)
    secs = make_stream(p, 5, seed=64, huge=False)
    bufs = encode_all(p, secs)
    assert all(b[5] & FLAG_PRED and b[5] & FLAG_BSPLINE and b[4] == VERSION_BSPLINE for b in bufs)
    dec = Decoder()
    for s, b in zip(secs, bufs):
        assert dec.decode(b) == s


# ------------------------------------------------------------------ prep: B-spline K reduction, adaptive K

def _curve(C, nframes):
    span = max(nframes - 1, 1)
    B = np.array([bspline_weights(C.shape[1], f / span) for f in range(nframes)])
    return np.rint(np.einsum('fk,nkd->nfd', B, C.astype(np.float64)))


def test_reduce_k_bspline_picks_smaller_k_only_within_tolerance():
    rnd = np.random.default_rng(5)
    n, d, F = 40, 3, 32
    # build K=5 control points whose curve IS representable with K=3 / K=2 / K=1 (quantised)
    base = rnd.integers(-500, 500, size=(n, 1, d))
    gre = np.array([0, 1, 3, 5, 6])[None, :, None]                                         # Greville abscissae x 6: controls on these give a linear curve
    lin = base + gre * rnd.integers(-9, 9, size=(n, 1, d))                                 # linear motion (exactly representable with K=2)
    C = np.repeat(base, 5, axis=1)                                                         # constant
    C[10:20] = lin[10:20]                                                                  # linear motion
    C[20:] = rnd.integers(-300, 300, size=(n - 20, 5, d))                                  # arbitrary
    kk, rows = prep._reduce_k_bspline(C, (1, 2, 3, 5), 1, F)
    assert all(kk[i] == 1 for i in range(0, 10))
    assert all(kk[i] <= 3 for i in range(10, 20))                                          # a line needs K=2 (K=3 allowed by tolerance)
    for i in range(n):                                                                     # decoded reduced curve within tolerance of the K=5 one
        Ck = np.array(rows[i]).reshape(1, kk[i], d)
        assert np.abs(_curve(Ck, F) - _curve(C[i:i + 1], F)).max() <= 1
        if kk[i] > 1:
            assert (Ck[0, 0] == C[i, 0]).all() and (Ck[0, -1] == C[i, -1]).all()             # boundary knots stay exact
    kk0, _ = prep._reduce_k_bspline(C, (1, 2, 3, 5), 0, F)                                  # tol 0: lossless w.r.t. the K=5 curve
    for i in range(n):
        Ck = np.array(_[i]).reshape(1, kk0[i], d)
        assert np.abs(_curve(Ck, F) - _curve(C[i:i + 1], F)).max() == 0


def test_reduce_adaptive_matches_uniform_call_and_handles_mixed_kn():
    rnd = np.random.default_rng(6)
    C = rnd.integers(-200, 200, size=(30, 9, 3))
    kn = np.full(30, 9)
    for bsp in (False, True):
        a = prep._reduce_adaptive(C, kn, (1, 2, 3, 5, 9), 1, bsp, 32)
        b = prep._reduce_k_bspline(C, (1, 2, 3, 5, 9), 1, 32) if bsp else prep._reduce_k(C, (1, 2, 3, 5, 9), 1)
        assert a[0] == b[0] and a[1] == b[1]
    kn = np.array([9] * 10 + [5] * 10 + [3] * 10)
    kk, rows = prep._reduce_adaptive(C, kn, (1, 2, 3, 5, 9), 1, True, 32)
    assert all(k <= m for k, m in zip(kk, kn))
    assert all(len(r) == 3 * k for r, k in zip(rows, kk))
    assert max(kk[10:20]) <= 5 and max(kk[20:]) <= 3


GOLDEN_V6 = 'c5dc96d96d594211591d9d827396f79c5d7718a7c153537713baafb136a4a4c7'
