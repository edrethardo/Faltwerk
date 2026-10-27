"""(S2): .csgs v3 motion-prediction mode – bit-exact roundtrip for every mode combination, header/version
checks, v1 byte identity without the flag, prediction helpers (resampling, end velocity, neighbour mean)."""
import random
import struct

import pytest

from codec.csgs import (FLAG_PRED, PRED_SPATIAL, PRED_TEMPORAL, PRED_VEL0, VERSION_PRED, Decoder, Encoder, Params,
                        State, end_velocity, predict_diffs, resample_knot, stream_sizes)
from tests.test_csgs import GROUPS, N_SN, make_stream


def params(pred, knn_k=0, shared=False, L=8, ks=(1, 2, 3, 5), key_interval=3):
    nq = sum(d for _, d in GROUPS) + N_SN
    return Params(L=L, ks=list(ks), groups=GROUPS, n_sn=N_SN, offsets=[0.5 * i for i in range(nq)],
                  scales=[0.001 * (i + 1) for i in range(nq)], shared_knot=shared, key_interval=key_interval,
                  pred=pred, knn_k=knn_k)


def roundtrip(p, secs):
    enc, dec = Encoder(p), Decoder()
    bufs = [enc.encode(s) for s in secs]
    for s, b in zip(secs, bufs):
        out = dec.decode(b)
        assert out == s, out.first_difference(s)
        assert dec.params == p
    return bufs


@pytest.mark.parametrize('mode', [0, PRED_TEMPORAL, PRED_SPATIAL, PRED_TEMPORAL | PRED_SPATIAL, PRED_VEL0,
                                  PRED_TEMPORAL | PRED_SPATIAL | PRED_VEL0])
@pytest.mark.parametrize('shared', [False, True])
def test_roundtrip_all_modes(mode, shared):
    p0 = params(None, shared=shared)
    secs = make_stream(p0, 4, n0=60, seed=mode * 7 + int(shared))
    p = params([mode] * len(GROUPS), knn_k=3 if mode & PRED_SPATIAL else 0, shared=shared)
    bufs = roundtrip(p, secs)
    assert all(b[4] == VERSION_PRED and (b[5] & FLAG_PRED) for b in bufs)


def test_mixed_modes_per_group_and_knn_k_variants():
    p0 = params(None)
    secs = make_stream(p0, 3, n0=80, seed=11)
    modes = [PRED_TEMPORAL | PRED_SPATIAL, PRED_TEMPORAL, PRED_SPATIAL | PRED_VEL0, 0, PRED_VEL0]
    for k in (1, 6):
        roundtrip(params(modes, knn_k=k), secs)


def test_v1_bytes_unchanged_without_pred():
    p = params(None)
    secs = make_stream(p, 3, n0=50, seed=2)
    bufs = [Encoder(p).encode(s, key=True) for s in secs[:1]]
    assert bufs[0][4] == 1 and not (bufs[0][5] & FLAG_PRED)
    sizes = stream_sizes(bufs[0])
    assert set(sizes) == {'birth', 'death', 'ids', 'k', 'knots_P', 'knots_R', 'knots_S', 'knots_O', 'knots_D', 'sn', 'header'}


def test_header_carries_modes_and_rejects_mismatch():
    p = params([1, 2, 3, 4, 7], knn_k=5)
    secs = make_stream(params(None), 1, n0=10, seed=3)
    buf = Encoder(p).encode(secs[0])
    assert Decoder().decode(buf).ids == secs[0].ids
    d = Decoder(); d.decode(buf)
    assert d.params.pred == [1, 2, 3, 4, 7] and d.params.knn_k == 5
    bad = bytearray(buf); bad[4] = 1                     # version 1 but FLAG_PRED set
    bad[-4:] = struct.pack('<I', __import__('zlib').crc32(bytes(bad[:-4])) & 0xffffffff)
    with pytest.raises(ValueError):
        Decoder().decode(bytes(bad))
    bad = bytearray(buf); bad[4] = 9
    bad[-4:] = struct.pack('<I', __import__('zlib').crc32(bytes(bad[:-4])) & 0xffffffff)
    with pytest.raises(ValueError):
        Decoder().decode(bytes(bad))


def test_params_validation():
    with pytest.raises(ValueError):
        params([PRED_SPATIAL] * 5, knn_k=0)
    with pytest.raises(ValueError):
        params([0] * 5, knn_k=3)
    with pytest.raises(ValueError):
        params([0] * 4)
    nq = sum(d for _, d in GROUPS) + N_SN
    with pytest.raises(ValueError):                      # P must come before a spatial group
        Params(L=8, ks=[1, 2], groups=[('R', 4), ('P', 3)], n_sn=0, offsets=[0.0] * 7, scales=[1.0] * 7,
               pred=[PRED_SPATIAL, 0], knn_k=2)


def test_resample_knot_is_linear_and_exact_at_knots():
    row = [0, 100, 40, 60, 80, 0]                          # K=3, d=2: knots (0,100) (40,60) (80,0)
    assert resample_knot(row, 3, 2, 0, 5) == [0, 100]
    assert resample_knot(row, 3, 2, 2, 5) == [40, 60]
    assert resample_knot(row, 3, 2, 4, 5) == [80, 0]
    assert resample_knot(row, 3, 2, 1, 5) == [20, 80]
    assert resample_knot(row, 3, 2, 1, 2) == [80, 0]       # target K=2: knot 1 = end
    assert resample_knot([7, 8], 1, 2, 3, 5) == [7, 8]     # constant curve
    assert resample_knot([1, 2], 2, 1, 1, 3) == [2]        # rounding half up: 1 + (1*1*2+2)//4 = 2


def test_end_velocity_and_predict_diffs():
    assert end_velocity([0, 0, 0, 31, 31, 31], 2, 3, 32) == [1, 1, 1]       # 31 steps over 31 frames
    assert end_velocity([5, 5, 5, 5, 5, 5], 2, 3, 32) == [0, 0, 0]
    assert end_velocity([9], 1, 1, 32) == [0]
    assert end_velocity([0, 8], 2, 1, 1) == [0]
    assert end_velocity([0, 10, 30], 3, 1, 5) == [10]                        # 20 * 2 / 4 = 10
    # two neighbours moving +4 / +2 per knot on K=5 -> mean +3 on a K=5 target, +6 on K=3 (two steps)
    n1 = [0, 4, 8, 12, 16]; n2 = [0, 2, 4, 6, 8]
    assert predict_diffs([n1, n2], [5, 5], 1, 5, 2) == [3, 3, 3, 3]
    assert predict_diffs([n1, n2], [5, 5], 1, 3, 2) == [6, 6]
    assert predict_diffs([], [], 1, 3, 0) == [0, 0]
    assert predict_diffs([[0, -3]], [2], 1, 5, 1) == [-1, 0, -1, -1]        # resampled -1 -1 -2 -3 (round half up)


def test_state_after_keeps_velocity_only_for_continuing():
    p = params([PRED_VEL0] * 5)
    secs = make_stream(params(None), 2, n0=30, seed=5)
    st = State.after(secs[0], p)
    assert len(st.vel) == len(GROUPS) and all(len(v) == len(st.ids) for v in st.vel)
    for g, (_, d) in enumerate(GROUPS):
        for i, sid in enumerate(st.ids):
            j = secs[0].ids.index(sid)
            assert st.vel[g][i] == end_velocity(secs[0].knots[g][j], secs[0].k[g][j], d, secs[0].nframes)


def test_random_access_key_variant_in_pred_mode():
    p = params([PRED_TEMPORAL | PRED_SPATIAL] * 5, knn_k=4)
    secs = make_stream(params(None), 3, n0=40, seed=9)
    bufs = roundtrip(p, secs)
    kb = Encoder(p).encode(secs[2], key=True)
    assert Decoder().decode(kb) == secs[2]
    with pytest.raises(ValueError):                      # delta section without its predecessor
        Decoder().decode(bufs[1])
