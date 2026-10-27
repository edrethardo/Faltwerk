"""(S1): .csgs v2 SN-VQ mode (SH degree per splat + codebook indices) - roundtrip, state carry, rejects."""
import hashlib
import random

import pytest

from codec.csgs import (Decoder, Encoder, Params, Section, State, VERSION, VERSION_SN_VQ, FLAG_SN_VQ, sn_dim,
                        stream_sizes, vq_row)
from tests.test_csgs import GROUPS, make_stream, encode_all

N_SN = 45
LEVELS = [(1, [8]), (3, [16, 4])]           # degree 1: one stage of 8; degree 3: 16 entries + 4 residual entries


def vq_params(levels=LEVELS, key_interval=3, L=8, shared=True):
    nq = sum(d for _, d in GROUPS) + N_SN
    zero = [120 + (i % 7) for i in range(N_SN)]
    return Params(L=L, ks=[1, 2, 3, 5], groups=GROUPS, n_sn=N_SN, offsets=[0.5 * i for i in range(nq)],
                  scales=[0.001 * (i + 1) for i in range(nq)], shared_knot=shared, key_interval=key_interval,
                  sn_vq=levels, sn_zero=zero)


def random_codebooks(p, rnd):
    cb = []
    for deg, sizes in p.sn_vq:
        d = sn_dim(deg)
        lvl = [[[rnd.randint(0, 255) for _ in range(d)] for _ in range(sizes[0])]]
        for s in sizes[1:]:
            lvl.append([[rnd.randint(-6, 6) for _ in range(d)] for _ in range(s)])
        cb.append(lvl)
    return cb


def make_vq_stream(p, nsec, seed=0, n0=40):
    """v1 random sections (tests.test_csgs.make_stream) + SN-VQ fields: new codebooks in every key section, every
    coded splat gets a random level and indices; continuing splats keep theirs; sn = vq_row(...)."""
    rnd = random.Random(seed)
    secs = make_stream(p, nsec, n0=n0, seed=seed)
    degs = [d for d, _ in p.sn_vq]
    cb = None
    carry = {}                                    # id -> (deg, idx, sn) of continuing splats
    for s in secs:
        key = s.index % p.key_interval == 0
        if key:
            cb = random_codebooks(p, rnd)
            s.sn_cb = cb
        s.sn_deg, s.sn_idx, s.sn = [], [], []
        for i, sid in enumerate(s.ids):
            if not key and i < s.n_cont:
                dg, ix, row = carry[sid]
            else:
                lv = rnd.randrange(len(degs) + 1)
                dg = degs[lv - 1] if lv else 0
                ix = [rnd.randrange(sz) for sz in p.sn_vq[lv - 1][1]] if lv else []
                row = vq_row(lv, ix, cb, p.sn_zero, degs)
            s.sn_deg.append(dg); s.sn_idx.append(ix); s.sn.append(row)
            if s.death[i] == s.nframes + 1:
                carry[sid] = (dg, ix, row)
    return secs


def test_vq_row_sums_stages_clips_and_keeps_zero_row_beyond_the_degree():
    degs = [1, 3]
    cb = [[[[10] * 9]], [[[250] * 45], [[10] * 45, [-20] * 45]]]
    zero = list(range(45))
    assert vq_row(0, [], cb, zero, degs) == zero
    r = vq_row(1, [0], cb, zero, degs)
    assert r[:9] == [10] * 9 and r[9:] == zero[9:]
    assert vq_row(2, [0, 0], cb, zero, degs) == [255] * 45         # 250 + 10 clipped
    assert vq_row(2, [0, 1], cb, zero, degs) == [230] * 45


def test_sn_dim():
    assert [sn_dim(d) for d in (0, 1, 2, 3)] == [0, 9, 24, 45]


@pytest.mark.parametrize('shared', [True, False])
def test_vq_stream_roundtrip_bitexact_with_state_carry(shared):
    p = vq_params(shared=shared)
    secs = make_vq_stream(p, 7, seed=31 + shared)
    assert any(s.n_cont and s.index % p.key_interval for s in secs)      # delta sections with continuing splats
    bufs = encode_all(p, secs)
    dec = Decoder()
    for s, b in zip(secs, bufs):
        out = dec.decode(b)
        assert out == s, out.first_difference(s)
        assert out.key == (s.index % p.key_interval == 0)
        assert dec.params == p and dec.params.sn_vq == LEVELS
        assert b[4] == VERSION_SN_VQ and b[5] & FLAG_SN_VQ
        sz = stream_sizes(b)
        assert sz['sn'] == 0                                            # raw SN stream is empty in VQ mode
        assert (sz['sn_cb'] > 0) == out.key                             # codebooks only in key sections
    assert encode_all(p, secs) == bufs


def test_vq_random_access_from_key_section_carries_codebooks():
    p = vq_params(key_interval=3)
    secs = make_vq_stream(p, 8, seed=5)
    bufs = encode_all(p, secs)
    dec = Decoder()
    for s, b in zip(secs[3:], bufs[3:]):
        assert dec.decode(b) == s
    with pytest.raises(ValueError, match='key section'):
        Decoder().decode(bufs[4])


def test_v1_sections_still_decode_and_carry_no_vq_fields():
    from tests.test_csgs import params
    p = params(True, key_interval=2)
    secs = make_stream(p, 3, seed=21)
    bufs = encode_all(p, secs)
    assert all(b[4] == VERSION for b in bufs)
    dec = Decoder()
    for s, b in zip(secs, bufs):
        out = dec.decode(b)
        assert out == s and out.sn_deg is None and out.sn_idx is None and out.sn_cb is None
        assert 'sn_cb' not in stream_sizes(b)


def test_vq_format_is_frozen():
    """golden digest of a synthetic v2 stream: any change of models, contexts or layout must bump VERSION_SN_VQ."""
    p = vq_params(key_interval=2)
    h = hashlib.sha256(b''.join(encode_all(p, make_vq_stream(p, 3, seed=77)))).hexdigest()
    assert h == GOLDEN_V2, h


GOLDEN_V2 = '3927b223866506f9347581589896184aece07aa27241db78ae58945e78525057'


def test_vq_index_cost_is_near_log2_size():
    """uniform random indices of a 4096-entry codebook cost about 12 bit each (adaptive model, 1 stage)."""
    p = vq_params(levels=[(3, [4096])], key_interval=100)
    secs = make_vq_stream(p, 1, seed=9, n0=3000)
    secs[0].sn_deg = [3] * len(secs[0].ids)
    rnd = random.Random(1)
    secs[0].sn_idx = [[rnd.randrange(4096)] for _ in secs[0].ids]
    secs[0].sn = [vq_row(1, ix, secs[0].sn_cb, p.sn_zero, [3]) for ix in secs[0].sn_idx]
    b = Encoder(p).encode(secs[0])
    bits = stream_sizes(b)['sn_idx'] * 8 / len(secs[0].ids)
    assert 12.0 <= bits < 12.6, bits
    assert Decoder().decode(b) == secs[0]


def test_vq_encoder_rejects_bad_input():
    p = vq_params(key_interval=10)
    secs = make_vq_stream(p, 2, seed=12)
    enc = Encoder(p); enc.encode(secs[0])
    bad = make_vq_stream(p, 2, seed=12)[1]
    bad.sn_cb = secs[0].sn_cb
    with pytest.raises(ValueError, match='key sections only'):
        enc.encode(bad)
    s = make_vq_stream(p, 1, seed=13)[0]
    s.sn[0][0] = (s.sn[0][0] + 1) % 256
    with pytest.raises(ValueError, match='VQ reconstruction'):
        Encoder(p).encode(s)
    s = make_vq_stream(p, 1, seed=14)[0]
    s.sn_deg[0] = 2
    with pytest.raises(ValueError, match='sn_deg'):
        Encoder(p).encode(s)
    s = make_vq_stream(p, 1, seed=15)[0]
    i = next(i for i, d in enumerate(s.sn_deg) if d == 3)
    s.sn_idx[i] = [99, 0]
    with pytest.raises(ValueError, match='sn_idx'):
        Encoder(p).encode(s)
    from tests.test_csgs import params
    s = make_stream(params(True), 1, seed=16)[0]
    s.sn_deg = [0] * len(s.ids)
    with pytest.raises(ValueError, match='need params.sn_vq'):
        Encoder(params(True)).encode(s)
    with pytest.raises(ValueError, match='sn_vq levels'):
        vq_params(levels=[(3, [8]), (1, [8])])
    with pytest.raises(ValueError, match='sn_vq levels'):
        Params(L=8, ks=[1], groups=GROUPS, n_sn=9, offsets=[0.0] * 23, scales=[1.0] * 23, sn_vq=[(3, [8])])


def test_vq_state_carries_codebooks_and_per_splat_assignment():
    p = vq_params(key_interval=10)
    secs = make_vq_stream(p, 2, seed=17)
    st0 = State.after(secs[0], p)
    assert st0.sn_cb is secs[0].sn_cb
    st1 = State.after(secs[1], p, st0)
    assert st1.sn_cb is secs[0].sn_cb
    cont = [i for i, dc in enumerate(secs[1].death) if dc == secs[1].nframes + 1]
    assert st1.sn_deg == [secs[1].sn_deg[i] for i in cont] and st1.sn_idx == [secs[1].sn_idx[i] for i in cont]
