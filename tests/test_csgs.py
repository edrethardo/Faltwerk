""".csgs container + rANS roundtrip tests (synthetic data; real P1 data when CSGS_CURVES/CSGS_TRACKS are set)."""
import os
import random

import pytest

from codec import rans
from codec.csgs import Decoder, Encoder, Params, Section, State, stream_sizes

GROUPS = [('P', 3), ('R', 4), ('S', 3), ('O', 1), ('D', 3)]
N_SN = 6


def params(shared=True, key_interval=3, L=8, ks=(1, 2, 3, 5)):
    nq = sum(d for _, d in GROUPS) + N_SN
    return Params(L=L, ks=list(ks), groups=GROUPS, n_sn=N_SN, offsets=[0.5 * i for i in range(nq)],
                  scales=[0.001 * (i + 1) for i in range(nq)], shared_knot=shared, key_interval=key_interval)


def make_stream(p, nsec, n0=40, seed=0, T=None, huge=True, sizes=None):
    """random but valid sections: lifetimes, continuing splats, K per group, knots (some huge), static SN."""
    rnd = random.Random(seed)
    T = T if T is not None else nsec * p.L
    live = []                  # (id, last knots per group, sn)
    next_id = 0
    out = []
    for si in range(nsec):
        g0 = si * p.L
        nf = min(p.L, T - g0)
        n_new = (n0 if si == 0 else rnd.randint(0, n0 // 2)) if sizes is None else sizes[si]
        ids, birth, death, sn = [], [], [], []
        k = [[] for _ in GROUPS]; knots = [[] for _ in GROUPS]
        rows = [(sid, last, s, True) for sid, last, s in live] + \
               [(next_id + i, None, [rnd.randint(0, 255) for _ in range(N_SN)], False) for i in range(n_new)]
        next_id += n_new
        new_live = []
        for sid, last, s, cont in rows:
            ids.append(sid); sn.append(s)
            b = 0 if cont else rnd.randrange(nf)
            if not cont:
                birth.append(b)
            dc = rnd.choice([nf + 1] * 3 + list(range(b + 1, nf + 2)))
            death.append(dc)
            lasts = []
            for g, (_, dm) in enumerate(GROUPS):
                K = rnd.choice(p.ks)
                if last is not None and p.shared_knot:
                    row = list(last[g])
                else:
                    base = last[g] if last is not None else [rnd.randint(-50, 2000) for _ in range(dm)]
                    row = [v + rnd.randint(-3, 3) for v in base]
                for j in range(1, K):
                    for c in range(dm):
                        r = rnd.random()
                        dv = rnd.randint(-2, 2) if r < 0.8 else rnd.randint(-300, 300) if r < 0.97 or not huge \
                            else rnd.choice([1, -1]) * rnd.randint(1 << 16, 1 << 22)
                        row.append(row[(j - 1) * dm + c] + dv)
                k[g].append(K); knots[g].append(row)
                lasts.append(row[-dm:])
            if dc == nf + 1:
                new_live.append((sid, lasts, s))
        live = new_live
        out.append(Section(index=si, g0=g0, nframes=nf, ids=ids, n_cont=len(rows) - n_new, birth=birth, death=death,
                           k=k, knots=knots, sn=sn))
    return out


def encode_all(p, secs):
    enc = Encoder(p)
    return [enc.encode(s) for s in secs]


# ------------------------------------------------------------------ rANS core

def test_rans_roundtrip_models_bits_and_escapes():
    rnd = random.Random(1)
    models = [rans.AdaptiveModel(n) for n in (1, 2, 7, rans.NTOK)]
    ops = []
    for _ in range(20000):
        r = rnd.random()
        if r < 0.4:
            mi = rnd.randrange(3)
            ops.append(('s', mi, rnd.randrange(models[mi].n) if rnd.random() < 0.3 else 0))
        elif r < 0.6:
            nb = rnd.randint(0, 40)
            ops.append(('b', nb, rnd.getrandbits(nb) if nb else 0))
        elif r < 0.8:
            ops.append(('u', 3, rnd.choice([0, 1, 15, 16, 17, 1000, (1 << 32) + rans.ESC - 2, rnd.getrandbits(31)])))
        else:
            ops.append(('i', 3, rnd.choice([0, -1, 1, -8, 8, -(1 << 31), (1 << 31) - 1, rnd.randint(-99, 99)])))
    e = rans.Encoder()
    for op, a, v in ops:
        {'s': lambda: e.put(models[a], v), 'b': lambda: e.put_bits(v, a), 'u': lambda: e.put_uint(models[a], v),
         'i': lambda: e.put_sint(models[a], v)}[op]()
    buf = e.finish()
    dm = [rans.AdaptiveModel(n) for n in (1, 2, 7, rans.NTOK)]
    d = rans.Decoder(buf)
    for op, a, v in ops:
        got = {'s': lambda: d.get(dm[a]), 'b': lambda: d.get_bits(a), 'u': lambda: d.get_uint(dm[a]),
               'i': lambda: d.get_sint(dm[a])}[op]()
        assert got == v, (op, a, v, got)
    assert d.done()


def test_rans_skewed_source_beats_8_bits():
    m = rans.AdaptiveModel(rans.NTOK)
    e = rans.Encoder()
    rnd = random.Random(2)
    vals = [0 if rnd.random() < 0.9 else rnd.randint(-3, 3) for _ in range(10000)]
    for v in vals:
        e.put_sint(m, v)
    buf = e.finish()
    assert len(buf) * 8 / len(vals) < 0.9          # ~0.75 bit/symbol entropy, adaptive overhead small
    d, m2 = rans.Decoder(buf), rans.AdaptiveModel(rans.NTOK)
    assert [d.get_sint(m2) for _ in vals] == vals and d.done()


def test_model_cdf_invariants_after_many_updates():
    m = rans.AdaptiveModel(rans.NTOK)
    rnd = random.Random(3)
    for _ in range(50000):
        m.update(0 if rnd.random() < 0.99 else rnd.randrange(m.n))
        assert m.total <= rans.COUNT_LIMIT
    assert m.cdf[0] == 0 and m.cdf[-1] == rans.PROB_SCALE
    assert all(m.cdf[i + 1] > m.cdf[i] for i in range(m.n))


def test_model_gives_rest_to_most_frequent_symbol():
    m = rans.AdaptiveModel(8)
    for _ in range(3000):
        m.update(5)
    f = [m.cdf[i + 1] - m.cdf[i] for i in range(8)]
    assert f[5] == max(f) and f[5] > rans.PROB_SCALE - 64 and sum(f) == rans.PROB_SCALE


def test_decoder_detects_unconsumed_or_extra_bytes():
    m = rans.AdaptiveModel(rans.NTOK)
    e = rans.Encoder()
    vals = list(range(-500, 500, 7))
    for v in vals:
        e.put_sint(m, v)
    buf = e.finish()
    d, m2 = rans.Decoder(buf), rans.AdaptiveModel(rans.NTOK)
    for v in vals[:-1]:
        assert d.get_sint(m2) == v
    assert not d.done()                              # one symbol left
    d, m2 = rans.Decoder(buf + b'\x00'), rans.AdaptiveModel(rans.NTOK)
    for v in vals:
        assert d.get_sint(m2) == v
    assert not d.done()                              # trailing byte


def test_format_is_frozen():
    """golden digest of a synthetic stream: any change of models, contexts or layout must bump VERSION."""
    import hashlib
    p = params(True, key_interval=2)
    h = hashlib.sha256(b''.join(encode_all(p, make_stream(p, 3, seed=21)))).hexdigest()
    assert h == GOLDEN, h


GOLDEN = 'c1542d9af5a18a6bc0822cd137e7f13c78dfd087732a0e40cad158457238e334'


def test_empty_encoder_is_empty():
    assert rans.Encoder().finish() == b''


# ------------------------------------------------------------------ container

@pytest.mark.parametrize('shared', [True, False])
def test_stream_roundtrip_bitexact(shared):
    p = params(shared)
    secs = make_stream(p, 7, seed=4 + shared)
    bufs = encode_all(p, secs)
    dec = Decoder()
    for s, b in zip(secs, bufs):
        out = dec.decode(b)
        assert out == s, out.first_difference(s)
        assert out.key == (s.index % p.key_interval == 0)
        assert dec.params == p
    assert encode_all(p, secs) == bufs          # deterministic


def test_random_access_from_key_section():
    p = params(True, key_interval=3)
    secs = make_stream(p, 8, seed=7)
    bufs = encode_all(p, secs)
    for start in (3, 6):
        dec = Decoder()
        for s, b in zip(secs[start:], bufs[start:]):
            out = dec.decode(b)
            assert out == s, out.first_difference(s)
    with pytest.raises(ValueError, match='key section'):
        Decoder().decode(bufs[4])
    dec = Decoder()
    dec.decode(bufs[0])
    with pytest.raises(ValueError, match='key section'):
        dec.decode(bufs[2])                      # skipped section 1


def test_key_section_is_bigger_than_its_delta_version():
    p = params(True, key_interval=100)
    secs = make_stream(p, 3, seed=8)
    delta = encode_all(p, secs)[2]
    enc = Encoder(p)
    enc.encode(secs[0]); enc.encode(secs[1])
    key = enc.encode(secs[2], key=True)
    assert len(key) > len(delta)
    assert stream_sizes(key)['ids'] > 0 and stream_sizes(delta)['ids'] == 0
    assert Decoder().decode(key) == secs[2]


def test_empty_and_short_sections():
    p = params(True, key_interval=4, L=8)
    # sizes: 0 splats in section 0, then newborns, then a section with no newborns, last section 3 frames long
    secs = make_stream(p, 5, seed=9, T=4 * 8 + 3, sizes=[0, 5, 0, 0, 2])
    assert len(secs[0].ids) == 0 and secs[-1].nframes == 3
    bufs = encode_all(p, secs)
    dec = Decoder()
    for s, b in zip(secs, bufs):
        assert dec.decode(b) == s
    sz = stream_sizes(bufs[0])
    assert all(sz[k] == 0 for k in sz if k != 'header')
    one = make_stream(params(True, L=1), 3, seed=10, n0=5)          # one-frame sections
    dec = Decoder()
    for s, b in zip(one, encode_all(params(True, L=1), one)):
        assert dec.decode(b) == s


def test_corruption_is_detected():
    p = params(True)
    secs = make_stream(p, 1, seed=11)
    b = bytearray(encode_all(p, secs)[0])
    b[len(b) // 2] ^= 0x10
    with pytest.raises(ValueError, match='CRC'):
        Decoder().decode(bytes(b))


def test_encoder_rejects_inconsistent_input():
    p = params(True, key_interval=10)
    secs = make_stream(p, 2, seed=12, huge=False)
    enc = Encoder(p)
    enc.encode(secs[0])
    bad = make_stream(p, 2, seed=12, huge=False)[1]
    assert bad.n_cont > 0
    bad.knots[0][0][0] += 1                       # breaks the shared boundary knot
    with pytest.raises(ValueError, match='shared knot'):
        enc.encode(bad)
    with pytest.raises(ValueError, match='K not in'):
        s = make_stream(p, 1, seed=13)[0]; s.k[1][0] = 4; Encoder(p).encode(s)


def test_state_carries_only_continuing_splats():
    p = params(True)
    s = make_stream(p, 1, seed=14)[0]
    st = State.after(s, p)
    assert st.ids == [sid for sid, dc in zip(s.ids, s.death) if dc == s.nframes + 1]


# ------------------------------------------------------------------ real P1 data (optional, slow)

@pytest.mark.skipif(not (os.environ.get('CSGS_CURVES') and os.environ.get('CSGS_TRACKS')), reason='no P1 data')
def test_real_p1_curves_first_section_roundtrip():
    from codec.prep import sections_from_curves
    p, secs, _ = sections_from_curves(os.environ['CSGS_CURVES'], os.environ['CSGS_TRACKS'], max_sections=1)
    b = Encoder(p).encode(secs[0])
    assert Decoder().decode(b) == secs[0]
