"""x integration: SN-VQ (flag 4, v2) and motion prediction (flag 8, v3) are independent and combinable;
the version is the highest one a set flag needs."""
import copy

import pytest

from codec.csgs import (FLAG_PRED, FLAG_SN_VQ, PRED_SPATIAL, PRED_TEMPORAL, PRED_VEL0, VERSION, VERSION_PRED,
                        VERSION_SN_VQ, Decoder, Encoder, Params, stream_sizes)
from tests.test_csgs import GROUPS, make_stream
from tests.test_csgs_snvq import LEVELS, N_SN, make_vq_stream, vq_params

MODES = [PRED_TEMPORAL | PRED_SPATIAL, PRED_TEMPORAL, PRED_SPATIAL | PRED_VEL0, PRED_TEMPORAL, PRED_VEL0]


def combo_params(vq, pred, knn_k=3, shared=True):
    p = vq_params(shared=shared)
    return Params(p.L, p.ks, p.groups, p.n_sn, p.offsets, p.scales, p.shared_knot, p.key_interval,
                  sn_vq=LEVELS if vq else None, sn_zero=p.sn_zero if vq else None,
                  pred=MODES if pred else None, knn_k=knn_k if pred else 0)


def plain(s):
    """the same section without the SN-VQ fields (its sn rows are valid raw SN rows)"""
    c = copy.copy(s)
    c.sn_deg = c.sn_idx = c.sn_cb = None
    return c


def encode(p, secs):
    enc = Encoder(p)
    return [enc.encode(s) for s in secs]


@pytest.mark.parametrize('shared', [True, False])
def test_v2_plus_v3_roundtrip_bitexact(shared):
    p = combo_params(True, True, shared=shared)
    secs = make_vq_stream(p, 7, seed=91 + shared, n0=60)
    assert any(s.n_cont and s.index % p.key_interval for s in secs)
    bufs = encode(p, secs)
    dec = Decoder()
    for s, b in zip(secs, bufs):
        assert b[4] == VERSION_PRED and b[5] & FLAG_PRED and b[5] & FLAG_SN_VQ
        out = dec.decode(b)
        assert out == s, out.first_difference(s)
        assert dec.params == p and dec.params.sn_vq == LEVELS and dec.params.pred == MODES
    assert encode(p, secs) == bufs                                   # deterministic


def test_version_follows_the_highest_needed_flag_and_streams_are_independent():
    base = combo_params(True, False)
    secs = make_vq_stream(base, 4, seed=5, n0=50)
    res = {}
    for name, (vq, pred) in {'v1': (False, False), 'v2': (True, False), 'v3': (False, True), 'v2v3': (True, True)}.items():
        p = combo_params(vq, pred)
        use = secs if vq else [plain(s) for s in secs]
        bufs = encode(p, use)
        res[name] = [stream_sizes(b) for b in bufs]
        want = VERSION_PRED if pred else VERSION_SN_VQ if vq else VERSION
        assert all(b[4] == want for b in bufs)
        assert all(bool(b[5] & FLAG_SN_VQ) == vq and bool(b[5] & FLAG_PRED) == pred for b in bufs)
        dec = Decoder()
        assert all(dec.decode(b) == s for b, s in zip(bufs, use))
    for a, b in zip(res['v2'], res['v2v3']):                       # prediction leaves the SN streams alone ...
        assert all(a[k] == b[k] for k in ('sn', 'sn_cb', 'sn_deg', 'sn_idx', 'birth', 'death', 'ids', 'k'))
    assert set(res['v2v3'][0]) >= {'sn_cb', 'sn_deg', 'sn_idx', 'knots_P'}


def test_random_access_key_section_with_both_flags():
    p = combo_params(True, True)
    secs = make_vq_stream(p, 8, seed=13, n0=60)
    bufs = encode(p, secs)
    dec = Decoder()
    for s, b in zip(secs[3:], bufs[3:]):
        assert dec.decode(b) == s


def test_all_three_flags_combined_roundtrip_and_independent_streams():
    """SN_VQ (4) + pred (8) + follow (16): version 4, bit-exact, SN/knot streams unchanged by the other modes."""
    from codec.csgs import FLAG_FOLLOW, VERSION_FOLLOW
    from test_csgs_follow import fix_shared, with_follow
    p = combo_params(True, True)
    base = make_vq_stream(p, 5, seed=44, n0=70)
    secs = with_follow(p, base)
    for o, s in zip(secs, base):
        o.sn_deg, o.sn_idx, o.sn_cb = s.sn_deg, s.sn_idx, s.sn_cb
    secs = fix_shared(p, secs)
    assert any(s.follow for s in secs)
    bufs = encode(p, secs)
    dec = Decoder()
    for s, b in zip(secs, bufs):
        out = dec.decode(b)
        assert out == s, out.first_difference(s)
        if s.follow:
            assert b[4] == VERSION_FOLLOW and b[5] & FLAG_FOLLOW and b[5] & FLAG_SN_VQ and b[5] & FLAG_PRED
            assert 'follow' in stream_sizes(b)
    # follow without pred / without vq also works (version 4 either way)
    for vq, pred in ((False, False), (True, False), (False, True)):
        q = combo_params(vq, pred)
        use = [plain(s) for s in secs] if not vq else secs
        d2 = Decoder()
        for s, b in zip(use, encode(q, use)):
            assert d2.decode(b) == s


def test_version_flag_mismatch_is_rejected():
    p = combo_params(True, True)
    b = bytearray(encode(p, make_vq_stream(p, 1, seed=1, n0=20))[0])
    for ver in (VERSION, VERSION_SN_VQ):
        b[4] = ver
        with pytest.raises(ValueError, match='disagree'):
            Decoder().decode(bytes(b))
