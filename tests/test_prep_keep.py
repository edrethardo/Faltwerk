"""(the contributor' PR #1 on top of S1-S7/P16): codec.prep keep (matte_cull) and sn_deg/sn_bits combined with the
master paths (plan tables, Morton order, SN-VQ guard)."""
import numpy as np
import pytest

from codec import prep
from codec.csgs import Decoder, Encoder
from test_prep_s2 import records, synth


def _roundtrip(p, secs):
    enc, dec = Encoder(p), Decoder()
    for s in secs:
        assert dec.decode(enc.encode(s)) == s


def test_keep_all_true_is_identical_to_no_keep(tmp_path):
    cp, tp = synth(tmp_path)
    p0, s0, _ = prep.sections_from_curves(cp, tp, qpos=0.01, quat='raw', tol=1)
    keep = np.ones(17, bool)
    p1, s1, st = prep.sections_from_curves(cp, tp, qpos=0.01, quat='raw', tol=1, keep=keep)
    assert p0 == p1 and s0 == s1 and st['keep_kept'] == 17


@pytest.mark.parametrize('order', ['id', 'morton'])
def test_keep_drops_splats_and_renumbers_densely(tmp_path, order):
    cp, tp = synth(tmp_path, seed=1)
    drop = {1, 5, 13}                                       # two of section 0 (one continues), one newborn of section 1
    keep = np.array([i not in drop for i in range(17)])
    newid = np.cumsum(keep) - 1
    p0, s0, _ = prep.sections_from_curves(cp, tp, qpos=0.01, quat='raw', tol=1)
    p1, s1, _ = prep.sections_from_curves(cp, tp, qpos=0.01, quat='raw', tol=1, keep=keep, newborn_order=order)
    assert p0 == p1                                         # quantiser ranges / position grid ignore keep
    for a, b in zip(s0, s1):
        assert b.ids == sorted(b.ids) and len(b.ids) == len([i for i in a.ids if i not in drop])
        nn = len(b.ids) - b.n_cont
        if nn:
            assert b.ids[b.n_cont:] == [b.ids[b.n_cont] + i for i in range(nn)]
        if order == 'id':
            assert set(b.ids) == {int(newid[i]) for i in a.ids if i not in drop}
        nc = a.n_cont                                       # same per-splat content as the full stream minus the dropped
        full = [(a.birth[i - nc] if i >= nc else -1, a.death[i], tuple(a.k[g][i] for g in range(len(p0.groups))),
                 tuple(tuple(a.knots[g][i]) for g in range(len(p0.groups))), tuple(a.sn[i]))
                for i in range(len(a.ids)) if a.ids[i] not in drop]
        assert sorted(full) == records(b, p1)
    assert set(s1[0].ids) == set(range(len(s1[0].ids)))
    _roundtrip(p1, s1)


def test_sn_deg_and_bits(tmp_path):
    cp, tp = synth(tmp_path, seed=2)
    p0, s0, _ = prep.sections_from_curves(cp, tp, qpos=0.01, quat='raw', tol=1)
    p8, s8, _ = prep.sections_from_curves(cp, tp, qpos=0.01, quat='raw', tol=1, sn_deg=3, sn_bits=8)
    assert p0 == p8 and s0 == s8                            # defaults = the earlier streams
    p5, s5, _ = prep.sections_from_curves(cp, tp, qpos=0.01, quat='raw', tol=1, sn_bits=5)
    assert max(max(r) for s in s5 for r in s.sn) <= 31
    assert np.allclose(np.asarray(p5.scales[-9:]) * 31, np.asarray(p0.scales[-9:]) * 255)
    pz, sz, _ = prep.sections_from_curves(cp, tp, qpos=0.01, quat='raw', tol=1, sn_deg=0)
    assert all(v == 0 for s in sz for r in s.sn for v in r) and all(v == 0 for v in pz.scales[-9:])
    _roundtrip(p5, s5); _roundtrip(pz, sz)


def test_sn_bits_with_sn_vq_is_rejected(tmp_path):
    cp, tp = synth(tmp_path)
    with pytest.raises(ValueError):
        prep.sections_from_curves(cp, tp, qpos=0.01, sn_bits=6, sn_vq={'levels': [(1, [16])]})
