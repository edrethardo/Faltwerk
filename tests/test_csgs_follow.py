"""(P8b): .csgs v2 'follow' mode — dormant (occluded) splats whose position knots are not sent but predicted by the
decoder from the motion of their nearest explicit neighbours (codec/csgs.py apply_follow, docs/format-csgs.md)."""
import random

import pytest

from codec.csgs import (FLAG_FOLLOW, FOLLOW_K, VERSION_FOLLOW, Decoder, Encoder, Section, State, apply_follow, stream_sizes)
from test_csgs import GROUPS, encode_all, make_stream, params


def with_follow(p, secs, seed=1, frac=0.3):
    """random follow masks on group P (never knot 0, only K >= 2), masked knots scrambled, then canonicalised."""
    rnd = random.Random(seed)
    out = []
    for s in secs:
        n = len(s.ids)
        masks = []
        for i in range(n):
            K = s.k[0][i]
            m = 0
            if K >= 2 and rnd.random() < frac:
                for j in range(1, K):
                    if rnd.random() < 0.6:
                        m |= 1 << j
            masks.append(m)
        rows = [list(r) for r in s.knots[0]]
        for i, m in enumerate(masks):
            for j in range(1, s.k[0][i]):
                if m >> j & 1:
                    for c in range(3):
                        rows[i][j * 3 + c] = rnd.randint(-10 ** 6, 10 ** 6)     # whatever: prediction overrides
        sec = Section(index=s.index, g0=s.g0, nframes=s.nframes, ids=s.ids, n_cont=s.n_cont, birth=s.birth, death=s.death,
                      k=s.k, knots=[rows] + s.knots[1:], sn=s.sn, follow=masks)
        out.append(apply_follow(p, sec))
    return out


def fix_shared(p, secs):
    """after canonicalisation the shared knot of continuing splats must still equal the previous (final) last knot."""
    prev = None
    fixed = []
    for s in secs:
        if prev is not None and p.shared_knot:
            st = State.after(prev, p)
            for g, (_, d) in enumerate(GROUPS):
                for i in range(s.n_cont):
                    s.knots[g][i][:d] = st.last[g][i]
            s = apply_follow(p, s)
        fixed.append(s); prev = s
    return fixed


@pytest.mark.parametrize('shared', [True, False])
def test_follow_roundtrip_bitexact(shared):
    p = params(shared, key_interval=3, ks=(1, 2, 3, 5, 9))
    secs = fix_shared(p, with_follow(p, make_stream(p, 4, seed=5, huge=False)))
    assert any(any(s.follow) for s in secs)
    bufs = encode_all(p, secs)
    assert all(b[4] == VERSION_FOLLOW and b[5] & FLAG_FOLLOW for b in bufs)       # version byte: follow mode
    dec = Decoder()
    for s, b in zip(secs, bufs):
        assert dec.decode(b) == s
        assert 'follow' in stream_sizes(b)


def test_no_follow_stays_version_1():
    p = params(True)
    bufs = encode_all(p, make_stream(p, 2, seed=3))
    assert all(b[4] == 1 for b in bufs) and 'follow' not in stream_sizes(bufs[0])


def test_prediction_is_neighbour_mean_displacement():
    """4 explicit neighbours move by (+8, -4, 0) per knot; the follower holds its anchor and moves the same way."""
    p = params(False, ks=(1, 2, 3, 5))
    K = 5
    nb = [[x, y, 0] for x, y in ((0, 0), (10, 0), (0, 10), (10, 10))]
    rows = [[v for j in range(K) for v in (b[0] + 8 * j, b[1] - 4 * j, b[2])] for b in nb]
    far = [5000, 5000, 5000]
    rows.append([v for j in range(K) for v in (far[0] - 100 * j, far[1], far[2])])        # far away: not a neighbour
    rows.append([5, 5, 0, 99, 99, 99, 77, 77, 77, 13, 21, 0, 0, 0, 0])                     # follower, knots 1,2 masked
    n = len(rows)
    sec = Section(index=0, g0=0, nframes=8, ids=list(range(n)), n_cont=0, birth=[0] * n, death=[9] * n,
                  k=[[K] * n] + [[1] * n for _ in GROUPS[1:]],
                  knots=[rows] + [[[0] * d for _ in range(n)] for _, d in GROUPS[1:]], sn=[[0] * 6 for _ in range(n)],
                  follow=[0] * (n - 1) + [0b110])
    out = apply_follow(p, sec)
    assert FOLLOW_K == 4
    assert out.knots[0][-1][:15] == [5, 5, 0, 13, 1, 0, 21, -3, 0, 13, 21, 0, 0, 0, 0]
    dec = Decoder()
    assert dec.decode(Encoder(p).encode(out)) == out


def test_prediction_rounds_to_nearest():
    """mean displacement 0.75 steps rounds to 1 (floor((S + D*n/2) / (D*n))), -0.75 to -1."""
    p = params(False, ks=(1, 2))
    rows = [[0, 0, 0, d, -d, 0] for d in (1, 1, 1, 0)] + [[1, 1, 0, 50, 50, 50]]
    n = len(rows)
    sec = Section(index=0, g0=0, nframes=4, ids=list(range(n)), n_cont=0, birth=[0] * n, death=[5] * n,
                  k=[[2] * n] + [[1] * n for _ in GROUPS[1:]],
                  knots=[rows] + [[[0] * d for _ in range(n)] for _, d in GROUPS[1:]], sn=[[0] * 6 for _ in range(n)],
                  follow=[0] * (n - 1) + [0b10])
    assert apply_follow(p, sec).knots[0][-1] == [1, 1, 0, 2, 0, 0]


def test_follow_rejects_bad_masks():
    p = params(True, ks=(1, 2, 3, 5))
    s = make_stream(p, 1, seed=2)[0]
    n = len(s.ids)
    i = next(i for i in range(n) if s.k[0][i] >= 2)
    m = [0] * n; m[i] = 1                                                # knot 0 may not be predicted
    bad = Section(index=0, g0=0, nframes=s.nframes, ids=s.ids, n_cont=s.n_cont, birth=s.birth, death=s.death,
                  k=s.k, knots=s.knots, sn=s.sn, follow=m)
    with pytest.raises(ValueError):
        Encoder(p).encode(bad)
    m[i] = 2                                                              # not canonical (knot 1 not the prediction)
    bad.follow = m
    bad.knots[0][i][3:6] = [10 ** 7, 10 ** 7, 10 ** 7]
    with pytest.raises(ValueError, match='prediction'):
        Encoder(p).encode(bad)


def test_followers_are_cheaper_than_sending_their_knots():
    p = params(False, key_interval=1, ks=(1, 2, 3, 5, 9))
    base = make_stream(p, 1, n0=300, seed=9, huge=False)[0]
    for i in range(len(base.ids)):                                       # moving splats with real knot data
        base.k[0][i] = 9
        b = base.knots[0][i][:3]
        base.knots[0][i] = [v + random.Random(i * 9 + j).randint(-40, 40) for j in range(9) for v in b]
    plain = Encoder(p).encode(base)
    f = with_follow(p, [base], seed=4, frac=0.5)[0]
    assert len(Encoder(p).encode(f)) < len(plain)
