"""(S7): per-section quantisation table (flag RESCALE, version 5): the previous section's last knots move to the
new table (shared knot / residual predictor / end velocity), bit-exact roundtrip, golden bytes of tables-without-change."""
import hashlib

import pytest

from codec.csgs import (Decoder, Encoder, FLAG_RESCALE, Params, VERSION_RESCALE, needs_rescale, rescale_value,
                        stream_sizes, _unpack_header)
from test_csgs import GROUPS, N_SN, make_stream, params


def scaled(p, f_pos=1.0, f_attr=1.0, pred=None):
    """same stream parameters, other knot table: position scale x f_pos, attribute scales x f_attr, offsets shifted a bit."""
    off, sc = list(p.offsets), list(p.scales)
    n = sum(d for _, d in GROUPS)
    for i in range(n):
        sc[i] = sc[i] * (f_pos if i < 3 else f_attr)
        off[i] = off[i] + (0.0003 * i if f_attr != 1.0 or f_pos != 1.0 else 0.0)
    return Params(p.L, p.ks, p.groups, p.n_sn, off, sc, p.shared_knot, p.key_interval, pred=pred,
                  knn_k=p.knn_k)


def chain_shared(secs, plist):
    """make a shared-knot stream consistent with per-section tables: knot 0 of every continuing splat = the previous
    section's last knot moved to this section's table (what the container requires)."""
    prev = None
    for s, p in zip(secs, plist):
        if prev is not None and s.index % p.key_interval:
            sp, last = prev
            for g, (_, d) in enumerate(GROUPS):
                q0 = sum(dd for _, dd in GROUPS[:g])
                for i in range(s.n_cont):
                    cur = last[g][i]
                    new = [rescale_value(cur[c], sp.offsets[q0 + c], sp.scales[q0 + c], p.offsets[q0 + c], p.scales[q0 + c])
                           for c in range(d)]
                    s.knots[g][i][:d] = new
        cont = [i for i, dc in enumerate(s.death) if dc == s.nframes + 1]
        prev = (p, [[s.knots[g][i][-d:] for i in cont] for g, (_, d) in enumerate(GROUPS)])
    return secs


def run(secs, plist):
    enc, bufs = Encoder(plist[0]), []
    for s, p in zip(secs, plist):
        enc.params = p
        bufs.append(enc.encode(s))
    dec = Decoder()
    out = [dec.decode(b) for b in bufs]
    return bufs, out


@pytest.mark.parametrize('shared', [True, False])
def test_rescale_roundtrip_and_flags(shared):
    p = params(shared, key_interval=4)
    secs = make_stream(p, 6, seed=31, huge=False)
    tables = [p, scaled(p, 0.5, 1.0), scaled(p, 0.5, 1.0), scaled(p, 2.0, 0.75), p, scaled(p, 1.0, 2.0)]
    tables = [Params(t.L, t.ks, t.groups, t.n_sn, t.offsets, t.scales, shared, 4) for t in tables]
    if shared:
        secs = chain_shared(secs, tables)
    bufs, out = run(secs, tables)
    assert out == secs
    flags = [b[5] & FLAG_RESCALE for b in bufs]
    # section 0 key; 1 changes (flag), 2 same as 1 (no flag), 3 changes, 4 is a KEY section (no reference, no flag), 5 changes
    assert [bool(f) for f in flags] == [False, True, False, True, False, True]
    assert [b[4] for b in bufs] == [VERSION_RESCALE if f else 1 for f in flags]
    assert needs_rescale(tables[0], tables[1]) and not needs_rescale(tables[1], tables[2])


def test_rescale_identity_is_exact_when_scale_is_a_power_of_two_and_offset_equal():
    assert rescale_value(1234, 0.5, 0.001, 0.5, 0.001) == 1234
    assert rescale_value(-77, 0.25, 0.125, 0.25, 0.25) == -38 or rescale_value(-77, 0.25, 0.125, 0.25, 0.25) == -39
    assert rescale_value(40, 0.0, 0.5, 0.0, 0.25) == 80


def test_rescale_with_velocity_prediction():
    p = Params(8, [1, 2, 3, 5], GROUPS, N_SN, params().offsets, params().scales, False, 4, pred=[4] * 5)
    secs = make_stream(p, 4, seed=33, huge=False)
    tables = [p] + [Params(8, [1, 2, 3, 5], GROUPS, N_SN, scaled(p, 1.5, 1.0).offsets, scaled(p, 1.5, 1.0).scales, False, 4,
                           pred=[4] * 5) for _ in range(3)]
    bufs, out = run(secs, tables)
    assert out == secs


def test_sn_table_must_not_change_inside_a_key_interval():
    p = params(False, key_interval=4)
    secs = make_stream(p, 2, seed=34, huge=False)
    bad = list(p.scales); bad[-1] *= 2
    p2 = Params(p.L, p.ks, p.groups, p.n_sn, p.offsets, bad, False, 4)
    enc = Encoder(p)
    enc.encode(secs[0])
    enc.params = p2
    with pytest.raises(ValueError):
        enc.encode(secs[1])


def test_without_table_change_streams_are_byte_identical_to_the_plain_encoder():
    p = params(True, key_interval=3)
    secs = make_stream(p, 4, seed=35)
    a = [Encoder(p).encode(secs[0])]
    enc = Encoder(p)
    plain = [enc.encode(s) for s in secs]
    enc2 = Encoder(p)
    same = []
    for s in secs:
        enc2.params = Params(p.L, p.ks, p.groups, p.n_sn, list(p.offsets), list(p.scales), p.shared_knot, p.key_interval)
        same.append(enc2.encode(s))
    assert plain == same and all(b[4] == 1 for b in plain)


def test_decoder_rejects_flag_table_disagreement():
    p = params(False, key_interval=4)
    secs = make_stream(p, 2, seed=36, huge=False)
    tables = [p, scaled(p, 0.5, 1.0)]
    tables = [Params(t.L, t.ks, t.groups, t.n_sn, t.offsets, t.scales, False, 4) for t in tables]
    bufs, _ = run(secs, tables)
    # decoder that saw section 0 under table p, but is handed a section 1 without the flag (patch version+flag+crc)
    import struct
    import zlib
    b = bytearray(bufs[1][:-4])
    b[4] = 1; b[5] &= ~FLAG_RESCALE
    b = bytes(b) + struct.pack('<I', zlib.crc32(bytes(b)) & 0xffffffff)
    dec = Decoder()
    dec.decode(bufs[0])
    with pytest.raises(ValueError):
        dec.decode(b)
