#!/usr/bin/env python3
"""reference dump of the Python decoder (codec/falt.py) for the TypeScript port tests.

usage: python web/bench/dump_ref.py <out_dir> s0000.csgs [s0001.csgs ...]

Per section writes <out_dir>/<name>.ref.json (header, shapes, CRC32 of every decoded array) and <name>.ref.bin
(all decoded integers, int32 little-endian, in the order listed in the JSON). Additionally `curve` holds the
dequantised, curve-evaluated attributes (codec/web convention: P, R nlerp, S log, O logit, D sh0) of a fixed sample
of splats at a few fractional frame times, computed independently with numpy (the oracle for web/src/pack.ts).
"""
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from codec.falt import Decoder  # noqa: E402


def crc(a):
    return zlib.crc32(np.ascontiguousarray(a, dtype='<i4').tobytes()) & 0xffffffff


def eval_curve(row, K, d, nframes, g0, t, bspline=False):
    C = np.asarray(row, np.float64).reshape(K, d)
    if K == 1:
        return C[0]
    if bspline:                                   # v6: clamped cubic B-spline (codec.falt.bspline_weights)
        from codec.falt import bspline_weights
        w = bspline_weights(K, min(max((t - g0) / max(nframes - 1, 1), 0.0), 1.0))
        return np.asarray(w) @ C
    tau = (t - g0) * (K - 1) / max(nframes - 1, 1)
    tau = min(max(tau, 0.0), K - 1)
    j = min(int(np.floor(tau)), K - 2)
    w = tau - j
    return (1 - w) * C[j] + w * C[j + 1]


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    dec = Decoder()
    for path in sys.argv[2:]:
        buf = open(path, 'rb').read()
        t0 = time.time()
        sec = dec.decode(buf)
        dt = time.time() - t0
        p = dec.params
        n = len(sec.ids)
        arrays = [('ids', np.array(sec.ids, np.int64).astype('<u4').view('<i4')), ('birth', np.array(sec.birth)),
                  ('death', np.array(sec.death))]
        for g, (tag, dm) in enumerate(p.groups):
            arrays.append((f'k_{tag}', np.array(sec.k[g])))
            arrays.append((f'knots_{tag}', np.concatenate([np.array(r, np.int64) for r in sec.knots[g]])))
        arrays.append(('sn', np.array(sec.sn, np.int64).reshape(-1)))
        if p.sn_vq is not None:      # v2: degree per splat, indices flat (stages of the splat's level), key codebooks flat
            arrays.append(('sn_deg', np.array(sec.sn_deg, np.int64)))
            arrays.append(('sn_idx', np.array([x for row in sec.sn_idx for x in row], np.int64)))
            if sec.sn_cb is not None:
                arrays.append(('sn_cb', np.array([v for lvl in sec.sn_cb for st in lvl for e in st for v in e], np.int64)))
        name = os.path.basename(path).rsplit('.', 1)[0]
        meta = dict(file=os.path.basename(path), bytes=len(buf), index=sec.index, g0=sec.g0, nframes=sec.nframes,
                    key=sec.key, n=n, n_cont=sec.n_cont, groups=p.groups, ks=p.ks, n_sn=p.n_sn, L=p.L,
                    sn_vq=p.sn_vq, python_decode_s=round(dt, 2), arrays=[])
        with open(os.path.join(out, name + '.ref.bin'), 'wb') as f:
            for nm, a in arrays:
                a = np.ascontiguousarray(a.astype('<i4'))
                meta['arrays'].append(dict(name=nm, len=int(a.size), crc32=crc(a)))
                f.write(a.tobytes())
        # curve oracle: dequantised attributes of 64 sample splats at fractional times
        rng = np.random.default_rng(1267)
        sample = sorted(rng.choice(n, size=min(int(os.environ.get('REF_SAMPLE', '64')), n), replace=False).tolist())
        times = [sec.g0 + x for x in (0.0, 0.37, 7.5, sec.nframes - 1.25, sec.nframes - 0.5)]
        off, sc = p.offsets, p.scales
        curve = []
        for i in sample:
            q0 = 0
            vals = {}
            for g, (tag, dm) in enumerate(p.groups):
                K = sec.k[g][i]
                row = np.asarray(sec.knots[g][i], np.float64).reshape(K, dm)
                deq = np.asarray(off[q0:q0 + dm]) + row * np.asarray(sc[q0:q0 + dm])
                q0 += dm
                vals[tag] = [eval_curve(deq.reshape(-1), K, dm, sec.nframes, sec.g0, t, p.bspline).tolist() for t in times]
                if tag == 'R':
                    vals[tag] = [(np.array(v) / max(np.linalg.norm(v), 1e-12)).tolist() for v in vals[tag]]
            sn = (np.asarray(off[q0:]) + np.asarray(sec.sn[i], np.float64) * np.asarray(sc[q0:])).tolist()
            curve.append(dict(i=i, vals=vals, sn=sn))
        meta['curve'] = dict(times=times, splats=curve)
        json.dump(meta, open(os.path.join(out, name + '.ref.json'), 'w'))
        print(f'{name}: n={n} n_cont={sec.n_cont} key={sec.key} decode {dt:.1f}s', flush=True)


if __name__ == '__main__':
    main()
