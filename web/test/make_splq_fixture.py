#!/usr/bin/env python3
"""synthetic .splq fixture for the TypeScript tests (random content, no scene data).

usage: /usr/bin/python3 web/test/make_splq_fixture.py web/test/fixtures
Packs random plyq-style integers with web/bench/make_viewer_data.pack_splq, once plain and once under a similarity
(rotation x scale + translation like the MLSharp registration), and stores the expected world attributes computed
independently in float64 with experiments/james/p8_lib.apply_similarity (the oracle eval_all.py uses).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'bench'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'experiments', 'james'))
import make_viewer_data as V  # noqa: E402
import p8_lib as L  # noqa: E402


def main(out):
    rng = np.random.default_rng(1274)
    n = 64
    q = {'P': rng.integers(0, 1024, (n, 3)), 'R': rng.integers(-127, 128, (n, 4)), 'S': rng.integers(0, 256, (n, 3)),
         'O': rng.integers(0, 256, n), 'D': rng.integers(0, 256, (n, 3))}
    q['R'][0] = [127, 0, 0, 0]
    q['P'][1] = [1023, 1023, 1023]
    off = [float(np.float32(x)) for x in [-1.1, 0.02, -0.9] + [0.0] * 4 + [-9.0, -8.5, -8.0, -6.0, -1.7, -1.5, -1.2]]
    sc = [float(np.float32(x)) for x in [0.0021] * 3 + [1 / 127] * 4 + [0.03, 0.031, 0.029, 0.05, 0.012, 0.013, 0.011]]
    f32 = lambda a: a.astype(np.float32).astype(np.float64)  # noqa: E731
    deq = {k: np.asarray(off[o:o + d]) + q[k].reshape(n, d) * np.asarray(sc[o:o + d])
           for k, o, d in (('P', 0, 3), ('R', 3, 4), ('S', 7, 3), ('O', 10, 1), ('D', 11, 3))}
    th = 0.7
    Rm = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]]) @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])          # 180 deg about x then yaw (like MLSharp: y,z flipped)
    M = np.eye(4); M[:3, :3] = 0.88 * Rm; M[:3, 3] = [0.25, 1.65, 2.05]
    cases = {}
    for name, sim in (('plain', None), ('sim', M)):
        b = V.pack_splq(q, off, sc, V.similarity(M) if sim is not None else None)
        open(os.path.join(out, f'synth_{name}.splq'), 'wb').write(b)
        p = {'means': f32(deq['P']), 'quats': f32(deq['R']), 'scales': f32(deq['S']), 'opacities': f32(deq['O'][:, 0]),
             'sh0': f32(deq['D'])[:, None]}
        p['quats'] = p['quats'] / np.linalg.norm(p['quats'], axis=1, keepdims=True)
        if sim is not None:
            p = L.apply_similarity(p, sim)
        cases[name] = {'P': np.asarray(p['means'], float).tolist(), 'R': np.asarray(p['quats'], float).tolist(),
                       'S': np.asarray(p['scales'], float).tolist(), 'O': np.asarray(p['opacities'], float).tolist(),
                       'D': np.asarray(p['sh0'], float)[:, 0].tolist()}
    json.dump({'n': n, 'cases': cases}, open(os.path.join(out, 'synth_splq.ref.json'), 'w'))
    # 16-bit positions + SH bands (degree 2 = 24 SN scalars), no similarity
    run = rng.integers(0, 65536, (n, 3)); run[1] = [65535, 65535, 65535]
    sn = rng.integers(0, 256, (n, 24))
    so = [float(np.float32(-0.3 + 0.01 * k)) for k in range(24)]
    ss = [float(np.float32(0.002 + 0.0001 * k)) for k in range(24)]
    off16 = list(off); sc16 = list(sc); sc16[:3] = [float(np.float32(x)) for x in (3.1e-5, 1.7e-5, 2.2e-5)]
    b = V.pack_splq(q, off16, sc16, sn=(sn, so, ss), run=run)
    open(os.path.join(out, 'synth_p16sn.splq'), 'wb').write(b)
    P = np.asarray(off16[:3]) + run * np.asarray(sc16[:3])
    SN = np.asarray(so) + sn * np.asarray(ss)
    json.dump({'n': n, 'P': f32(P).tolist(), 'SN': f32(SN).tolist(), 'm': 24},
              open(os.path.join(out, 'synth_p16sn.ref.json'), 'w'))
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1])
