#!/usr/bin/env python3
"""Deepen invisible hat-basis opacity stretches without changing visible fades.

Usage: deep_floor.py curves.npz out.npz [--deep -12]
"""
import json
import sys
import numpy as np

src, dst = sys.argv[1:3]
DEEP = float(sys.argv[sys.argv.index('--deep') + 1]) if '--deep' in sys.argv else -12.0
CUT = float(np.log(1 / 255 / (1 - 1 / 255)))
z = np.load(src)
assert str(z['basis']) == 'hat', 'deep_floor: only for the hat basis'
out = {k: z[k] for k in z.files}
stats = {'cut_logit': CUT, 'deep_logit': DEEP, 'knots': 0, 'below': 0, 'deepened': 0}
for k in z.files:
    if not (k.startswith('g') and k.endswith('_opacities')) or z[k].ndim != 2:
        continue
    O = z[k].astype(np.float32)
    b = O < CUT
    pad = np.pad(b, ((0, 0), (1, 1)), constant_values=True)
    ok = b & pad[:, :-2] & pad[:, 2:]
    if not bool(z['soft_knot0']):
        ok[:, 0] = False
        ok[:, -1] = False
    out[k] = np.where(ok, np.minimum(O, DEEP), O).astype(z[k].dtype)
    stats['knots'] += int(O.size)
    stats['below'] += int(b.sum())
    stats['deepened'] += int((ok & (O > DEEP)).sum())
np.savez(dst, **out)
with open(dst.replace('.npz', '') + '.json', 'w') as f:
    json.dump(stats, f, indent=1)
print(json.dumps(stats))
