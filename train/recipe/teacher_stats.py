#!/usr/bin/env python3
"""teacher statistics for the standard-recipe rules of the images source with per-frame teacher models
(P43 stage 1): extent of the frame-0 teacher (opacity >= minop) and the median nearest-neighbour spacing of its splats.
  python teacher_stats.py <ply> [--minop 0.005]   (train/rig/rig_lib.py for the PLY reader)
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rig'))
import rig_lib as L                                      # noqa: E402
import torch                                            # noqa: E402

path = sys.argv[1]
if '{' in path:                                         # a per-frame pattern: frame 0
    path = path.format(0)
minop = float(sys.argv[sys.argv.index('--minop') + 1]) if '--minop' in sys.argv else 0.005
p = L.ply_params(L.read_ply(path))
m = np.asarray(p['means'], np.float32)
o = 1 / (1 + np.exp(-np.asarray(p['opacities'], np.float32).reshape(-1)))
sel = o >= minop
m = m[sel]
ext = float(np.linalg.norm(m.max(0) - m.min(0)))
rng = np.random.default_rng(0)
qi = rng.choice(len(m), min(20000, len(m)), replace=False)
M = torch.from_numpy(m)
nn = []
for i in range(0, len(qi), 500):                        # exact NN of a 20k sample against all splats (no scipy here)
    d = torch.cdist(M[qi[i:i + 500]], M)
    d[torch.arange(d.shape[0]), torch.from_numpy(qi[i:i + 500])] = float('inf')
    nn.append(d.min(1).values.numpy())
nn = np.concatenate(nn)
print(json.dumps({'ply': path, 'n': int(sel.sum()), 'ext': ext, 'nn_p10_p50_p90': [float(np.percentile(nn, q)) for q in (10, 50, 90)],
                  'sigma_p10': float(np.percentile(np.exp(np.asarray(p['scales'], np.float32)[sel].max(-1)), 10)),
                  'sigma_p50': float(np.median(np.exp(np.asarray(p['scales'], np.float32)[sel].max(-1))))}))
