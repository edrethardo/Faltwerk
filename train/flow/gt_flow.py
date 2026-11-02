#!/usr/bin/env python3
"""optical flow of the GT between neighbouring frames, per camera (classic OpenCV DIS, CPU).

For every frame pair (t, t+1), t = 0 .. T-2, and every camera of the E2-format GT dir:
  forward flow F (DIS, preset medium, on the full-resolution luminance), backward flow B (t+1 -> t),
  forward/backward consistency conf = exp(-|F(x) + B(x + F(x))|^2 / thr), thr = 0.01 (|F|^2 + |B_w|^2) + 0.5
  (UnFlow-style occlusion test, soft), times the GT matte (max channel >= 2/255, dilated 2 px) unless --matte none
  (real backgrounds, e.g. N3DV). Flow and confidence are area-downsampled to --scale (flow in pixels OF THAT scale).
Output: <out>/f<t:03d>.npz with flow int16 [C, h, w, 2] (x FLOW_Q px) and conf uint8 [C, h, w] (x 255), meta.json.
No GPU, no torch: runs with any python3 with cv2 + numpy.
  python3 gt_flow.py --gt <gt_dir> --out <dir> [--scale 0.5] [--frames N] [--matte auto|none] [--workers 20]
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

FLOW_Q = 16          # int16 units per pixel (range +-2048 px at the stored scale)

ap = argparse.ArgumentParser()
ap.add_argument('--gt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--scale', type=float, default=0.5)
ap.add_argument('--frames', type=int, default=None)
ap.add_argument('--matte', default='auto', choices=('auto', 'none'))
ap.add_argument('--workers', type=int, default=20)
ap.add_argument('--preset', default='medium', choices=('fast', 'medium', 'ultrafast'))
A = ap.parse_args()
META = json.load(open(os.path.join(A.gt, 'cameras.json')))
NC = len(META['cameras'])
# E2 GT dirs carry width/height, N3DV res, W/H (portrait rigs): accept all three spellings
W = int(META.get('width') or META.get('W') or META['res'])
H = int(META.get('height') or META.get('H') or META['res'])
w, h = int(round(W * A.scale)), int(round(H * A.scale))
T = min(A.frames or META['frames'], META['frames'])


def gray(t, c):
    im = cv2.imread(os.path.join(A.gt, f'f{t:03d}', f'c{c:02d}.png'), cv2.IMREAD_COLOR)    # BGR
    return im


def one_pair(t):
    cv2.setNumThreads(1)
    preset = {'fast': cv2.DISOPTICAL_FLOW_PRESET_FAST, 'medium': cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
              'ultrafast': cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST}[A.preset]
    dis = cv2.DISOpticalFlow_create(preset)
    fl = np.zeros((NC, h, w, 2), np.int16)
    cf = np.zeros((NC, h, w), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    st = {'mag': [], 'conf': []}
    for c in range(NC):
        i0, i1 = gray(t, c), gray(t + 1, c)
        g0, g1 = cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
        F = dis.calc(g0, g1, None)
        B = dis.calc(g1, g0, None)
        Bw = cv2.remap(B, xx + F[..., 0], yy + F[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        err = ((F + Bw) ** 2).sum(-1)
        thr = 0.01 * ((F ** 2).sum(-1) + (Bw ** 2).sum(-1)) + 0.5
        conf = np.exp(-err / thr)
        if A.matte == 'auto':
            m = (i0.max(-1) >= 2).astype(np.uint8)
            m = cv2.dilate(m, np.ones((5, 5), np.uint8))
            conf = conf * m
        Fs = cv2.resize(F, (w, h), interpolation=cv2.INTER_AREA) * A.scale
        Cs = cv2.resize(conf.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        fl[c] = np.clip(np.rint(Fs * FLOW_Q), -32767, 32767).astype(np.int16)
        cf[c] = np.clip(np.rint(Cs * 255), 0, 255).astype(np.uint8)
        sel = Cs > 0.5
        if sel.any():
            st['mag'].append(float(np.linalg.norm(Fs[sel], axis=-1).mean())); st['conf'].append(float(sel.mean()))
    np.savez_compressed(os.path.join(A.out, f'f{t:03d}.npz'), flow=fl, conf=cf)
    return t, (float(np.mean(st['mag'])) if st['mag'] else 0.0), (float(np.mean(st['conf'])) if st['conf'] else 0.0)


def main():
    os.makedirs(A.out, exist_ok=True)
    t0 = time.time()
    todo = [t for t in range(T - 1) if not os.path.exists(os.path.join(A.out, f'f{t:03d}.npz'))]
    stats = {}
    with Pool(A.workers) as p:
        for t, mag, cfr in p.imap_unordered(one_pair, todo):
            stats[t] = {'mean_flow_px_conf05': round(mag, 4), 'conf05_share': round(cfr, 5)}
            print(f'flow f{t:03d}: |F| {mag:.3f} px (scale {A.scale}), conf>0.5 share {cfr:.4f} ({time.time() - t0:.0f} s)', flush=True)
    meta = {'gt': A.gt, 'scale': A.scale, 'size_wh': [w, h], 'frames': T, 'pairs': 'f<t>.npz = t -> t+1', 'n_cams': NC,
            'flow_q': FLOW_Q, 'conf_q': 255, 'method': f'OpenCV {cv2.__version__} DIS preset {A.preset}, fwd/bwd consistency',
            'matte': A.matte, 'secs': round(time.time() - t0, 1), 'stats': stats}
    json.dump(meta, open(os.path.join(A.out, 'meta.json'), 'w'), indent=1)
    print('P24FLOW OK', A.out, f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
