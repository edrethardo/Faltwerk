#!/usr/bin/env python3
"""P5 plausibility check: Python reference decode (codec/falt.py) + curve evaluation + gsplat render of the
same test cameras/frames the browser rendered; fg-PSNR browser vs GT, gsplat vs GT and browser vs gsplat.

Runs on a CUDA machine with torch + gsplat:
  python ref_render.py <csgs_dir> <gt_dir> <out_dir>      (out_dir holds bench.json + browser_*.rgb from bench.py)
Writes <out_dir>/psnr.json and merges it into bench.json; PNGs of a few comparisons into <out_dir>/cmp_*.png.
"""
import glob
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from codec.falt import Decoder, section_files  # noqa: E402


def psnr(a, b):
    return float(-10 * np.log10(max(((a - b) ** 2).mean(), 1e-10)))


def psnr_fg(a, b):                       # same mask as experiments/e3_track.psnr_fg
    m = ((a.sum(-1) > 0.01) | (b.sum(-1) > 0.01))[..., None].astype(np.float64)
    mse = (((a - b) ** 2) * m).sum() / max(m.sum() * 3, 1)
    return float(-10 * np.log10(max(mse, 1e-10)))


def section_arrays(sec, p):
    """dequantised knots per group as dense [n, Kmax, d] (padded with the last knot) + K per splat."""
    out, q0 = {}, 0
    for g, (tag, d) in enumerate(p.groups):
        K = np.array(sec.k[g])
        Km = int(K.max()) if K.size else 1                       # a section can be empty (0 splats)
        A = np.zeros((len(K), Km, d))
        for i, row in enumerate(sec.knots[g]):
            r = np.asarray(row, np.float64).reshape(-1, d)
            A[i, :len(r)] = r
            A[i, len(r):] = r[-1]
        off = np.asarray(p.offsets[q0:q0 + d]); sc = np.asarray(p.scales[q0:q0 + d]); q0 += d
        out[tag] = (K, (off + A * sc).astype(np.float32))
    sn = np.asarray(sec.sn, np.float64)
    if sn.size == 0:                                                # empty section
        sn = sn.reshape(0, len(p.scales) - q0)
    out['SN'] = (np.asarray(p.offsets[q0:]) + sn * np.asarray(p.scales[q0:])).astype(np.float32)
    out['BSPLINE'] = bool(getattr(p, 'bspline', False))        # P16: knots are control points of a clamped B-spline
    return out


def eval_at(arr, sec, t):
    """curves at frame t (same formula as the WGSL eval pass)."""
    n = len(sec.ids)
    birth = np.zeros(n); birth[sec.n_cont:] = sec.g0 + np.asarray(sec.birth)
    death = sec.g0 + np.asarray(sec.death, np.float64)
    fi = math.floor(t)
    alive = (birth <= fi) & (fi < death)
    span = max(sec.nframes - 1, 1)
    vals = {}
    for tag in ('P', 'R', 'S', 'O', 'D'):
        K, A = arr[tag]
        if arr.get('BSPLINE'):                                  # P16: weights per distinct K, same Cox-de Boor as the decoder
            from codec.falt import bspline_weights
            v = np.zeros((n, A.shape[2]))
            u = min(max((t - sec.g0) / span, 0.0), 1.0)
            for kv in np.unique(K).tolist():
                sel = np.nonzero(K == kv)[0]
                w = np.asarray(bspline_weights(int(kv), u))
                v[sel] = np.einsum('k,nkd->nd', w, A[sel, :kv].astype(np.float64))
            vals[tag] = v
            continue
        km1 = (K - 1).astype(np.float64)
        tau = np.clip((t - sec.g0) * km1 / span, 0, km1)
        j = np.minimum(np.floor(tau), np.maximum(km1 - 1, 0)).astype(int)
        w = (tau - j)[:, None]
        a = A[np.arange(n), j]
        b = A[np.arange(n), np.minimum(j + 1, A.shape[1] - 1)]
        v = a * (1 - w) + b * w
        vals[tag] = np.where((K > 1)[:, None], v, A[:, 0])
    return alive, vals


def main():
    import torch                              # GPU deps only here: the numpy helpers above run on any CPU
    from PIL import Image
    from gsplat import rasterization
    csgs_dir, gt_dir, out = sys.argv[1:4]
    bench = json.load(open(os.path.join(out, 'bench.json')))
    sc0 = next(s for s in bench['scenarios'] if s.get('test_renders'))
    meta = json.load(open(os.path.join(gt_dir, 'cameras.json')))
    cams = {f'c{i:02d}': c for i, c in enumerate(meta['cameras'])}
    dec = Decoder()
    secs = []
    t0 = time.time()
    for f in section_files(csgs_dir):
        s = dec.decode(open(f, 'rb').read())
        secs.append((s, section_arrays(s, dec.params)))
    t_dec = time.time() - t0
    print(f'python decode {t_dec:.1f}s', flush=True)
    dev = 'cuda'
    rows = []
    for tr in sc0['test_renders']:
        f, cn = tr['frame'], tr['cam']
        sec, arr = next((s, a) for s, a in secs if s.g0 <= f < s.g0 + s.nframes)
        alive, v = eval_at(arr, sec, float(tr['t']))
        idx = np.nonzero(alive)[0]
        c = cams[cn]
        W, H = c['w'], c['h']
        vm = np.eye(4, dtype=np.float32); vm[:3, :3] = c['R']; vm[:3, 3] = c['t']
        K = np.array([[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1]], np.float32)
        T = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)
        q = v['R'][idx]; q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        col = np.concatenate([v['D'][idx][:, None], arr['SN'][idx].reshape(-1, 15, 3)], 1)
        with torch.no_grad():
            rgb, _, _ = rasterization(T(v['P'][idx]), T(q), torch.exp(T(v['S'][idx])), torch.sigmoid(T(v['O'][idx, 0])),
                                      T(col), T(vm)[None], T(K)[None], W, H, sh_degree=3, packed=False)
        ref = rgb[0].clamp(0, 1).cpu().numpy()
        ref8 = np.round(ref * 255) / 255
        br = np.frombuffer(open(os.path.join(out, tr['file']), 'rb').read(), np.uint8).reshape(H, W, 3) / 255.0
        gt = np.asarray(Image.open(os.path.join(gt_dir, f'f{f:03d}', f'{cn}.png')))[..., :3] / 255.0
        row = dict(cam=cn, frame=f, t=tr['t'], alive=int(alive.sum()),
                   browser_vs_gt=dict(psnr=psnr(br, gt), fg_psnr=psnr_fg(br, gt)),
                   gsplat_vs_gt=dict(psnr=psnr(ref8, gt), fg_psnr=psnr_fg(ref8, gt)),
                   browser_vs_gsplat=dict(psnr=psnr(br, ref8), fg_psnr=psnr_fg(br, ref8),
                                          max_abs_8bit=int(np.abs(br - ref8).max() * 255 + 0.5)),
                   eval_check=tr.get('eval_check'))
        rows.append(row)
        print(f"{cn} f{f:03d}: fg-PSNR browser/GT {row['browser_vs_gt']['fg_psnr']:.2f}  gsplat/GT {row['gsplat_vs_gt']['fg_psnr']:.2f}"
              f"  browser/gsplat {row['browser_vs_gsplat']['fg_psnr']:.2f} dB", flush=True)
        if f in (20,) or (cn == 'c04'):
            im = np.concatenate([gt, ref8, br], 1)
            Image.fromarray((im * 255).astype(np.uint8)).save(os.path.join(out, f'cmp_{cn}_f{f:03d}.png'))
    mean = lambda k1, k2: float(np.mean([r[k1][k2] for r in rows]))
    summ = dict(n=len(rows), python_decode_s=round(t_dec, 1),
                mean_fg_psnr_browser_vs_gt=mean('browser_vs_gt', 'fg_psnr'),
                mean_fg_psnr_gsplat_vs_gt=mean('gsplat_vs_gt', 'fg_psnr'),
                mean_fg_psnr_browser_vs_gsplat=mean('browser_vs_gsplat', 'fg_psnr'),
                mean_psnr_browser_vs_gt=mean('browser_vs_gt', 'psnr'),
                mean_psnr_gsplat_vs_gt=mean('gsplat_vs_gt', 'psnr'))
    res = dict(summary=summ, rows=rows, note='cmp_*.png: GT | gsplat(ref decode) | browser')
    json.dump(res, open(os.path.join(out, 'psnr.json'), 'w'), indent=1)
    bench['plausibility'] = summ
    json.dump(bench, open(os.path.join(out, 'bench.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
