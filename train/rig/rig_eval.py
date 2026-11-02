#!/usr/bin/env python3
"""Decoded quality of a Faltwerk stream on a matted multi-camera rig (GPU, gsplat).

  P8_GT=<gt dir> python train/rig/rig_eval.py <falt dir> <out.json> [--fstep 1]

The stream is decoded with the Python reference decoder (codec/falt.py), the curves are evaluated per frame like the
WebGPU player (web/bench/ref_render.eval_at) and the held-out test cameras of <gt dir>/cameras.json are rendered with
gsplat (8-bit rounded). The GT frames are RGBA, RGB premultiplied by the matte (black outside).
  black   : PSNR (full image), fg-PSNR (pixels where GT or render is non-black), SSIM against the premultiplied GT
  red/white: render composited over the colour with its own alpha, GT composited over the same colour with its matte;
            PSNR, hole % (matte >= 0.5 but rendered alpha < 0.5) and dark % (inside the matte, luminance > 0.15 below
            the GT) - near-transparent or black splats are invisible on black and show up only here.
Rate: bytes of all sections over the frames at the GT fps.
"""
import json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'web', 'bench'))
import rig_track as e3                                       # noqa: E402  reads $P8_GT/cameras.json at import
from ref_render import section_arrays, eval_at               # noqa: E402
from codec.falt import Decoder, section_files                # noqa: E402

LUM = torch.tensor([0.299, 0.587, 0.114], device=e3.dev)
BGS = {'red': (1.0, 0.0, 0.0), 'white': (1.0, 1.0, 1.0)}


def main():
    sdir, out = sys.argv[1:3]
    fstep = int(sys.argv[sys.argv.index('--fstep') + 1]) if '--fstep' in sys.argv else 1
    fps = float(e3.meta.get('fps', 30.0))
    t0 = time.time()
    dec, secs, nbytes = Decoder(), [], 0
    for f in section_files(sdir):
        b = open(f, 'rb').read(); nbytes += len(b)
        s = dec.decode(b)
        secs.append((s, section_arrays(s, dec.params)))
    T = secs[-1][0].g0 + secs[-1][0].nframes
    T_ = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=e3.dev)   # noqa: E731
    acc = {'psnr': [], 'psnr_fg': [], 'ssim': [], 'alive': []}
    rw = {b: {'psnr': [], 'hole': [], 'dark': []} for b in BGS}
    for f in range(0, T, fstep):
        sec, arr = next((s, a) for s, a in secs if s.g0 <= f < s.g0 + s.nframes)
        alive, v = eval_at(arr, sec, float(f))
        idx = np.nonzero(alive)[0]
        q = {'means': T_(v['P'][idx]), 'quats': torch.nn.functional.normalize(T_(v['R'][idx]), dim=-1),
             'scales': T_(v['S'][idx]), 'opacities': T_(v['O'][idx, 0]), 'sh0': T_(v['D'][idx][:, None]),
             'shN': T_(arr['SN'][idx].reshape(len(idx), -1, 3))}
        with torch.no_grad():
            rgb, alpha, _ = e3.render(q, e3.test)            # rgb over black (premultiplied), alpha [C,H,W,1]
            g = e3.load_frame(f, rgba=True)[e3.test]
            gt, ga = g[..., :3], g[..., 3:4]
            im = torch.round(rgb * 255) / 255
            acc['psnr'].append(e3.psnr(im, gt)); acc['psnr_fg'].append(e3.psnr_fg(im, gt))
            acc['ssim'].append(float(e3.ssim(im, gt))); acc['alive'].append(len(idx))
            inside = ga[..., 0] >= 0.5
            for b, col in BGS.items():
                bg = torch.tensor(col, device=e3.dev)
                ib = torch.round((rgb + (1 - alpha) * bg).clamp(0, 1) * 255) / 255
                gb = (gt + (1 - ga) * bg).clamp(0, 1)
                rw[b]['psnr'].append(e3.psnr(ib, gb))
                n_in = max(int(inside.sum()), 1)
                rw[b]['hole'].append(float((inside & (alpha[..., 0] < 0.5)).sum()) / n_in * 100)
                dark = inside & ((ib * LUM).sum(-1) < (gb * LUM).sum(-1) - 0.15)
                rw[b]['dark'].append(float(dark.sum()) / n_in * 100)
        print(f'frame {f}: N={len(idx)} PSNR {acc["psnr"][-1]:.2f} fg {acc["psnr_fg"][-1]:.2f} SSIM {acc["ssim"][-1]:.4f} '
              + ' '.join(f'{b} {rw[b]["psnr"][-1]:.2f} dB hole {rw[b]["hole"][-1]:.2f}%' for b in BGS), flush=True)
    res = {'stream': sdir, 'gt': e3.GT, 'test_cams': [e3.cams[i].get('name', i) for i in e3.test], 'frames': T,
           'fstep': fstep, 'fps': fps, 'bytes': nbytes, 'mbit_s': nbytes * 8 / (T / fps) / 1e6,
           'mean_psnr': float(np.mean(acc['psnr'])), 'mean_psnr_fg': float(np.mean(acc['psnr_fg'])),
           'mean_ssim': float(np.mean(acc['ssim'])), 'splats_per_frame': float(np.mean(acc['alive'])),
           'per_frame': acc, 'rw': {b: {k: float(np.mean(x)) for k, x in d.items()} for b, d in rw.items()},
           'seconds': round(time.time() - t0, 1),
           'note': 'decoded stream, gsplat render of the test cameras, 8-bit rounded; black = premultiplied GT'}
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('mbit_s', 'mean_psnr', 'mean_psnr_fg', 'mean_ssim', 'splats_per_frame', 'rw')}))
    print('RIG EVAL OK', out, flush=True)


if __name__ == '__main__':
    main()
