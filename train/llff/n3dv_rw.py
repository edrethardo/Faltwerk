#!/usr/bin/env python3
"""red/white side measures of a decoded N3DV .csgs stream (held-out camera, every --fstep-th frame).
N3DV has no matte: the full image is the content. The decoded stream is rendered with alpha and composited on RED and on
WHITE (gsplat, near plane 0.9 x min LLFF near like n3dv_eval.py):
  hole %  = share of pixels with alpha < 0.5 (background shows through; ideal 0 on N3DV)
  dark %  = share of pixels whose luminance is > 0.15 below the GT (dark splats / see-through), per background
  psnr    = full-image PSNR per background
Adds the block 'rw' to the eval JSON of n3dv_eval.py (or writes <out>).
  E3_GT=<gt> python n3dv_rw.py <csgs_dir> <eval.json> [--fstep 4]
"""
import functools, glob, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'web', 'bench'))
import e3_track as e3                                      # noqa: E402
from ref_render import section_arrays, eval_at            # noqa: E402
from codec.falt import Decoder, section_files  # noqa: E402

NEAR = 0.9 * min(c['near'] for c in e3.cams)
e3.rasterization = functools.partial(e3.rasterization, near_plane=NEAR)
LUM = torch.tensor([0.299, 0.587, 0.114], device=e3.dev)
BGS = {'red': (1.0, 0.0, 0.0), 'white': (1.0, 1.0, 1.0)}


def main():
    csgs_dir, js = sys.argv[1:3]
    fstep = int(sys.argv[sys.argv.index('--fstep') + 1]) if '--fstep' in sys.argv else 4
    dec, secs = Decoder(), []
    for f in section_files(csgs_dir):
        s = dec.decode(open(f, 'rb').read())
        secs.append((s, section_arrays(s, dec.params)))
    T = secs[-1][0].g0 + secs[-1][0].nframes
    T_ = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=e3.dev)
    acc = {b: {'hole': [], 'dark': [], 'psnr': []} for b in BGS}
    for f in range(0, T, fstep):
        sec, arr = next((s, a) for s, a in secs if s.g0 <= f < s.g0 + s.nframes)
        alive, v = eval_at(arr, sec, float(f))
        idx = np.nonzero(alive)[0]
        q = {'means': T_(v['P'][idx]), 'quats': T_(v['R'][idx]), 'scales': T_(v['S'][idx]), 'opacities': T_(v['O'][idx, 0]),
             'sh0': T_(v['D'][idx][:, None]), 'shN': T_(arr['SN'][idx].reshape(len(idx), -1, 3))}
        with torch.no_grad():
            rgb, alpha, _ = e3.render(q, e3.test)                     # rgb over black (premultiplied), alpha [C,H,W,1]
            gt = e3.load_frame(f)[e3.test]
            gl = (gt * LUM).sum(-1)
            for b, col in BGS.items():
                bg = torch.tensor(col, device=e3.dev)
                im = torch.round((rgb + (1 - alpha) * bg).clamp(0, 1) * 255) / 255
                acc[b]['hole'].append(float((alpha[..., 0] < 0.5).float().mean()) * 100)
                acc[b]['dark'].append(float(((im * LUM).sum(-1) < gl - 0.15).float().mean()) * 100)
                acc[b]['psnr'].append(e3.psnr(im, gt))
        print(f'frame {f}: ' + ' '.join(f'{b} hole {acc[b]["hole"][-1]:.3f}% dark {acc[b]["dark"][-1]:.3f}% '
                                        f'psnr {acc[b]["psnr"][-1]:.2f}' for b in BGS), flush=True)
    rw = {b: {k: float(np.mean(v)) for k, v in d.items()} for b, d in acc.items()}
    rw['fstep'] = fstep; rw['near'] = NEAR
    res = json.load(open(js)) if os.path.exists(js) else {}
    res['rw'] = rw
    json.dump(res, open(js, 'w'), indent=1)
    print('RW', json.dumps(rw), flush=True)


if __name__ == '__main__':
    main()
