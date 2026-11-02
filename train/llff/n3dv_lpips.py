#!/usr/bin/env python3
"""LPIPS (AlexNet and VGG, Zhang et al. 2018, `lpips` package v0.1 weights) of the saved test-camera
renders against the GT frames; CPU only. Adds lpips_alex/lpips_vgg (per frame + mean) to the
eval JSON written by n3dv_eval.py. Papers on N3DV report LPIPS-VGG or LPIPS-Alex - both given.

  python n3dv_lpips.py <eval.json> [--threads 20]
"""
import json, os, re, sys, time
import numpy as np
import torch
import lpips
from PIL import Image


def main():
    path = sys.argv[1]
    th = int(sys.argv[sys.argv.index('--threads') + 1]) if '--threads' in sys.argv else 20
    torch.set_num_threads(th)
    res = json.load(open(path))
    rdir, gt = res['renders'], res['gt']
    assert rdir and os.path.isdir(rdir), 'run n3dv_eval.py with --save-renders first'
    nets = {n: lpips.LPIPS(net=n, verbose=False).eval() for n in ('alex', 'vgg')}
    dv = 'cuda' if '--cuda' in sys.argv else 'cpu'     # optional GPU (same weights/inputs; default CPU unchanged)
    nets = {n: net.to(dv) for n, net in nets.items()}
    files = sorted(f for f in os.listdir(rdir) if f.endswith('.png'))
    out = {n: [] for n in nets}
    t0 = time.time()
    to_t = lambda im: (torch.from_numpy(np.asarray(im)[..., :3].copy()).permute(2, 0, 1)[None].float() / 127.5 - 1).to(dv)
    with torch.no_grad():
        for f in files:
            m = re.match(r'f(\d+)_c(\d+)\.png', f)
            t, c = int(m.group(1)), int(m.group(2))
            a = to_t(Image.open(os.path.join(rdir, f)))
            b = to_t(Image.open(os.path.join(gt, f'f{t:03d}', f'c{c:02d}.png')))
            for n, net in nets.items():
                out[n].append(float(net(a, b)))
            if t % 10 == 0:
                print(f'{f}: alex {out["alex"][-1]:.4f} vgg {out["vgg"][-1]:.4f} ({time.time() - t0:.0f} s)', flush=True)
    for n in nets:
        res[f'lpips_{n}'] = out[n]; res[f'mean_lpips_{n}'] = float(np.mean(out[n]))
    res['lpips_note'] = 'lpips package v0.1 (AlexNet / VGG16 backbones, torchvision ImageNet weights), inputs scaled to [-1,1], CPU'
    json.dump(res, open(path, 'w'), indent=1)
    print(f'LPIPS OK {path}: alex {res["mean_lpips_alex"]:.4f} vgg {res["mean_lpips_vgg"]:.4f} over {len(files)} images')


if __name__ == '__main__':
    main()
