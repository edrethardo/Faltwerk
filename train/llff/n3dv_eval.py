#!/usr/bin/env python3
"""decoded quality of a .csgs stream on an N3DV scene (GPU, train venv, under the GPU lock).

  E3_GT=<gt_dir> python n3dv_eval.py <csgs_dir> <out.json> [--save-renders]

Like e8_decoded_psnr: Python reference decoder (codec/falt.py), curve evaluation as the WebGPU player (ref_render.eval_at),
gsplat render of the TEST camera(s) for EVERY frame, 8-bit rounded. Metrics on the full image (real background, no mask):
PSNR (per frame, then mean over frames like the literature), SSIM (3DGS implementation: 11x11 gaussian window, sigma 1.5,
the one used by 3DGStream/4DGC code bases; e3.ssim). LPIPS is added by n3dv_lpips.py (CPU venv) from the saved renders.
Rate: bytes of all .csgs sections / frames -> KB/frame and Mbit/s at the GT fps (30). Splats: alive per frame.
"""
import functools, glob, json, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'web', 'bench'))
import e3_track as e3                                      # noqa: E402
from ref_render import section_arrays, eval_at            # noqa: E402
from codec.falt import Decoder, section_files  # noqa: E402

NEAR = 0.9 * min(c['near'] for c in e3.cams)              # same near-plane culling as n3dv_train.py (LLFF bounds)
e3.rasterization = functools.partial(e3.rasterization, near_plane=NEAR)


def main():
    csgs_dir, out = sys.argv[1:3]
    save = '--save-renders' in sys.argv
    fps = float(e3.meta.get('fps', 30.0))
    t0 = time.time()
    dec, secs, nbytes, sizes = Decoder(), [], 0, []
    for f in section_files(csgs_dir):
        b = open(f, 'rb').read(); nbytes += len(b); sizes.append(len(b))
        s = dec.decode(b)
        secs.append((s, section_arrays(s, dec.params)))
    t_dec = time.time() - t0
    T = secs[-1][0].g0 + secs[-1][0].nframes
    print(f'decoded {len(secs)} sections, {nbytes} B, T={T}, {t_dec:.1f} s', flush=True)
    rdir = csgs_dir.rstrip('/') + '.renders'
    if save:
        os.makedirs(rdir, exist_ok=True)
    T_ = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=e3.dev)
    psnr, ssim, alive_n = [], [], []
    for f in range(T):
        sec, arr = next((s, a) for s, a in secs if s.g0 <= f < s.g0 + s.nframes)
        alive, v = eval_at(arr, sec, float(f))
        idx = np.nonzero(alive)[0]
        q = {'means': T_(v['P'][idx]), 'quats': T_(v['R'][idx]), 'scales': T_(v['S'][idx]), 'opacities': T_(v['O'][idx, 0]),
             'sh0': T_(v['D'][idx][:, None]), 'shN': T_(arr['SN'][idx].reshape(len(idx), -1, 3))}
        with torch.no_grad():
            rgb, _, _ = e3.render(q, e3.test)
            rgb = torch.round(rgb * 255) / 255
            gt = e3.load_frame(f)[e3.test]
            psnr.append(e3.psnr(rgb, gt)); ssim.append(float(e3.ssim(rgb, gt))); alive_n.append(len(idx))
            if save:
                for ci, c in enumerate(e3.test):
                    Image.fromarray((rgb[ci].cpu().numpy() * 255).astype(np.uint8)).save(f'{rdir}/f{f:03d}_c{c:02d}.png')
        print(f'frame {f}: N={len(idx)} PSNR {psnr[-1]:.2f} dB SSIM {ssim[-1]:.4f}', flush=True)
    pa = np.array(psnr)
    res = {'csgs_dir': csgs_dir, 'gt': e3.GT, 'test_cams': [e3.cams[i]['name'] for i in e3.test], 'frames': T, 'fps': fps,
           'bytes': nbytes, 'bytes_per_section': sizes, 'kb_per_frame': nbytes / 1000 / T, 'mb_per_frame': nbytes / 1e6 / T,
           'mbit_s': nbytes * 8 / (T / fps) / 1e6, 'python_decode_s': round(t_dec, 1),
           'psnr': psnr, 'ssim': ssim, 'alive': alive_n, 'mean_psnr': float(pa.mean()), 'min_psnr': float(pa.min()),
           'argmin_psnr': int(pa.argmin()), 'last_psnr': float(pa[-1]), 'mean_ssim': float(np.mean(ssim)),
           'splats_per_frame': float(np.mean(alive_n)), 'bytes_per_splat_frame': nbytes / float(np.sum(alive_n)),
           'renders': rdir if save else None,
           'note': 'full-image PSNR / SSIM (3DGS impl.) of the gsplat render of the DECODED .csgs on the held-out camera, '
                   '8-bit rounded, mean over frames; rate from the .csgs bytes at the GT fps'}
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('bytes', 'kb_per_frame', 'mbit_s', 'mean_psnr', 'min_psnr', 'argmin_psnr', 'last_psnr',
                                          'mean_ssim', 'splats_per_frame', 'bytes_per_splat_frame')}, indent=1))
    print('P6EVAL OK', out, flush=True)


if __name__ == '__main__':
    main()
