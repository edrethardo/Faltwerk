#!/usr/bin/env python3
"""N3DV scene -> E2-style ground truth folder for the E7/E8 trainer (CPU only).

  python prep_frames.py --scene coffee_martini --frames 96 [--res 1352x1014] [--jobs 12]

Protocol (3DGStream CVPR'24 / HiCoM NeurIPS'24 / 4DGC CVPR'25): the 2704x2028 videos downsampled by two
(1352x1014), cam00 held out for testing, the remaining cameras train, the first 300 frames at 30 fps.
Frames: ffmpeg (imageio-ffmpeg static binary) decodes the mp4 at full resolution, PIL bicubic resize to 1352x1014 (same
filter as the 3DGS loader used by 4D-GS/HiCoM: PIL resize default), PNG. Output
  $ROOT/data/gt/n3dv_<scene>_<W>/f<t:03d>/c<i:02d>.png   (c index = sorted position of the camera, like COLMAP)
  $ROOT/data/gt/n3dv_<scene>_<W>/cameras.json             (E2 format: OpenCV w2c R,t, fx,fy,cx,cy, split; frames = extracted)
Re-running with a larger --frames extends the directory (existing PNGs are kept).
Cameras: poses_bounds.npy rows are 3x5 [R|t|hwf] c2w in the LLFF axis convention (down, right, back) plus near/far,
one row per EXISTING cam*.mp4 in sorted order (README of the dataset). Conversion as in STG/4D-GS pre_n3d.py:
columns -> (right, up, back) = OpenGL c2w, flip y,z -> OpenCV c2w, invert -> w2c. Intrinsics: f/2, cx=W/2, cy=H/2.
Dataset licence CC BY-NC 4.0 (research use): download it yourself; do not redistribute derived frames.
"""
import argparse, glob, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

ROOT = os.environ.get('ROOT', 'work')


def cameras_from_llff(pb, names, W, H, test_name='cam00'):
    """pb: poses_bounds [N,17]; names: sorted video names. Returns E2 camera dicts (OpenCV w2c) + bounds."""
    assert pb.shape[0] == len(names), f'{pb.shape[0]} poses vs {len(names)} videos'
    poses = pb[:, :15].reshape(-1, 3, 5)
    bounds = pb[:, 15:]
    hwf = poses[0, :, 4]
    h0, w0, f0 = float(hwf[0]), float(hwf[1]), float(hwf[2])
    sx, sy = W / w0, H / h0
    cams = []
    for i, nm in enumerate(names):
        m = poses[i, :, :4]
        c2w = np.concatenate([m[:, 1:2], -m[:, 0:1], m[:, 2:3], m[:, 3:4]], 1)          # (right, up, back | t) = OpenGL
        c2w[:, 1:3] *= -1                                                               # OpenCV (right, down, forward)
        c2w4 = np.eye(4); c2w4[:3] = c2w
        w2c = np.linalg.inv(c2w4)
        cams.append({'name': nm, 'R': w2c[:3, :3].tolist(), 't': w2c[:3, 3].tolist(),
                     'fx': f0 * sx, 'fy': f0 * sy, 'cx': W / 2.0, 'cy': H / 2.0, 'w': W, 'h': H,
                     'near': float(bounds[i, 0]), 'far': float(bounds[i, 1]),
                     'split': 'test' if nm == test_name else 'train'})
    return cams, {'h0': h0, 'w0': w0, 'f0': f0}


def extract(args):
    """one camera: decode `frames` frames at full res, resize, write PNGs. Returns (name, n_written, secs)."""
    mp4, ci, frames, W, H, out, ffmpeg = args
    t0 = time.time()
    probe = subprocess.run([ffmpeg, '-hide_banner', '-i', mp4], capture_output=True, text=True).stderr
    import re
    m = re.search(r'(\d{3,5})x(\d{3,5})', probe)
    w0, h0 = int(m.group(1)), int(m.group(2))
    cmd = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', mp4, '-frames:v', str(frames), '-f', 'rawvideo',
           '-pix_fmt', 'rgb24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w0 * h0 * 3 * 2)
    n = 0
    for t in range(frames):
        buf = p.stdout.read(w0 * h0 * 3)
        if len(buf) < w0 * h0 * 3:
            break
        dst = f'{out}/f{t:03d}/c{ci:02d}.png'
        if os.path.exists(dst):
            n += 1; continue
        im = Image.frombuffer('RGB', (w0, h0), buf, 'raw', 'RGB', 0, 1)
        im = im.resize((W, H), Image.Resampling.BICUBIC)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        im.save(dst, compress_level=6)
        n += 1
    p.stdout.close(); p.wait()
    return os.path.basename(mp4), n, (w0, h0), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--res', default='1352x1014')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--fps', type=float, default=30.0)
    a = ap.parse_args()
    W, H = (int(x) for x in a.res.split('x'))
    src = os.path.join(ROOT, 'data', 'n3dv', a.scene)
    vids = sorted(glob.glob(os.path.join(src, 'cam*.mp4')))
    names = [os.path.basename(v)[:-4] for v in vids]
    pb = np.load(os.path.join(src, 'poses_bounds.npy'))
    cams, raw = cameras_from_llff(pb, names, W, H)
    out = os.path.join(ROOT, 'data', 'gt', f'n3dv_{a.scene}_{W}')       # one dir per scene; --frames extends it
    os.makedirs(out, exist_ok=True)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    jobs = [(v, i, a.frames, W, H, out, ff) for i, v in enumerate(vids)]
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(extract, jobs))
    for r in res:
        print(r, flush=True)
    n_frames = min(r[1] for r in res)
    assert n_frames == a.frames, f'only {n_frames} frames decoded'
    assert all(r[2] == (int(raw['w0']), int(raw['h0'])) for r in res), 'video size differs from hwf in poses_bounds'
    meta = {'dataset': 'N3DV (Li et al. CVPR 2022, CC BY-NC 4.0, measure only)', 'scene': a.scene, 'frames': a.frames,
            'fps': a.fps, 'res': W, 'width': W, 'height': H, 'source_res': [int(raw['w0']), int(raw['h0'])],
            'focal_full_res': raw['f0'], 'resize': 'ffmpeg decode full res -> PIL BICUBIC',
            'test_camera': 'cam00', 'n_train': sum(c['split'] == 'train' for c in cams),
            'background': None, 'cameras': cams}
    json.dump(meta, open(os.path.join(out, 'cameras.json'), 'w'), indent=1)
    sz = sum(os.path.getsize(f) for f in glob.glob(f'{out}/f*/c*.png'))
    print(f'OK {out}: {len(cams)} cams ({meta["n_train"]} train), {a.frames} frames, {sz / 2**30:.2f} GiB, {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
