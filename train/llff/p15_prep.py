#!/usr/bin/env python3
"""depth targets for the frame-0 fit of an N3DV scene (train venv).

  python p15_prep.py sparse <gt_dir>   COLMAP sparse depth of every TRAIN camera from <gt>/colmap/sparse_txt
                                       (images.txt keypoints with a 3D point, depth = z of the point in the camera of
                                       cameras.json) -> <gt>/sparse_depth.npz (c<i>_uv [n,2] pixel coords, c<i>_z, c<i>_err)
                                       + <gt>/sparse_depth.json (counts, depth percentiles). CPU.
  python p15_prep.py mono <gt_dir> [--model depth-anything/Depth-Anything-V2-Large-hf]
                                       monocular relative inverse depth (Depth Anything V2 via transformers, GPU) of frame
                                       0 of every TRAIN camera, scale/shift fitted per camera to the COLMAP sparse inverse
                                       depth (least squares, two passes with 2.5 MAD outlier rejection) ->
                                       <gt>/mono_invdepth.npz (c<i> float16 [H,W]) + <gt>/mono_fit.json (fit quality:
                                       correlation, median relative depth error at the sparse points)
  python p15_prep.py checkcams <gt_dir> <other cameras.json>   poses/intrinsics of both files equal (exit 1 if not)
  python p15_prep.py md5 <dir>         md5 of every PNG below <dir> (sorted, relative names) for frame comparisons
N3DV is CC BY-NC: everything written here stays on the GPU box.
"""
import hashlib, json, os, sys
import numpy as np


def cams_of(gt):
    return json.load(open(os.path.join(gt, 'cameras.json')))['cameras']


def sparse_points(gt):
    pts = {}
    for line in open(os.path.join(gt, 'colmap', 'sparse_txt', 'points3D.txt')):
        if line.startswith('#'):
            continue
        p = line.split()
        pts[int(p[0])] = (np.array([float(x) for x in p[1:4]]), float(p[7]), (len(p) - 8) // 2)
    obs = {}
    lines = [l for l in open(os.path.join(gt, 'colmap', 'sparse_txt', 'images.txt')) if not l.startswith('#')]
    for k in range(0, len(lines) - 1, 2):
        h = lines[k].split()
        if len(h) < 10:
            continue
        ci = int(h[9][1:3])                                     # c01.png -> camera index 1 (sorted position)
        a = lines[k + 1].split()
        rows = [(float(a[j]), float(a[j + 1]), int(a[j + 2])) for j in range(0, len(a) - 2, 3) if int(a[j + 2]) >= 0]
        obs[ci] = rows
    return pts, obs


def sparse(gt, max_err=2.0):
    cams = cams_of(gt)
    pts, obs = sparse_points(gt)
    out, stats = {}, {}
    for ci, rows in sorted(obs.items()):
        c = cams[ci]
        assert c['split'] == 'train', f'camera {ci} in the COLMAP model is not a train camera'
        R, t = np.array(c['R']), np.array(c['t'])
        uv, z, err, rep = [], [], [], []
        for u, v, pid in rows:
            X, e, _ = pts[pid]
            if e > max_err:
                continue
            Xc = R @ X + t
            if Xc[2] <= 0:
                continue
            uv.append((u, v)); z.append(Xc[2]); err.append(e)
            # reprojection sanity of the camera mapping (pose of cameras.json vs. COLMAP keypoint)
            rep.append(np.hypot(c['fx'] * Xc[0] / Xc[2] + c['cx'] - u, c['fy'] * Xc[1] / Xc[2] + c['cy'] - v))
        uv, z, err = np.array(uv, np.float32).reshape(-1, 2), np.array(z, np.float32), np.array(err, np.float32)
        out[f'c{ci:02d}_uv'], out[f'c{ci:02d}_z'], out[f'c{ci:02d}_err'] = uv, z, err
        stats[c['name']] = {'n': int(len(z)), 'z_p5_p50_p95': [round(float(x), 3) for x in np.percentile(z, [5, 50, 95])],
                            'reproj_px_median': round(float(np.median(rep)), 3)}
    np.savez(os.path.join(gt, 'sparse_depth.npz'), **out)
    json.dump({'max_err_px': max_err, 'cams': stats}, open(os.path.join(gt, 'sparse_depth.json'), 'w'), indent=1)
    bad = [k for k, v in stats.items() if v['reproj_px_median'] > 2.0]
    print(json.dumps(stats, indent=0)[:1500])
    if bad:
        print('REPROJECTION CHECK FAILED for', bad, file=sys.stderr); sys.exit(3)
    print(f'P15 SPARSE OK {sum(v["n"] for v in stats.values())} keypoints in {len(stats)} cams')


def mono(gt, model):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    cams = cams_of(gt)
    sp = np.load(os.path.join(gt, 'sparse_depth.npz'))
    proc = AutoImageProcessor.from_pretrained(model)
    net = AutoModelForDepthEstimation.from_pretrained(model).cuda().eval()
    out, fits = {}, {}
    for ci, c in enumerate(cams):
        if c['split'] != 'train' or f'c{ci:02d}_z' not in sp.files:
            continue
        im = Image.open(os.path.join(gt, 'f000', f'c{ci:02d}.png')).convert('RGB')
        W, H = im.size
        with torch.no_grad():
            x = proc(images=im, return_tensors='pt').to('cuda')
            d = net(**x).predicted_depth                          # relative inverse depth ("disparity")
            d = torch.nn.functional.interpolate(d[:, None], size=(H, W), mode='bicubic', align_corners=False)[0, 0]
        d = d.float().cpu().numpy()
        uv, z = sp[f'c{ci:02d}_uv'], sp[f'c{ci:02d}_z']
        ui = np.clip(np.round(uv[:, 0] - 0.5).astype(int), 0, W - 1); vi = np.clip(np.round(uv[:, 1] - 0.5).astype(int), 0, H - 1)
        xs, ys = d[vi, ui], 1.0 / z
        keep = np.ones(len(xs), bool)
        s = b = 0.0
        for _ in range(2):
            A = np.stack([xs[keep], np.ones(keep.sum())], 1)
            s, b = np.linalg.lstsq(A, ys[keep], rcond=None)[0]
            r = s * xs + b - ys
            mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-12
            keep = np.abs(r - np.median(r[keep])) < 2.5 * 1.4826 * mad
        inv = np.maximum(s * d + b, 1.0 / (100 * np.percentile(z, 99)))   # far clamp: 100 x the farthest sparse depth
        zi = 1.0 / inv[vi, ui]
        fits[c['name']] = {'scale': float(s), 'shift': float(b), 'inliers': int(keep.sum()), 'n': int(len(xs)),
                           'corr_invdepth': round(float(np.corrcoef(xs, ys)[0, 1]), 4),
                           'median_rel_depth_err': round(float(np.median(np.abs(zi - z) / z)), 4),
                           'neg_before_clamp_frac': round(float(((s * d + b) <= 0).mean()), 4)}
        out[f'c{ci:02d}'] = inv.astype(np.float16)
        print(c['name'], fits[c['name']], flush=True)
    np.savez(os.path.join(gt, 'mono_invdepth.npz'), **out)
    json.dump({'model': model, 'fits': fits}, open(os.path.join(gt, 'mono_fit.json'), 'w'), indent=1)
    print(f'P15 MONO OK {len(out)} cams, median rel err {np.median([f["median_rel_depth_err"] for f in fits.values()]):.3f}')


def checkcams(gt, other):
    a, b = cams_of(gt), json.load(open(other))['cameras']
    ok = len(a) == len(b) and all(x['name'] == y['name'] and x['split'] == y['split'] and
                                  np.allclose(x['R'], y['R']) and np.allclose(x['t'], y['t']) and
                                  abs(x['fx'] - y['fx']) < 1e-6 for x, y in zip(a, b))
    print('cameras equal' if ok else 'CAMERAS DIFFER')
    sys.exit(0 if ok else 1)


def md5(d):
    for root, _, files in sorted(os.walk(d)):
        for f in sorted(files):
            if f.endswith('.png'):
                p = os.path.join(root, f)
                print(hashlib.md5(open(p, 'rb').read()).hexdigest(), os.path.relpath(p, d))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'sparse':
        sparse(sys.argv[2])
    elif cmd == 'mono':
        mono(sys.argv[2], sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == '--model' else 'depth-anything/Depth-Anything-V2-Large-hf')
    elif cmd == 'checkcams':
        checkcams(sys.argv[2], sys.argv[3])
    elif cmd == 'md5':
        md5(sys.argv[2])
