#!/usr/bin/env python3
"""COLMAP model files from cameras.json (before triangulation) and init points + camera self-check
from the triangulated model (after). CPU, train venv (numpy, PIL).

  python colmap_model.py write <gt_dir> <work_dir>      cameras.txt / images.txt (poses fixed, TRAIN cams only, ids =
                                                        sorted order like COLMAP's ImageReader), empty points3D.txt,
                                                        frame-0 PNGs copied to <work_dir>/images/
  python colmap_model.py points <gt_dir> <work_dir>     reads <work_dir>/sparse_txt/points3D.txt ->
                                                        <gt_dir>/init_points.npz (xyz, rgb, err, ext) and
                                                        <gt_dir>/selfcheck.json
Self-check (E2 rule, no depth maps here): every triangulated point carries COLMAP's colour from a TRAIN image;
projected into the held-out cam00 the GT colour there must match far better than a shuffled control (< 50 % of the
control error), otherwise the LLFF -> OpenCV conversion is wrong. Also reported: COLMAP's mean reprojection error.
"""
import json, os, shutil, sys
import numpy as np
from PIL import Image


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([[Rxx - Ryy - Rzz, 0, 0, 0], [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0], [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                  [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    w, v = np.linalg.eigh(K)
    q = v[[3, 0, 1, 2], np.argmax(w)]
    return -q if q[0] < 0 else q


def write(gt, work):
    meta = json.load(open(os.path.join(gt, 'cameras.json')))
    cams = meta['cameras']
    train = [i for i, c in enumerate(cams) if c['split'] == 'train']
    os.makedirs(os.path.join(work, 'images'), exist_ok=True)
    os.makedirs(os.path.join(work, 'sparse_in'), exist_ok=True)
    c0 = cams[train[0]]
    assert all(abs(cams[i]['fx'] - c0['fx']) < 1e-6 for i in train), 'intrinsics differ between cameras'
    with open(os.path.join(work, 'sparse_in', 'cameras.txt'), 'w') as f:
        f.write(f"1 PINHOLE {c0['w']} {c0['h']} {c0['fx']} {c0['fy']} {c0['cx']} {c0['cy']}\n")
    with open(os.path.join(work, 'sparse_in', 'images.txt'), 'w') as f:
        for k, i in enumerate(train):                           # names c01.png ... sorted => COLMAP ids 1..n
            c = cams[i]
            q = rotmat2qvec(np.array(c['R']))
            t = c['t']
            f.write(f"{k + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 c{i:02d}.png\n\n")
            shutil.copyfile(os.path.join(gt, 'f000', f'c{i:02d}.png'), os.path.join(work, 'images', f'c{i:02d}.png'))
    open(os.path.join(work, 'sparse_in', 'points3D.txt'), 'w').close()
    json.dump({'camera_params': f"{c0['fx']},{c0['fy']},{c0['cx']},{c0['cy']}", 'n_images': len(train)},
              open(os.path.join(work, 'model.json'), 'w'))
    print(f'wrote {len(train)} train images + model to {work}; camera_params {c0["fx"]},{c0["fy"]},{c0["cx"]},{c0["cy"]}')


def project(c, X):
    R, t = np.array(c['R']), np.array(c['t'])
    Xc = X @ R.T + t
    z = Xc[:, 2]
    u = c['fx'] * Xc[:, 0] / z + c['cx']; v = c['fy'] * Xc[:, 1] / z + c['cy']
    return u, v, z


def colour_check(c, img, X, rgb):
    u, v, z = project(c, X)
    ok = (z > 0) & (u >= 0) & (u < c['w'] - 1) & (v >= 0) & (v < c['h'] - 1)
    ui, vi = np.round(u[ok]).astype(int), np.round(v[ok]).astype(int)
    g = img[vi, ui].astype(np.float32)
    err = np.abs(g - rgb[ok].astype(np.float32)).mean()
    rng = np.random.default_rng(0)
    ctrl = np.abs(g - rgb[ok][rng.permutation(ok.sum())].astype(np.float32)).mean()
    return {'name': c['name'], 'split': c['split'], 'points_in_view': int(ok.sum()), 'reproj_rgb_err': round(float(err), 2),
            'shuffled_control_err': round(float(ctrl), 2), 'pass': bool(err < 0.5 * ctrl)}


def points(gt, work):
    meta = json.load(open(os.path.join(gt, 'cameras.json')))
    cams = meta['cameras']
    rows = []
    for line in open(os.path.join(work, 'sparse_txt', 'points3D.txt')):
        if line.startswith('#'): continue
        p = line.split()                                   # POINT3D_ID X Y Z R G B ERROR TRACK[(IMAGE_ID, POINT2D_IDX)]
        rows.append([float(x) for x in p[1:8]] + [(len(p) - 8) // 2])
    a = np.array(rows)
    X, rgb, err, ntrack = a[:, :3].astype(np.float32), a[:, 3:6].astype(np.uint8), a[:, 6], a[:, 7]
    centres = np.array([-(np.array(c['R']).T @ np.array(c['t'])) for c in cams])
    cen = centres.mean(0)
    radius = float(np.linalg.norm(centres - cen, axis=1).max())     # 3DGS: cameras_extent = 1.1 * radius
    d = np.linalg.norm(X - cen, axis=1)
    keep = d <= np.percentile(d, 98)                              # drop far outliers for the bbox statistic
    ext_pts = float(np.linalg.norm(X[keep].max(0) - X[keep].min(0)))
    # scene scale handed to e3/e7 (lr of means = 1.6e-4*ext in the fit, DefaultStrategy scene_scale = ext/2, curvature
    # step 1e-3*ext): 2 x the 3DGS cameras_extent, so the densification thresholds equal 3DGS's. The point bbox is not
    # usable here: N3DV's background lies 10-50x farther than the subject (LLFF far bound up to ~340 units).
    ext = 2.2 * radius
    test = [c for c in cams if c['split'] == 'test']
    train = [c for c in cams if c['split'] == 'train']
    gt0 = lambda c: np.asarray(Image.open(os.path.join(gt, 'f000', f"c{cams.index(c):02d}.png")))[..., :3]
    checks = [colour_check(c, gt0(c), X, rgb) for c in test] + [colour_check(train[0], gt0(train[0]), X, rgb)]
    sc = {'n_points': int(len(X)), 'colmap_mean_reproj_err_px': round(float(err.mean()), 3),
          'colmap_median_reproj_err_px': round(float(np.median(err)), 3), 'mean_track_len': round(float(ntrack.mean()), 2),
          'ext_scene_scale': round(ext, 4), 'ext_bbox_diag_p98': round(ext_pts, 4), 'camera_radius': round(radius, 4),
          'cam_centre_spread': round(float(np.linalg.norm(centres.max(0) - centres.min(0))), 4),
          'depth_p5_p50_p95_from_test_cam': [round(float(x), 3) for x in np.percentile(project(test[0], X)[2], [5, 50, 95])],
          'checks': checks, 'pass': all(c['pass'] for c in checks)}
    np.savez(os.path.join(gt, 'init_points.npz'), xyz=X, rgb=rgb, err=err.astype(np.float32), keep=keep, ext=ext)
    json.dump(sc, open(os.path.join(gt, 'selfcheck.json'), 'w'), indent=1)
    print(json.dumps(sc, indent=1))
    if not sc['pass']:
        print('SELFCHECK FAILED', file=sys.stderr); sys.exit(3)


def fixids(gt, work):
    """rewrite sparse_in/images.txt with the image ids COLMAP assigned in db.db (matched by name); robust against the
    order ImageReader used."""
    import sqlite3
    ids = dict(sqlite3.connect(os.path.join(work, 'db.db')).execute('SELECT name, image_id FROM images').fetchall())
    p = os.path.join(work, 'sparse_in', 'images.txt')
    out = []
    for line in open(p):
        parts = line.split()
        if len(parts) == 10:
            parts[0] = str(ids[parts[9]]); out.append(' '.join(parts) + '\n\n')
    open(p, 'w').write(''.join(out))
    print(f'images.txt ids from database: {sorted(ids.values())[:3]} ... {len(ids)} images')


if __name__ == '__main__':
    {'write': write, 'points': points, 'fixids': fixids}[sys.argv[1]](sys.argv[2], sys.argv[3])
