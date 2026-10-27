#!/usr/bin/env python3
"""(P9): data for the comparison viewer web/compare.html. Never writes into the repo.

  splq      (the GPU host, train venv: numpy)  P8-plyq frames -> one .splq blob per frame (format: web/src/splq.ts)
            make_viewer_data.py splq <plyq_method_dir> <out_dir> [--align mlsharp_align_refined.json] [--frames 50]
            The integers are recovered from <NNNNN>.npz (dequantised by ply_codec.py) with the affine params of the
            frame's own .csgs header: the viewer shows exactly the plyq values (q8 + rANS rate), only the rANS layer
            is dropped for transport (the TS rANS decoder needs ~0.4 s per 300K-splat frame, too slow for playback).
  gt        (the GPU host, PIL)  GT frames fNNN/cNN.png (RGB used as is, like eval_all.py) -> <out>/cNN/fNNN.jpg
            make_viewer_data.py gt <gt_dir> <out_dir> --cams 15,24 --frames 64 [--quality 88]
  manifest  (the build host, stdlib only)  recipe JSON -> manifest.json (orbit + test cameras from cameras.json, pane list,
            rate labels read from the run JSONs). make_viewer_data.py manifest <recipe.json>
            Recipe format: web/viewer.recipe.example.json. Paths in the recipe are relative to the recipe's directory
            (= the served root, data/viewer/).
"""
import argparse
import glob
import json
import math
import os
import struct
import sys

SPLQ_MAGIC = b'SPLQ'
SPLQ_VERSION = 1
HDR_BYTES = 256


# ---------------------------------------------------------------- .splq
def csgs_affine(path):
    """offsets/scales (P3 R4 S3 O1 D3) of a P8-plyq single-frame .csgs section (codec/csgs.py header layout)."""
    buf = open(path, 'rb').read(4096)
    assert buf[:4] == b'CSGS', path
    o = struct.calcsize('<4sBBIIHHIII')
    _, G = struct.unpack_from('<BB', buf, o); o += 2
    groups = []
    for _ in range(G):
        t, d = struct.unpack_from('<BB', buf, o); o += 2
        groups.append((chr(t), d))
    nk = buf[o]; o += 1 + nk
    (n_sn,) = struct.unpack_from('<H', buf, o); o += 2 + n_sn
    assert groups == [('P', 3), ('R', 4), ('S', 3), ('O', 1), ('D', 3)] and n_sn == 0, (groups, n_sn)
    nq = 14
    off = struct.unpack_from(f'<{nq}f', buf, o); o += 4 * nq
    sc = struct.unpack_from(f'<{nq}f', buf, o)
    return off, sc


def rotmat_to_quat(R):
    """same as experiments/james/p8_lib.rotmat_to_quat (wxyz)."""
    w = math.sqrt(max(0.0, 1 + R[0][0] + R[1][1] + R[2][2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0][0] - R[1][1] - R[2][2])) / 2, R[2][1] - R[1][2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0][0] + R[1][1] - R[2][2])) / 2, R[0][2] - R[2][0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0][0] - R[1][1] + R[2][2])) / 2, R[1][0] - R[0][1])
    return [w, x, y, z]


def similarity(M):
    """4x4 world matrix (s * rotation + t, p8_lib.apply_similarity) -> rows A (3x4), unit quat of the rotation, log s."""
    import numpy as np
    A = np.asarray(M, np.float64)[:3]
    s = float(np.cbrt(np.linalg.det(A[:, :3])))
    return A, rotmat_to_quat((A[:, :3] / s).tolist()), math.log(s)


def splq_header(n, off, sc, sim=None, sn=False, run=False):
    flags = (1 if sim is not None else 0) | (2 if sn else 0) | (4 if run else 0)
    A, qr, logs = sim if sim is not None else ([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], [1, 0, 0, 0], 0.0)
    h = SPLQ_MAGIC + struct.pack('<III', SPLQ_VERSION, n, flags) + struct.pack('<14f', *off) + struct.pack('<14f', *sc)
    h += struct.pack('<12f', *[float(v) for row in A for v in row]) + struct.pack('<4f', *qr) + struct.pack('<f', logs)
    return h + b'\0' * (HDR_BYTES - len(h))


def pack_splq(q, off, sc, sim=None, sn=None, run=None):
    """q: dict of int arrays P [n,3] (0..1023), R [n,4] (-127..127), S [n,3], O [n], D [n,3] (0..255) -> bytes.
    optional blocks after the n x 16 B splat words (web/src/splq.ts):
      run: position ints [n,3] 0..65535 -> flag bit 2, n x 8 B (u32 x | y << 16, u32 z); off/sc[0:3] then apply to
           these 16-bit values and the 10-bit P field of w0 is 0 (large N3DV extents: 10 bit loses ~3.7 dB at cam00)
      sn:  (SN ints [n,m] 0..255, coefficient-major like pack.ts, m <= 45; off[45]; sc[45]) -> flag bit 1,
           u32 m, f32 off[45], f32 sc[45], then n x ceil4(m) B = higher SH bands (missing ones = 0)."""
    import numpy as np
    n = len(q['P'])
    P, R, S, O, D = (np.asarray(q[k], np.int64) for k in 'PRSOD')
    if run is not None:
        P = np.zeros((n, 3), np.int64)
    assert P.min() >= 0 and P.max() <= 1023, ('P range', P.min(), P.max())
    assert R.min() >= -127 and R.max() <= 127, ('R range', R.min(), R.max())
    for k, a in (('S', S), ('O', O), ('D', D)):
        assert a.min() >= 0 and a.max() <= 255, (k, a.min(), a.max())
    w = np.zeros((n, 4), np.uint32)
    w[:, 0] = (P[:, 0] | (P[:, 1] << 10) | (P[:, 2] << 20)).astype(np.uint32)
    Ru = (R & 255).astype(np.uint32)
    w[:, 1] = Ru[:, 0] | (Ru[:, 1] << 8) | (Ru[:, 2] << 16) | (Ru[:, 3] << 24)
    w[:, 2] = (S[:, 0] | (S[:, 1] << 8) | (S[:, 2] << 16) | (O << 24)).astype(np.uint32)
    w[:, 3] = (D[:, 0] | (D[:, 1] << 8) | (D[:, 2] << 16)).astype(np.uint32)
    out = splq_header(n, off, sc, sim, sn is not None, run is not None) + w.astype('<u4').tobytes()
    if run is not None:
        P16 = np.asarray(run, np.int64)
        assert P16.shape == (n, 3) and P16.min() >= 0 and P16.max() <= 65535, ('P16', P16.shape)
        w2 = np.zeros((n, 2), np.uint32)
        w2[:, 0] = (P16[:, 0] | (P16[:, 1] << 16)).astype(np.uint32); w2[:, 1] = P16[:, 2].astype(np.uint32)
        out += w2.astype('<u4').tobytes()
    if sn is not None:
        qs, so, ss = sn
        qs = np.asarray(qs, np.int64); m = qs.shape[1]
        assert qs.shape[0] == n and 0 < m <= 45 and qs.min() >= 0 and qs.max() <= 255, ('SN', qs.shape)
        so, ss = list(so) + [0.0] * (45 - len(so)), list(ss) + [0.0] * (45 - len(ss))
        st = (m + 3) // 4 * 4
        blk = np.zeros((n, st), np.uint8); blk[:, :m] = qs
        out += struct.pack('<I', m) + struct.pack('<45f', *so) + struct.pack('<45f', *ss) + blk.tobytes()
    return out


def recover_ints(z, off, sc):
    """npz floats (off + q * sc in float64, stored as float32) -> the plyq integers; asserts an exact roundtrip."""
    import numpy as np
    cols = {'P': (z['means'], 0, 3), 'R': (z['quats'], 3, 4), 'S': (z['scales'], 7, 3),
            'O': (z['opacities'][:, None], 10, 1), 'D': (z['sh0'].reshape(len(z['sh0']), 3), 11, 3)}
    q, worst = {}, 0.0
    for k, (v, o, d) in cols.items():
        o_ = np.asarray(off[o:o + d], np.float64); s_ = np.asarray(sc[o:o + d], np.float64)
        qi = np.round((v.astype(np.float64) - o_) / s_).astype(np.int64)
        back = (o_ + qi * s_).astype(np.float32)
        err = float(np.abs(back - v).max())
        assert err == 0.0, (k, err)
        worst = max(worst, err)
        q[k] = qi[:, 0] if k == 'O' else qi
    return q, worst


def cmd_splq(a):
    import numpy as np
    os.makedirs(a.out, exist_ok=True)
    align = json.load(open(a.align))['frames'] if a.align else None
    files = sorted(glob.glob(os.path.join(a.src, '*.npz')))[:a.frames]
    rows = []
    for f in files:
        fi = int(os.path.basename(f)[:-4])
        off, sc = csgs_affine(f[:-4] + '.csgs')
        z = np.load(f)
        q, err = recover_ints(z, off, sc)
        sim = similarity(align[str(fi)]['M_world']) if align else None
        b = pack_splq(q, off, sc, sim)
        open(os.path.join(a.out, f'{fi:05d}.splq'), 'wb').write(b)
        rows.append({'frame': fi, 'n': len(q['P']), 'bytes': len(b), 'csgs_bytes': os.path.getsize(f[:-4] + '.csgs')})
        print(json.dumps(rows[-1]), flush=True)
    json.dump({'src': a.src, 'align': a.align, 'frames': rows, 'roundtrip': 'exact (npz == off + q*sc)'},
              open(os.path.join(a.out, 'splq.json'), 'w'), indent=1)
    print('SPLQ OK', a.out, len(rows), 'frames')


# ---------------------------------------------------------------- GT stills
def cmd_gt(a):
    from PIL import Image
    cams = [int(c) for c in a.cams.split(',')]
    n = 0
    for c in cams:
        od = os.path.join(a.out, f'c{c:02d}')
        os.makedirs(od, exist_ok=True)
        for f in range(a.frames):
            src = os.path.join(a.gt, f'f{f:03d}', f'c{c:02d}.png')
            if not os.path.exists(src):
                break
            Image.open(src).convert('RGB').save(os.path.join(od, f'f{f:03d}.jpg'), quality=a.quality)
            n += 1
    print('GT OK', a.out, n, 'images')


# ---------------------------------------------------------------- manifest
def jget(root, spec):
    """{"json": path, "path": "a.b.c"} -> value; "path" may also be a list of keys (for keys containing dots)."""
    v = json.load(open(os.path.join(root, spec['json'])))
    keys = spec['path'] if isinstance(spec['path'], list) else spec['path'].split('.')
    for k in keys:
        v = v[k]
    return v


def fmt_rate(x):
    """two decimals below 100 Mbit/s, whole numbers above."""
    return (f'{x:.2f}' if x < 100 else f'{x:.0f}').replace('.', ',')


def fmt_metric(x, fmt):
    """fmt: Python format spec (e.g. '.2f', '.4f'); decimal comma."""
    return format(float(x), fmt).replace('.', ',')


def orbit_from_cameras(meta, front_idx):
    """centre = least-squares intersection of all optical axes, up = -mean camera y, front = towards camera front_idx."""
    cams = meta['cameras']
    S = [[0.0] * 3 for _ in range(3)]; b = [0.0] * 3; up = [0.0] * 3
    for c in cams:
        R, t = c['R'], c['t']
        pos = [-(R[0][j] * t[0] + R[1][j] * t[1] + R[2][j] * t[2]) for j in range(3)]
        d = R[2]
        for i in range(3):
            up[i] -= R[1][i]
            for j in range(3):
                m = (1.0 if i == j else 0.0) - d[i] * d[j]
                S[i][j] += m; b[i] += m * pos[j]
    center = solve3(S, b)
    if 'center' in meta:
        center = meta['center']
    if 'up' in meta:
        up = meta['up']
    nu = math.sqrt(sum(u * u for u in up)); up = [u / nu for u in up]
    c0 = cams[front_idx]
    R, t = c0['R'], c0['t']
    pos = [-(R[0][j] * t[0] + R[1][j] * t[1] + R[2][j] * t[2]) for j in range(3)]
    v = [pos[i] - center[i] for i in range(3)]
    dist = math.sqrt(sum(x * x for x in v))
    vu = sum(v[i] * up[i] for i in range(3))
    front = [v[i] - vu * up[i] for i in range(3)]
    nf = math.sqrt(sum(x * x for x in front)); front = [x / nf for x in front]
    return dict(center=center, up=up, front=front, dist=dist, elev_deg=math.degrees(math.asin(max(-1, min(1, vu / dist)))),
                fovy_deg=math.degrees(2 * math.atan(0.5 * c0['h'] / c0['fy'])))


def solve3(A, b):
    import copy
    M = [row[:] + [b[i]] for i, row in enumerate(copy.deepcopy(A))]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(3):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [M[r][k] - f * M[c][k] for k in range(4)]
    return [M[i][3] / M[i][i] for i in range(3)]


def csgs_sections(root, d):
    out = []
    for f in sorted(os.listdir(os.path.join(root, d))):
        if not f.endswith('.csgs'):
            continue
        with open(os.path.join(root, d, f), 'rb') as fh:
            h = fh.read(20)
        out.append(dict(url=f'{d.rstrip("/")}/{f}', g0=int.from_bytes(h[10:14], 'little'),
                        nframes=int.from_bytes(h[16:18], 'little'), bytes=os.path.getsize(os.path.join(root, d, f))))
    return out


def cmd_manifest(a):
    root = os.path.dirname(os.path.abspath(a.recipe))
    rec = json.load(open(a.recipe))
    scenes = []
    for s in rec['scenes']:
        meta = json.load(open(os.path.join(root, s['cameras'])))
        cams = meta['cameras']
        test = s.get('testcams') or [i for i, c in enumerate(cams) if c['split'] == 'test']
        front = s.get('front', test[0])
        panes = []
        for p in s['panes']:
            q = dict(id=p['id'], title=p['title'], kind=p['kind'])
            if p['kind'] == 'csgs':
                q['sections'] = csgs_sections(root, p['dir'])
            else:
                fr = sorted(f for f in os.listdir(os.path.join(root, p['dir'])) if f.endswith('.splq'))
                q['frames'] = [f'{p["dir"].rstrip("/")}/{f}' for f in fr]
                q['bytes'] = sum(os.path.getsize(os.path.join(root, p['dir'], f)) for f in fr)
            rates = [dict(label=r['label'], mbit_s=float(jget(root, r))) for r in p.get('rates', [])]
            q['rates'] = rates
            q['rate_label'] = ' · '.join(f'{r["label"]} {fmt_rate(r["mbit_s"])}'.strip() for r in rates) + (' Mbit/s' if rates else '')
            # quality label from run JSONs, e.g. [{"label": "fg-PSNR", "json": ..., "path": "variants.x.fg_psnr_gt", "fmt": ".2f", "unit": " dB"}]
            mets = [dict(label=m['label'], value=float(jget(root, m)), fmt=m.get('fmt', '.2f'), unit=m.get('unit', '')) for m in p.get('metrics', [])]
            if mets:
                q['metrics'] = mets
                q['metric_label'] = ' · '.join(f'{m["label"]} {fmt_metric(m["value"], m["fmt"])}{m["unit"]}' for m in mets)
            for k in ('note', 'default_off', 'source'):
                if k in p:
                    q[k] = p[k]
            panes.append(q)
        nfr = s.get('frames') or min([sum(x['nframes'] for x in q['sections']) if q['kind'] == 'csgs' else len(q['frames'])
                                      for q in panes])
        gt = None
        if s.get('gt'):
            gt = dict(dir=s['gt']['dir'], ext=s['gt'].get('ext', 'jpg'),
                      cams=sorted(int(d[1:]) for d in os.listdir(os.path.join(root, s['gt']['dir'])) if d.startswith('c')))
            if s['gt'].get('title'):          # pane title of the stills, e.g. "Original (the vendor demo)"
                gt['title'] = s['gt']['title']
        ref = None
        # near plane of the scene (gsplat near_plane used by the training/eval): recipe `near`, else 0.9 x min LLFF
        # `near` of the cameras (N3DV cameras.json), else absent (viewer default 0.01)
        near = s.get('near')
        if near is None and all('near' in c for c in cams):
            near = 0.9 * min(c['near'] for c in cams)
        if s.get('reference_fg_psnr'):
            ref = {k: jget(root, v) for k, v in s['reference_fg_psnr'].items()}
        scenes.append(dict(id=s['id'], title=s['title'], fps=float(meta['fps']), frames=int(nfr), note=s.get('note', ''),
                           orbit=orbit_from_cameras(meta, front), gt=gt, panes=panes, reference_fg_psnr=ref,
                           default_snap=s.get('default_snap'),
                           near=near,
                           testcams=[dict(idx=i, name=cams[i].get('name', f'c{i:02d}'), R=cams[i]['R'], t=cams[i]['t'],
                                          fx=cams[i]['fx'], fy=cams[i]['fy'], cx=cams[i]['cx'], cy=cams[i]['cy'],
                                          w=cams[i]['w'], h=cams[i]['h']) for i in test]))
    out = os.path.join(root, rec.get('out', 'manifest.json'))
    json.dump(dict(generated_from=os.path.basename(a.recipe), scenes=scenes), open(out, 'w'), indent=1)
    for s in scenes:
        print(s['id'], s['frames'], 'frames,', [(p['id'], p['rate_label']) for p in s['panes']])
    print('MANIFEST OK', out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('splq'); p.add_argument('src'); p.add_argument('out'); p.add_argument('--align')
    p.add_argument('--frames', type=int, default=10 ** 6)
    p = sub.add_parser('gt'); p.add_argument('gt'); p.add_argument('out'); p.add_argument('--cams', required=True)
    p.add_argument('--frames', type=int, default=10 ** 6); p.add_argument('--quality', type=int, default=88)
    p = sub.add_parser('manifest'); p.add_argument('recipe')
    a = ap.parse_args(argv)
    {'splq': cmd_splq, 'gt': cmd_gt, 'manifest': cmd_manifest}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
