"""Rig helpers: 3DGS PLY reader, splat dict conventions, Morton order, metrics on numpy/torch.
No GPU import at module level (ply_codec.py uses it on CPU)."""
import math
import numpy as np

SH_C0 = 0.28209479177387814


def read_ply(path):
    """binary little-endian 3DGS PLY -> structured array (x y z f_dc_* opacity scale_* rot_* [...])."""
    with open(path, 'rb') as f:
        hdr = []
        while True:
            line = f.readline().decode('ascii', 'replace').strip()
            hdr.append(line)
            if line == 'end_header':
                break
        assert 'format binary_little_endian 1.0' in hdr, hdr[:3]
        n = int(next(l.split()[2] for l in hdr if l.startswith('element vertex')))
        tmap = {'float': 'f4', 'double': 'f8', 'uchar': 'u1', 'int': 'i4', 'uint': 'u4', 'short': 'i2', 'ushort': 'u2'}
        dt = np.dtype([(p.split()[2], '<' + tmap[p.split()[1]]) for p in hdr if p.startswith('property')])
        return np.frombuffer(f.read(n * dt.itemsize), dt, count=n)


def ply_params(a):
    """INRIA 3DGS conventions (checked on real 3DGS PLY files: |rot| = 1, rot_0 = w (median 0.83), scale = log metres
    (exp -> 6e-6 .. 0.13 m), opacity = logit, f_dc = SH0 coefficient): float32 dict in the e3 render layout."""
    q = np.stack([a[f'rot_{i}'] for i in range(4)], 1).astype(np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return {'means': np.stack([a['x'], a['y'], a['z']], 1).astype(np.float32), 'quats': q,
            'scales': np.stack([a[f'scale_{i}'] for i in range(3)], 1).astype(np.float32),
            'opacities': a['opacity'].astype(np.float32),
            'sh0': np.stack([a[f'f_dc_{i}'] for i in range(3)], 1).astype(np.float32)[:, None]}


def morton_order(x, bits=10):
    """splat order along a 3D Z-curve (lossless reordering, helps the DPCM of absolute first knots)."""
    lo, hi = x.min(0), x.max(0)
    g = np.clip(((x - lo) / np.maximum(hi - lo, 1e-9) * ((1 << bits) - 1)).astype(np.int64), 0, (1 << bits) - 1)
    code = np.zeros(len(x), np.int64)
    for b in range(bits):
        for d in range(3):
            code |= ((g[:, d] >> b) & 1) << (3 * b + d)
    return np.argsort(code, kind='stable')


def rotmat_to_quat(R):
    """3x3 rotation -> unit quaternion wxyz."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def quat_mul(a, b):
    """Hamilton product, wxyz, a [4] or [N,4] times b [N,4]."""
    aw, ax, ay, az = np.moveaxis(np.broadcast_to(a, b.shape), -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def apply_similarity(p, M):
    """splat dict (ply_params layout) under X' = M[:3,:3] X + M[:3,3] with M[:3,:3] = s * rotation."""
    A = np.asarray(M, np.float64)[:3]
    s = float(np.cbrt(np.linalg.det(A[:, :3])))
    Rw = A[:, :3] / s
    out = dict(p)
    out['means'] = (p['means'].astype(np.float64) @ A[:, :3].T + A[:, 3]).astype(np.float32)
    out['scales'] = (p['scales'] + math.log(s)).astype(np.float32)
    out['quats'] = quat_mul(rotmat_to_quat(Rw), p['quats'].astype(np.float64)).astype(np.float32)
    return out


def psnr_np(a, b):
    return float(-10 * np.log10(max(((a - b) ** 2).mean(), 1e-10)))


def psnr_fg_np(a, b):          # same mask as e3.psnr_fg: pixels where GT or render is non-black
    m = ((a.sum(-1) > 0.01) | (b.sum(-1) > 0.01))[..., None].astype(np.float64)
    return float(-10 * np.log10(max((((a - b) ** 2) * m).sum() / max(m.sum() * 3, 1), 1e-10)))
