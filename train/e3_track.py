#!/usr/bin/env python3
"""E3 – Our own 4DGS reconstruction on the E2 ground truth (GPU, train venv, gsplat).

Model (v3): a pool of splats, each with a lifetime [birth, death) in frames.
  * frame 0: full 3DGS fit from back-projected depth (SH3, DefaultStrategy densification, L1 + 0.2 D-SSIM).
  * every REFRESH frames: densify/prune phase on that frame (all attributes). Splats the strategy removes die;
    clones/splits get new ids and are born -> lifetimes, like a per-splat life window.
  * other frames: warm start from t-1, track per-frame means, quats, scales, opacity and colour DC (sh0);
    higher SH bands (shN) stay as fitted at the last refresh.
Black background (the reference renders are opaque black, alpha is always 255).
Metrics: PSNR (full image) and PSNR_fg (pixels where GT or render is non-black; the honest one).
Output: <run>/tracks.npz (per-frame arrays over the global id space, NaN when not alive; birth, death, shN),
        <run>/e3.json, <run>/preview_f000.png and preview_last.png (GT | ours, test views).
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

ROOT = os.environ.get('ROOT', 'work')
NAME = os.environ.get('SCENE', 'scene')
SMOKE = os.environ.get('SMOKE') == '1'
GT = os.environ.get('E3_GT') or os.path.join(ROOT, 'data', 'gt', f"{NAME}_{os.environ.get('E2_CHUNKS', '2')}c{os.environ.get('E2_RES', '512')}")   # E3_GT: any E2-format dir (P6 N3DV)
RUN = os.environ.get('RUN', os.path.join(ROOT, 'runs', 'e3', time.strftime('%Y%m%d-%H%M%S')))
N_INIT = int(os.environ.get('E3_N', '20000' if SMOKE else '100000'))
IT0 = int(os.environ.get('E3_IT0', '600' if SMOKE else '20000'))
ITT = int(os.environ.get('E3_ITT', '20' if SMOKE else '300'))
ITR = int(os.environ.get('E3_ITR', '200' if SMOKE else '1500'))
REFRESH = int(os.environ.get('E3_REFRESH', '2' if SMOKE else '8'))
SH = int(os.environ.get('E3_SH', '3'))
REG = float(os.environ.get('E3_REG', '0'))
dev = 'cuda'
meta = json.load(open(os.path.join(GT, 'cameras.json')))
cams = meta['cameras']
T = 4 if SMOKE else int(os.environ.get('E3_FRAMES', meta['frames']))
train = [i for i, c in enumerate(cams) if c['split'] == 'train']
test = [i for i, c in enumerate(cams) if c['split'] == 'test']
TRACKED = ['means', 'quats', 'scales', 'opacities', 'sh0']     # per-frame attributes


def viewmat(c):
    m = np.eye(4, dtype=np.float32); m[:3, :3] = c['R']; m[:3, 3] = c['t']; return m


VM = torch.tensor(np.stack([viewmat(c) for c in cams]), device=dev)
K = torch.tensor(np.stack([[[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1]] for c in cams]), dtype=torch.float32, device=dev)
W = int(meta.get('width', meta['res'])); H = int(meta.get('height', meta['res']))   # non-square GT (P6 N3DV 1352x1014); default unchanged


def load_frame(t):
    ims = [np.asarray(Image.open(f'{GT}/f{t:03d}/c{i:02d}.png'))[..., :3] for i in range(len(cams))]
    return torch.tensor(np.stack(ims), device=dev).float() / 255.0      # [C,H,W,3]


def render(p, ids, deg=None):
    """p: means, quats, scales (log), opacities (logit), sh0 [N,1,3], shN [N,K-1,3] (shN may have 0 bands)."""
    col = torch.cat([p['sh0'], p['shN']], 1) if p['shN'].shape[1] else p['sh0']
    if deg is None:
        deg = int(math.isqrt(col.shape[1]) - 1)
    else:
        col = col[:, :(deg + 1) ** 2]
    rgb, alpha, info = rasterization(p['means'], p['quats'], torch.exp(p['scales']), torch.sigmoid(p['opacities']),
                                     col, VM[ids], K[ids], W, H, sh_degree=deg, packed=False)
    return rgb.clamp(0, 1), alpha, info


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean().clamp_min(1e-10)))


def psnr_fg(a, b):
    m = ((a.sum(-1) > 0.01) | (b.sum(-1) > 0.01)).float()[..., None]
    mse = (((a - b) ** 2) * m).sum() / (m.sum() * 3).clamp_min(1)
    return float(-10 * torch.log10(mse.clamp_min(1e-10)))


def ssim(x, y):   # x,y [B,H,W,3] -> mean SSIM, 11x11 gaussian window
    x, y = x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2)
    g = torch.exp(-((torch.arange(11, device=x.device) - 5.0) ** 2) / (2 * 1.5 ** 2)); g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
    f = lambda z: Fn.conv2d(z, w, padding=5, groups=3)
    mx, my = f(x), f(y); sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))).mean()


def loss_fn(rgb, g):
    return 0.8 * (rgb - g).abs().mean() + 0.2 * (1 - ssim(rgb, g))


def init_points(gt0):
    pts, cols = [], []
    for i in train:
        c = cams[i]
        z = np.load(f'{GT}/f000/c{i:02d}_depth.npy').astype(np.float32)
        im = gt0[i].cpu().numpy()
        v, u = np.nonzero(z > 0)
        zz = z[v, u]
        X = np.stack([(u + .5 - c['cx']) / c['fx'] * zz, (v + .5 - c['cy']) / c['fy'] * zz, zz], 1)
        pts.append((X - np.array(c['t'])) @ np.array(c['R'])); cols.append(im[v, u])
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    sel = np.random.default_rng(0).choice(len(pts), min(N_INIT, len(pts)), replace=False)
    pts, cols = pts[sel], cols[sel]
    ext = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    n = len(pts)
    p = {'means': pts.astype(np.float32), 'quats': np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
         'scales': np.full((n, 3), math.log(ext / math.sqrt(n) * 0.7), np.float32),
         'opacities': np.full((n,), -2.0, np.float32),
         'sh0': ((cols - 0.5) / 0.28209479).astype(np.float32)[:, None],
         'shN': np.zeros((n, (SH + 1) ** 2 - 1, 3), np.float32)}
    return {k: torch.tensor(v, device=dev) for k, v in p.items()}, ext


def lr_table(ext, scale=1.0):
    return {'means': 1.6e-4 * ext * scale, 'quats': 1e-3, 'scales': 5e-3, 'opacities': 5e-2, 'sh0': 2.5e-3, 'shN': 2.5e-3 / 20}


def fit_densify(p, ids, gt, iters, ext, first):
    """Full optimisation with densification. Carries the non-trainable 'ids' through clone/split/prune."""
    P = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in p.items()})
    lrs = lr_table(ext, 1.0 if first else 0.3)
    opts = {k: torch.optim.Adam([P[k]], lr=lrs[k], eps=1e-15) for k in P}
    if first:
        strat = DefaultStrategy(refine_stop_iter=int(iters * 0.75), reset_every=3000, verbose=False)
    else:   # short refresh: refine from the start, no opacity reset (would wipe the tracked state);
            # gentler pruning / easier growth than the default, the pool shrank 98K -> 70K over 64 frames in v3
        strat = DefaultStrategy(refine_start_iter=100, refine_stop_iter=int(iters * 0.8), refine_every=100,
                                reset_every=10 ** 9, prune_opa=0.002, grow_grad2d=0.00015, verbose=False)
    strat.check_sanity(P, opts)
    P['ids'] = torch.nn.Parameter(ids.float(), requires_grad=False)   # added after the sanity check on purpose
    state = strat.initialize_state(scene_scale=ext / 2)
    sched = torch.optim.lr_scheduler.ExponentialLR(opts['means'], gamma=0.01 ** (1.0 / iters))
    for it in range(iters):
        deg = min(SH, it // 1000) if first else SH
        cam = list(np.random.choice(train, 4, replace=False))
        rgb, _, info = render(P, cam, deg)
        strat.step_pre_backward(P, opts, state, it, info)
        loss = loss_fn(rgb, gt[cam])
        loss.backward()
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        sched.step()
        strat.step_post_backward(P, opts, state, it, info, packed=False)
        if first and it % 2000 == 0:
            print(f'  it {it} N={P["means"].shape[0]} loss {loss.item():.4f}', flush=True)
    out = {k: v.detach().contiguous() for k, v in P.items() if k != 'ids'}
    return out, P['ids'].detach().round().long()


def track(p, gt, ext):
    lrs = {'means': 4e-5 * ext, 'quats': 5e-4, 'scales': 1e-3, 'opacities': 1e-2, 'sh0': 1e-3}
    prev = {k: p[k].detach().clone() for k in TRACKED}   # rate proxy: L1 on change vs t-1 (E3_REG, 0 = off)
    norm = {'means': ext * 1e-3, 'quats': 1e-2, 'scales': 1e-2, 'opacities': 1e-1, 'sh0': 1e-2}
    for k in p: p[k].requires_grad_(k in TRACKED)
    opt = torch.optim.Adam([{'params': [p[k]], 'lr': lrs[k]} for k in TRACKED], eps=1e-15)
    for _ in range(ITT):
        cam = list(np.random.choice(train, 4, replace=False))
        rgb, _, _ = render(p, cam)
        loss = loss_fn(rgb, gt[cam])
        if REG > 0:
            loss = loss + REG * sum(((p[k] - prev[k]).abs() / norm[k]).mean() for k in TRACKED)
        opt.zero_grad(); loss.backward(); opt.step()
    for k in p: p[k].requires_grad_(False)


def evaluate(p, gt):
    with torch.no_grad():
        rgb, _, _ = render(p, test)
        g = gt[test]
        return psnr(rgb, g), psnr_fg(rgb, g), rgb


def preview(gt, rgb, path):
    im = torch.cat([torch.cat([gt[test][i], rgb[i]], 1) for i in range(len(test))], 0)
    Image.fromarray((im.cpu().numpy() * 255).astype(np.uint8)).save(path)


def main():
    os.makedirs(RUN, exist_ok=True)
    t0 = time.time()
    gt = load_frame(0)
    p, ext = init_points(gt)
    ids = torch.arange(p['means'].shape[0], device=dev)
    p, ids = fit_densify(p, ids, gt, IT0, ext, first=True)
    ids = torch.arange(len(ids), device=dev)          # frame 0: every surviving splat is a fresh id
    next_id = len(ids)
    birth = {int(i): 0 for i in ids.tolist()}
    death = {}
    shN = {}
    frames = []                                        # per frame: (ids np, {attr: np})
    log = {'ext': ext, 'sh': SH, 'it0': IT0, 'itt': ITT, 'itr': ITR, 'refresh': REFRESH,
           'psnr': [], 'psnr_fg': [], 'alive': [], 'born': [], 'died': [], 'secs': []}

    def record(t, ts, kind):
        full, fg, rgb = evaluate(p, gt)
        frames.append((ids.cpu().numpy(), {k: p[k].cpu().numpy() for k in TRACKED}))
        if kind != 'track':                            # shN only changes at fit/refresh
            for i, row in zip(ids.tolist(), p['shN'].cpu().numpy()):
                shN[i] = row
        log['psnr'].append(full); log['psnr_fg'].append(fg); log['alive'].append(len(ids)); log['secs'].append(time.time() - ts)
        print(f'frame {t} [{kind}]: N={len(ids)} PSNR {full:.2f} dB, fg {fg:.2f} dB ({time.time() - ts:.1f}s)', flush=True)
        return rgb

    rgb = record(0, t0, 'fit'); preview(gt, rgb, os.path.join(RUN, 'preview_f000.png'))
    log['born'].append(len(ids)); log['died'].append(0)
    for t in range(1, T):
        ts = time.time()
        gt = load_frame(t)
        if t % REFRESH == 0:
            track(p, gt, ext)                          # bring the pool to frame t first, then densify/prune
            p, got = fit_densify(p, ids, gt, ITR, ext, first=False)
            got = got.cpu().numpy()
            new_ids = np.empty_like(got); seen = set(); born = 0
            for j, g in enumerate(got):
                g = int(g)
                if g in seen:                          # second copy of a split/clone -> newborn
                    new_ids[j] = next_id; birth[next_id] = t; next_id += 1; born += 1
                else:
                    new_ids[j] = g; seen.add(g)
            died = [int(i) for i in ids.tolist() if int(i) not in seen]
            for i in died:
                death[i] = t
            ids = torch.tensor(new_ids, device=dev)
            log['born'].append(born); log['died'].append(len(died))
            rgb = record(t, ts, f'refresh +{born} -{len(died)}')
        else:
            track(p, gt, ext)
            log['born'].append(0); log['died'].append(0)
            rgb = record(t, ts, 'track')
    preview(gt, rgb, os.path.join(RUN, 'preview_last.png'))

    # assemble over the global id space
    Ng = next_id
    out = {k: np.full((T, Ng) + frames[0][1][k].shape[1:], np.nan, np.float16 if k in ('scales', 'opacities', 'sh0') else np.float32)
           for k in TRACKED}
    for t, (fid, attrs) in enumerate(frames):
        for k in TRACKED:
            out[k][t, fid] = attrs[k]
    b = np.array([birth[i] for i in range(Ng)], np.int32)
    d = np.array([death.get(i, T) for i in range(Ng)], np.int32)
    sn = np.zeros((Ng,) + next(iter(shN.values())).shape, np.float16)
    for i, v in shN.items():
        sn[i] = v
    np.savez(os.path.join(RUN, 'tracks.npz'), birth=b, death=d, shN=sn, **out)
    log.update(N_total=Ng, mean_psnr=float(np.mean(log['psnr'])), mean_psnr_fg=float(np.mean(log['psnr_fg'])),
               secs_total=time.time() - t0)
    json.dump(log, open(os.path.join(RUN, 'e3.json'), 'w'), indent=1)
    print('E3 OK', RUN, f'ids {Ng}, mean PSNR {log["mean_psnr"]:.2f} dB, fg {log["mean_psnr_fg"]:.2f} dB')


if __name__ == '__main__':
    main()
