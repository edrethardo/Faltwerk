#!/usr/bin/env python3
"""Rig variant of train/e3_track.py (matted multi-camera captures): GT dir from $P8_GT, non-square W x H from cameras.json, frame-0
init from a visual hull of the GT alpha masks (no depth for real/generated video). Otherwise unchanged.

E3 – Our own 4DGS reconstruction on the E2 ground truth (GPU, train venv, gsplat).

Model (v3): a pool of splats, each with a lifetime [birth, death) in frames.
  * frame 0: full 3DGS fit from back-projected depth (SH3, DefaultStrategy densification, L1 + 0.2 D-SSIM).
  * every REFRESH frames: densify/prune phase on that frame (all attributes). Splats the strategy removes die;
    clones/splits get new ids and are born -> lifetimes, like a per-splat life window.
  * other frames: warm start from t-1, track per-frame means, quats, scales, opacity and colour DC (sh0);
    higher SH bands (shN) stay as fitted at the last refresh.
Black background (premultiplied RGBA GT, black outside the matte).
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
SMOKE = os.environ.get('SMOKE') == '1'
GT = os.environ['P8_GT']
RUN = os.environ.get('RUN', os.path.join(ROOT, 'runs', 'p8', 'e3', time.strftime('%Y%m%d-%H%M%S')))
N_INIT = int(os.environ.get('E3_N', '20000' if SMOKE else '100000'))
IT0 = int(os.environ.get('E3_IT0', '600' if SMOKE else '20000'))
ITT = int(os.environ.get('E3_ITT', '20' if SMOKE else '300'))
ITR = int(os.environ.get('E3_ITR', '200' if SMOKE else '1500'))
REFRESH = int(os.environ.get('E3_REFRESH', '2' if SMOKE else '8'))
SH = int(os.environ.get('E3_SH', '3'))
REG = float(os.environ.get('E3_REG', '0'))
# A matted rig GT is RGBA (RGB premultiplied by the matte, black outside). Against black only, dark
# opaque splats cost nothing in the train views and occlude the subject in new views. Both switches default off (= P8).
ALPHA_W = float(os.environ.get('P8_ALPHA', '0'))           # L1(rendered alpha, GT matte) weight
RANDBG = os.environ.get('P8_RANDBG', '0') in ('1', 'grey')  # random background colour per camera and step
RANDBG_GREY = os.environ.get('P8_RANDBG', '0') == 'grey'    # random GREY level (no hue) instead
RANDBG_P = float(os.environ.get('P8_RANDBG_P', '1'))         # share of steps with the random background (else black)
HEAD_W = float(os.environ.get('P8_HEAD_W', '0'))           # extra loss on the head band (face band)
HAND_W = float(os.environ.get('P8_HAND_W', '0'))           # P40: extra L1 on the hands (skin below the head), 0 = off
USE_ALPHA = ALPHA_W > 0 or RANDBG
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
W, H = meta['W'], meta['H']


def load_frame(t, rgba=None):
    """[C,H,W,3] (or [C,H,W,4] with the matte when rgba / USE_ALPHA) in [0,1]."""
    nc = 4 if (USE_ALPHA if rgba is None else rgba) else 3
    ims = [np.asarray(Image.open(f'{GT}/f{t:03d}/c{i:02d}.png'))[..., :nc] for i in range(len(cams))]
    return torch.tensor(np.stack(ims), device=dev).float() / 255.0


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


def train_loss(rgb, alpha, g):
    """rgb [C,H,W,3] (render on black = premultiplied), alpha [C,H,W,1], g [C,H,W,3|4] (RGB premultiplied, matte).
    With P8_RANDBG both sides are composited on the same random colour per camera, so splats that are opaque where the
    matte is 0 show up whatever their colour; P8_ALPHA adds L1(alpha, matte). Both off: exactly loss_fn on black."""
    gr = g[..., :3]
    if g.shape[-1] == 4 and RANDBG and (RANDBG_P >= 1 or float(torch.rand(())) < RANDBG_P):
        bg = torch.rand((rgb.shape[0], 1, 1, 1 if RANDBG_GREY else 3), device=rgb.device).expand(-1, -1, -1, 3)
        rgb = (rgb + (1 - alpha) * bg).clamp(0, 1)
        gr = gr + (1 - g[..., 3:]) * bg
    loss = loss_fn(rgb, gr)
    if g.shape[-1] == 4 and ALPHA_W > 0:
        loss = loss + ALPHA_W * (alpha - g[..., 3:]).abs().mean()
    if g.shape[-1] == 4 and HEAD_W > 0:                    # face/head band: top 15 % of the matte bbox, own loss term
        for c in range(len(g)):
            m = g[c, ..., 3] > 0.5
            ys = torch.nonzero(m.any(1)).squeeze(1)
            if len(ys) < 8:
                continue
            y0 = int(ys[0]); y1 = y0 + max(int(0.15 * (int(ys[-1]) - y0)), 8)
            xs = torch.nonzero(m[y0:y1].any(0)).squeeze(1)
            if len(xs) < 8:
                continue
            x0, x1 = max(int(xs[0]) - 12, 0), int(xs[-1]) + 13
            y0 = max(y0 - 12, 0); y1 = y1 + 12
            loss = loss + HEAD_W * loss_fn(rgb[c:c + 1, y0:y1, x0:x1], gr[c:c + 1, y0:y1, x0:x1])
    if g.shape[-1] == 4 and HAND_W > 0:                    # P40: hands = skin colour below 20 % of the matte bbox
        loss = loss + HAND_W * hand_l1(rgb, gr, g[..., 3])
    return loss


def hand_mask(gr, ga):
    """[C,H,W] bool: GT skin colour (YCrCb) inside the matte, below the top 20 % of the
    matte bbox (face/neck excluded), dilated 8 px (the render's smeared hand counts too)."""
    r, g_, b = gr[..., 0] * 255, gr[..., 1] * 255, gr[..., 2] * 255
    y = 0.299 * r + 0.587 * g_ + 0.114 * b
    cr, cb = (r - y) * 0.713 + 128, (b - y) * 0.564 + 128
    m = (ga > 0.5) & (y > 70) & (cr >= 136) & (cr <= 180) & (cb >= 77) & (cb <= 135)
    rows = torch.arange(ga.shape[1], device=ga.device)
    for c in range(len(ga)):
        ys = torch.nonzero((ga[c] > 0.5).any(1)).squeeze(1)
        if len(ys) < 8:
            m[c] = False
            continue
        m[c] &= (rows > ys[0] + 0.2 * (ys[-1] - ys[0]))[:, None]
    return torch.nn.functional.max_pool2d(m.float()[:, None], 17, stride=1, padding=8)[:, 0] > 0


def hand_l1(rgb, gr, ga):
    m = hand_mask(gr, ga)
    n = m.sum()
    if int(n) < 50:
        return rgb.sum() * 0.0
    return ((rgb - gr).abs().mean(-1) * m).sum() / n


def alpha_outside(alpha, ga, dil=8):
    """floater measures for one view: alpha [H,W] rendered, ga [H,W] GT matte. Returns (out, out_d, px_d):
    out   = rendered alpha mass where the matte < 0.5, relative to the matte's foreground area,
    out_d = the same outside the matte dilated by `dil` px (ignores edge softness: real floaters),
    px_d  = pixels with rendered alpha > 0.5 outside the dilated matte, relative to the foreground area."""
    m = (ga > 0.5).float()
    md = Fn.max_pool2d(m[None, None], 2 * dil + 1, stride=1, padding=dil)[0, 0]
    fg = m.sum().clamp_min(1)
    return (float((alpha * (1 - m)).sum() / fg), float((alpha * (1 - md)).sum() / fg),
            float(((alpha > 0.5).float() * (1 - md)).sum() / fg))


def init_points(gt0):
    """visual hull of frame 0: voxel centres whose projection lies inside the GT alpha mask (> 0.5) in every train
    view that sees them (>= 8 views); colour = mean GT colour over those views (occlusion ignored, training fixes it)."""
    masks = {i: torch.tensor(np.asarray(Image.open(f'{GT}/f000/c{i:02d}.png'))[..., 3] > 127, device=dev) for i in train}
    lo, hi = np.array(meta['bbox_lo'], np.float32), np.array(meta['bbox_hi'], np.float32)
    n_ax = 160
    ax = [torch.linspace(float(lo[d]), float(hi[d]), n_ax, device=dev) for d in range(3)]
    X = torch.stack(torch.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    inside = torch.zeros(len(X), device=dev); seen = torch.zeros(len(X), device=dev); col = torch.zeros(len(X), 3, device=dev)
    for i in train:
        c = cams[i]
        Xc = X @ torch.tensor(c['R'], device=dev).T + torch.tensor(c['t'], device=dev)
        u = (Xc[:, 0] / Xc[:, 2] * c['fx'] + c['cx']).long(); v = (Xc[:, 1] / Xc[:, 2] * c['fy'] + c['cy']).long()
        ok = (Xc[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        uu, vv = u.clamp(0, W - 1), v.clamp(0, H - 1)
        m = masks[i][vv, uu] & ok
        seen += ok.float(); inside += m.float(); col += gt0[i][vv, uu, :3] * m[:, None].float()
    keep = (seen >= 8) & (inside >= seen - 0.5)
    pts, cols = X[keep], col[keep] / inside[keep, None].clamp_min(1)
    pts = pts + (torch.rand_like(pts) - 0.5) * (torch.tensor(hi - lo, device=dev) / n_ax)
    print(f'  visual hull: {int(keep.sum())} of {len(X)} voxels', flush=True)
    g = torch.Generator(device='cpu').manual_seed(0)
    sel = torch.randperm(len(pts), generator=g)[:N_INIT].to(dev)
    pts, cols = pts[sel].cpu().numpy(), cols[sel].cpu().numpy()
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
        rgb, alpha, info = render(P, cam, deg)
        strat.step_pre_backward(P, opts, state, it, info)
        loss = train_loss(rgb, alpha, gt[cam])
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
        rgb, alpha, _ = render(p, cam)
        loss = train_loss(rgb, alpha, gt[cam])
        if REG > 0:
            loss = loss + REG * sum(((p[k] - prev[k]).abs() / norm[k]).mean() for k in TRACKED)
        opt.zero_grad(); loss.backward(); opt.step()
    for k in p: p[k].requires_grad_(False)


def evaluate(p, gt):
    with torch.no_grad():
        rgb, _, _ = render(p, test)
        g = gt[test][..., :3]
        return psnr(rgb, g), psnr_fg(rgb, g), rgb


def preview(gt, rgb, path):
    im = torch.cat([torch.cat([gt[test][i][..., :3], rgb[i]], 1) for i in range(len(test))], 0)
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
