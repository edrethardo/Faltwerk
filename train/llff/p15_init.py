#!/usr/bin/env python3
"""frame-0 fit of an N3DV scene ON TOP OF the P12 short fit (s3k / s3k_reg), with the static-foundation
measures as switches (GPU, train venv). Every switch defaults to "off", so without P15 switches this is P12's s3k run
(p12_init.fit with --iters 3000 --sh-every 500, same renderer, loss, near plane, COLMAP init, lr table).

  E3_GT=<gt_dir> python p15_init.py --out <dir> [P12 switches] [P15 switches]
  P15 switches
    --depth none|sparse|mono   depth guidance: COLMAP sparse depth of the train cams (<gt>/sparse_depth.npz, p15_prep.py
                               sparse; L1 on log depth at the observed keypoints) or monocular depth (Depth Anything V2,
                               scale/shift fitted per camera to the COLMAP sparse inverse depth, <gt>/mono_invdepth.npz,
                               p15_prep.py mono; L1 on inverse depth normalised by its per-image median)
    --depth-w <w>              weight, decays exponentially to w*--depth-final over the fit (3DGS depth regularisation)
    --expo                     per-TRAIN-camera affine colour/exposure correction (3x3 + offset, identity init) applied to
                               the render before the loss; cam00 is never corrected (evaluation without it). The
                               learned matrices are saved as <out>/expo.pt for the sequence run (n3dv_train P15_EXPO).
    --val-cam auto|<idx>       early stop over a held-out TRAIN camera (never cam00): the camera is removed from the
                               fit, its PSNR is checked every --val-every steps, the best state is kept; auto = the train
                               camera nearest to the centroid of the train cameras (no test-camera information)
    --patience <steps>         stop when the val PSNR has not improved for that many steps (0 = run all --iters)
    --val-refit                after the val run: refit on ALL train cams with the same schedule, stopped at the best
                               step of the val run (the val cam is used for the stopping time only)
    --aniso <r> --aniso-w <w>  penalty w*mean(relu(max_scale/min_scale - r)) (PhysGaussian-style anisotropy limit)
    --max-scale-rel <f>        hard limit: largest axis <= f * distance to the train-camera centroid (after each step)
    --bg <N> --bg-radius <R>   background set: N splats on a sphere of radius R (default: median LLFF far bound of the
                               train cams) around the camera centroid, coloured from random train pixels
Output like p12_init: <out>/init.pt, fit.json (+ val curve, best step, exposure stats), cam00.png (stays on the GPU box).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
import n3dv_train as nt                                     # noqa: E402  COLMAP init, near plane (patches e3)
import e3_track as e3                                       # noqa: E402
import p12_init as p12                                      # noqa: E402
from gsplat.strategy import DefaultStrategy                 # noqa: E402
from gsplat.strategy.ops import remove as gs_remove         # noqa: E402

dev = e3.dev


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=0)
    # P12 (short plan) switches, defaults = s3k
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--sh', type=int, default=3)
    ap.add_argument('--sh-every', type=int, default=500)
    ap.add_argument('--cams', type=int, default=4)
    ap.add_argument('--lr-scale', type=float, default=1.0)
    ap.add_argument('--refine-stop', type=float, default=0.75)
    ap.add_argument('--reset-every', type=int, default=3000)
    ap.add_argument('--prune-opa', type=float, default=0.005)
    ap.add_argument('--grow', type=float, default=0.0002)
    ap.add_argument('--scale-reg', type=float, default=0.0)
    ap.add_argument('--opa-reg', type=float, default=0.0)
    ap.add_argument('--eval-every', type=int, default=500, help='cam00 PSNR curve (logged only, never used to stop)')
    # P15
    ap.add_argument('--depth', default='none', choices=['none', 'sparse', 'mono'])
    ap.add_argument('--depth-w', type=float, default=0.0)
    ap.add_argument('--depth-final', type=float, default=0.01)
    ap.add_argument('--expo', action='store_true')
    ap.add_argument('--expo-lr', type=float, default=1e-3)
    ap.add_argument('--val-cam', default='')
    ap.add_argument('--val-every', type=int, default=250)
    ap.add_argument('--val-start', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=0)
    ap.add_argument('--val-refit', action='store_true')
    ap.add_argument('--aniso', type=float, default=0.0)
    ap.add_argument('--aniso-w', type=float, default=0.1)
    ap.add_argument('--max-scale-rel', type=float, default=0.0)
    ap.add_argument('--bg', type=int, default=0)
    ap.add_argument('--bg-radius', type=float, default=0.0)
    ap.add_argument('--near-prune', type=float, default=0.0, help='prune splats in a train frustum closer than f x its LLFF '
                    'near bound (every 100 steps + at the end; owner finding: ghosts right in front of the cameras)')
    ap.add_argument('--vis-prune', action='store_true', help='also prune splats seen (in frustum, beyond near) by <= 2 train '
                    'cams that lie closer than 2 x near to some train cam')
    ap.add_argument('--eval-only', action='store_true', help='recompute the metrics of <out>/init.pt into <out>/fit.json '
                    '(adds LPIPS when the first run had no lpips package)')
    a = ap.parse_args(argv)
    a.init, a.n_random, a.strategy = 'colmap', 0, 'default'   # what p12_init.init_points reads
    return a


def centres(ids):
    return np.array([-(np.array(e3.cams[i]['R']).T @ np.array(e3.cams[i]['t'])) for i in ids])


def auto_val_cam():
    tr = e3.train
    c = centres(tr)
    return int(tr[int(np.argmin(np.linalg.norm(c - c.mean(0), axis=1)))])


def bg_points(n, radius, rng):
    """n points on a sphere (radius around the train-camera centroid), hit by rays through random train pixels."""
    tr = e3.train
    c0 = centres(tr).mean(0)
    ims = {i: np.asarray(Image.open(f'{e3.GT}/f000/c{i:02d}.png'))[..., :3] for i in tr}
    ci = rng.choice(tr, n)
    u = rng.uniform(0, e3.W, n); v = rng.uniform(0, e3.H, n)
    X = np.zeros((n, 3)); col = np.zeros((n, 3))
    for i in np.unique(ci):
        s = ci == i; c = e3.cams[i]; R = np.array(c['R']); o = -(R.T @ np.array(c['t']))
        d = np.stack([(u[s] - c['cx']) / c['fx'], (v[s] - c['cy']) / c['fy'], np.ones(s.sum())], 1) @ R   # world dirs
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        oc = o - c0
        b = d @ oc; cc = oc @ oc - radius ** 2
        t = -b + np.sqrt(np.maximum(b * b - cc, 0))                   # far intersection (camera inside the sphere)
        X[s] = o + t[:, None] * d
        col[s] = ims[i][np.minimum(v[s].astype(int), e3.H - 1), np.minimum(u[s].astype(int), e3.W - 1)] / 255.0
    return X.astype(np.float32), col.astype(np.float32)


def init_points(gt0, a, rng):
    p, ext, n_col = p12.init_points(gt0, a, rng)
    n_bg = 0
    if a.bg > 0:
        R = a.bg_radius or float(np.median([e3.cams[i]['far'] for i in e3.train]))
        X, col = bg_points(a.bg, R, rng)
        n_bg = len(X)
        allp = np.concatenate([p['means'].cpu().numpy(), X])
        sc = nt.knn_scale(allp)
        n = len(allp)
        q = {'means': allp, 'quats': np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
             'scales': np.repeat(sc[:, None], 3, 1), 'opacities': np.full((n,), -2.0, np.float32),
             'sh0': np.concatenate([p['sh0'].cpu().numpy(), ((col - 0.5) / 0.28209479)[:, None]]),
             'shN': np.zeros((n, p['shN'].shape[1], 3), np.float32)}
        p = {k: torch.tensor(np.asarray(v, np.float32), device=dev) for k, v in q.items()}
        print(f'init: + {n_bg} background sphere points (radius {R:.1f})', flush=True)
    return p, ext, n_col, n_bg


def load_depth(a):
    if a.depth == 'sparse':
        d = np.load(os.path.join(e3.GT, 'sparse_depth.npz'))
        out = {}
        for i in e3.train:
            k = f'c{i:02d}'
            if k + '_uv' in d.files:
                out[i] = (torch.tensor(d[k + '_uv'], device=dev), torch.tensor(d[k + '_z'], device=dev))
        print(f'sparse depth: {sum(len(v[1]) for v in out.values())} keypoints in {len(out)} train cams', flush=True)
        return out
    if a.depth == 'mono':
        d = np.load(os.path.join(e3.GT, 'mono_invdepth.npz'))
        out = {i: torch.tensor(d[f'c{i:02d}'].astype(np.float32), device=dev) for i in e3.train if f'c{i:02d}' in d.files}
        print(f'mono inverse depth for {len(out)} train cams', flush=True)
        return out
    return None


def render_rgbd(P, ids, deg):
    col = torch.cat([P['sh0'], P['shN']], 1)[:, :(deg + 1) ** 2]
    out, alpha, info = e3.rasterization(P['means'], P['quats'], torch.exp(P['scales']), torch.sigmoid(P['opacities']),
                                        col, e3.VM[ids], e3.K[ids], e3.W, e3.H, sh_degree=deg, packed=False,
                                        render_mode='RGB+ED')
    return out[..., :3].clamp(0, 1), out[..., 3], alpha, info


def apply_expo(rgb, E, ids, ref=None):
    """rgb [C,H,W,3]; E [ncam,3,4] (zeros = identity); per camera rgb @ (I + E[:, :3])^T + E[:, 3].
    ref: train-camera indices whose mean correction is removed (gauge: without it a common colour shift of all train
    cameras is free and the model colours drift away from the uncorrected test camera). Elementwise multiply-sum: the
    einsum/bmm form of the 3x3 map made a training step 3.5x slower (p15_probe_expo.py: 589 vs 173 ms)."""
    Ei = E[torch.as_tensor([int(i) for i in ids], device=rgb.device)]
    if ref is not None:
        Ei = Ei - E[torch.as_tensor([int(i) for i in ref], device=rgb.device)].mean(0, keepdim=True)
    A = torch.eye(3, device=rgb.device) + Ei[:, :, :3]
    return ((rgb[..., None, :] * A[:, None, None]).sum(-1) + Ei[:, None, None, :, 3]).clamp(0, 1)


def gauge_fixed(E, ref):
    """E with the mean over the train cameras ref removed from their rows; all other rows (test cams) zero."""
    out = torch.zeros_like(E)
    r = torch.as_tensor([int(i) for i in ref], device=E.device)
    out[r] = E[r] - E[r].mean(0, keepdim=True)
    return out


def depth_loss(a, D, ids, dep):
    tot, n = 0.0, 0
    for k, i in enumerate(ids):
        if i not in dep:
            continue
        if a.depth == 'sparse':
            uv, z = dep[i]
            g = torch.stack([uv[:, 0] / e3.W * 2 - 1, uv[:, 1] / e3.H * 2 - 1], 1)[None, None]
            dr = F.grid_sample(D[k][None, None], g, align_corners=False).view(-1)
            ok = dr > 1e-3
            if ok.any():
                tot = tot + (torch.log(dr[ok]) - torch.log(z[ok])).abs().mean(); n += 1
        else:
            m = dep[i]
            inv = 1.0 / D[k].clamp_min(nt.NEAR)
            tot = tot + ((inv - m).abs().mean() / m.median().clamp_min(1e-6)); n += 1
    return tot / max(n, 1)


def floater_mask(means, train, f, vis):
    """bool [N]: splats inside a train frustum closer than f*near (f>0) and, with vis, splats seen by <= 2 train cams
    (in frustum beyond near) that are closer than 2*near to some train camera."""
    near = torch.zeros(means.shape[0], dtype=torch.bool, device=means.device)
    near2 = near.clone(); seen = torch.zeros(means.shape[0], dtype=torch.int32, device=means.device)
    for i in train:
        c = e3.cams[i]
        Xc = means @ torch.tensor(c['R'], device=means.device, dtype=torch.float32).T + torch.tensor(c['t'], device=means.device, dtype=torch.float32)
        z = Xc[:, 2]; zc = z.clamp_min(1e-6)
        u = c['fx'] * Xc[:, 0] / zc + c['cx']; v = c['fy'] * Xc[:, 1] / zc + c['cy']
        inimg = (z > 0) & (u >= 0) & (u < e3.W) & (v >= 0) & (v < e3.H)
        if f > 0:                                            # in the cone in front of the camera, or in a ball around it
            cen = -(torch.tensor(c['R'], device=means.device, dtype=torch.float32).T @ torch.tensor(c['t'], device=means.device, dtype=torch.float32))
            near |= (inimg & (z < f * c['near'])) | ((means - cen).norm(dim=1) < f * c['near'])
        near2 |= inimg & (z < 2 * c['near'])
        seen += (inimg & (z >= c['near'])).int()
    m = near
    if vis:   # supervised by <= 2 train views and close to the rig, or not seen beyond near by any train view at all
        m = m | (near2 & (seen <= 2)) | (seen == 0)
    return m


def fit(p, gt0, ext, a, log, train, stop_at=None, val=None, dep=None):
    """P12 fit (DefaultStrategy, short schedule) + P15 measures. Returns (params padded to SH3, E or None, best_it)."""
    P = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in p.items()})
    lrs = e3.lr_table(ext, a.lr_scale)
    opts = {k: torch.optim.Adam([P[k]], lr=lrs[k], eps=1e-15) for k in P}
    E = None
    if a.expo:
        E = torch.nn.Parameter(torch.zeros((len(e3.cams), 3, 4), device=dev))
        eopt = torch.optim.Adam([E], lr=a.expo_lr)
    strat = DefaultStrategy(refine_stop_iter=int(a.iters * a.refine_stop),
                            reset_every=a.reset_every if a.reset_every > 0 else 10 ** 9,
                            prune_opa=a.prune_opa, grow_grad2d=a.grow, verbose=False)
    strat.check_sanity(P, opts)
    state = strat.initialize_state(scene_scale=ext / 2)
    sched = torch.optim.lr_scheduler.ExponentialLR(opts['means'], gamma=0.01 ** (1.0 / a.iters))
    c0 = torch.tensor(centres(train).mean(0), dtype=torch.float32, device=dev)
    test_gt = gt0[e3.test]
    need_d = dep is not None and a.depth_w > 0
    best, bestE, best_it, best_ps = None, None, 0, -1.0
    ts = time.time()
    n_it = a.iters if stop_at is None else stop_at
    for it in range(n_it):
        deg = min(a.sh, it // a.sh_every)
        cam = list(np.random.choice(train, a.cams, replace=False))
        if need_d:
            rgb, D, _, info = render_rgbd(P, cam, deg)
        else:
            rgb, _, info = e3.render(P, cam, deg)
        strat.step_pre_backward(P, opts, state, it, info)
        if E is not None:
            rgb = apply_expo(rgb, E, cam, ref=train)
        loss = e3.loss_fn(rgb, gt0[cam])
        if need_d:
            w = a.depth_w * a.depth_final ** (it / max(a.iters - 1, 1))
            loss = loss + w * depth_loss(a, D, cam, dep)
        if a.scale_reg > 0:
            loss = loss + a.scale_reg * torch.exp(P['scales']).mean()
        if a.opa_reg > 0:
            loss = loss + a.opa_reg * torch.sigmoid(P['opacities']).mean()
        if a.aniso > 0:
            s = torch.exp(P['scales'])
            loss = loss + a.aniso_w * torch.relu(s.max(1).values / s.min(1).values.clamp_min(1e-12) - a.aniso).mean()
        loss.backward()
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        if E is not None:
            eopt.step(); eopt.zero_grad(set_to_none=True)
        sched.step()
        strat.step_post_backward(P, opts, state, it, info, packed=False)
        if (a.near_prune > 0 or a.vis_prune) and (it + 1) % 100 == 0:
            with torch.no_grad():
                fm = floater_mask(P['means'], train, a.near_prune, a.vis_prune)
            if fm.any():
                gs_remove(params=P, optimizers=opts, state=state, mask=fm)
                log['n_pruned_floaters'] = log.get('n_pruned_floaters', 0) + int(fm.sum())
        if a.max_scale_rel > 0:
            with torch.no_grad():
                lim = torch.log(a.max_scale_rel * (P['means'] - c0).norm(dim=1).clamp_min(1e-6))
                P['scales'].data = torch.minimum(P['scales'].data, lim[:, None])
        if it % 1000 == 0:
            print(f'  it {it} N={P["means"].shape[0]} loss {loss.item():.4f} {(it + 1) / (time.time() - ts):.1f} it/s', flush=True)
        if val is not None and (it + 1) % a.val_every == 0 and it + 1 >= a.val_start:
            with torch.no_grad():
                r, _, _ = e3.render(P, [val])
                r = torch.round(r * 255) / 255
                vp = e3.psnr(r, gt0[val:val + 1])
            log['val_curve'].append({'it': it + 1, 'N': int(P['means'].shape[0]), 'psnr_val': round(vp, 3)})
            if vp > best_ps:
                best_ps, best_it = vp, it + 1
                best = {k: v.detach().clone() for k, v in P.items()}
                bestE = None if E is None else gauge_fixed(E.detach(), train)
            print(f'  val it {it + 1}: {e3.cams[val]["name"]} {vp:.2f} dB (best {best_ps:.2f} @ {best_it})', flush=True)
            if a.patience and it + 1 - best_it >= a.patience:
                print(f'  early stop at {it + 1} (no val gain for {a.patience} steps)', flush=True)
                break
        if a.eval_every and ((it + 1) % a.eval_every == 0 or it + 1 == n_it):
            with torch.no_grad():
                r, _, _ = e3.render(P, e3.test)
                ps = e3.psnr(r, test_gt)
            log['curve'].append({'it': it + 1, 'N': int(P['means'].shape[0]), 'psnr_test': round(ps, 3),
                                 'loss': round(loss.item(), 5), 'secs': round(time.time() - ts, 1)})
            print(f'  eval it {it + 1}: cam00 {ps:.2f} dB N={P["means"].shape[0]} (log only)', flush=True)
    if best is not None:
        out, Eo = best, bestE
    else:
        out, Eo = {k: v.detach().contiguous() for k, v in P.items()}, (None if E is None else gauge_fixed(E.detach(), train))
    if out['shN'].shape[1] < 15:
        out['shN'] = torch.cat([out['shN'], torch.zeros((out['shN'].shape[0], 15 - out['shN'].shape[1], 3), device=dev)], 1)
    if a.near_prune > 0 or a.vis_prune:                      # final pass (the kept state may predate the last prune)
        keep = ~floater_mask(out['means'], train, a.near_prune, a.vis_prune)
        out = {k: v[keep] for k, v in out.items()}
    return {k: v.contiguous() for k, v in out.items()}, Eo, best_it


def main(argv=None):
    a = parse(argv)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    gt0 = e3.load_frame(0)
    if a.eval_only:
        fj = os.path.join(a.out, 'fit.json')
        log = json.load(open(fj))
        q = torch.load(os.path.join(a.out, 'init.pt'), map_location=dev)['p']
        m = p12.metrics(q, gt0, a.out)
        log['metrics_recomputed'] = {k: m[k] for k in ('psnr_test', 'ssim_test', 'psnr_train_mean') if k in m}
        log.update({k: v for k, v in m.items() if k.startswith('lpips')})
        if 'lpips_vgg_test' in m:
            log.pop('lpips_note', None)
        log['metrics_recomputed_gpu'] = torch.cuda.get_device_name(0)
        json.dump(log, open(fj, 'w'), indent=1)
        print(f'P15 EVAL OK {a.out}: cam00 {m["psnr_test"]:.2f} (fit.json {log["psnr_test"]:.2f}) '
              f'lpips_vgg {m.get("lpips_vgg_test", float("nan")):.4f}', flush=True)
        return
    p, ext, n_col, n_bg = init_points(gt0, a, rng)
    dep = load_depth(a)
    log = {'config': vars(a), 'gt': e3.GT, 'ext': ext, 'near': nt.NEAR, 'n_init': int(p['means'].shape[0]), 'n_colmap': n_col,
           'n_bg': n_bg, 'train_cams': len(e3.train), 'test_cams': [e3.cams[i]['name'] for i in e3.test], 'curve': [],
           'val_curve': []}
    train = list(e3.train)
    if a.val_cam:
        val = auto_val_cam() if a.val_cam == 'auto' else int(a.val_cam)
        assert val in e3.train and val not in e3.test, 'the validation camera must be a TRAIN camera (never cam00)'
        log['val_cam'] = e3.cams[val]['name']
        print(f'val cam {e3.cams[val]["name"]} (held out of the fit, used for the stopping time only)', flush=True)
        q, E, best_it = fit(p, gt0, ext, a, log, [i for i in train if i != val], val=val, dep=dep)
        log['best_it'] = best_it
        log['iters_run'] = log['val_curve'][-1]['it'] if log['val_curve'] else a.iters
        if a.val_refit:
            torch.manual_seed(a.seed); np.random.seed(a.seed)
            log['curve_valrun'], log['curve'] = log['curve'], []
            q, E, _ = fit(p, gt0, ext, a, log, train, stop_at=best_it, dep=dep)
            log['iters_effective'] = best_it
        else:
            log['iters_effective'] = best_it
    else:
        q, E, _ = fit(p, gt0, ext, a, log, train, dep=dep)
        log['iters_effective'] = a.iters
    secs = time.time() - t0
    torch.save({'p': q, 'ext': ext, 'secs': secs}, os.path.join(a.out, 'init.pt'))
    m = p12.metrics(q, gt0, a.out)                          # cam00 without exposure correction; train PSNR raw
    if E is not None:
        torch.save({'E': E.cpu(), 'train': train, 'note': 'rgb_train = rgb @ (I+E[:, :, :3])^T + E[:, :, 3]'}, os.path.join(a.out, 'expo.pt'))
        with torch.no_grad():
            tr = [e3.psnr(apply_expo(e3.render(q, [c])[0], E, [c]), gt0[c:c + 1]) for c in train]
        m['psnr_train_mean_expo'] = float(np.mean(tr))
        m['expo'] = {'mean_abs_gain_offdiag': float(E[train, :, :3].abs().mean()),
                     'gain_diag_per_cam': [[round(1 + float(E[c, j, j]), 4) for j in range(3)] for c in train],
                     'offset_per_cam': [[round(float(E[c, j, 3]), 4) for j in range(3)] for c in train],
                     'test_rows_zero': bool((E[e3.test] == 0).all())}
    log.update(m, N=int(q['means'].shape[0]), secs=round(secs, 1), gpu=torch.cuda.get_device_name(0))
    json.dump(log, open(os.path.join(a.out, 'fit.json'), 'w'), indent=1)
    print(f'P15 INIT OK {a.out}: N={log["N"]} cam00 {log["psnr_test"]:.2f} dB SSIM {log["ssim_test"]:.4f} '
          f'train {log["psnr_train_mean"]:.2f} dB lpips_vgg {log.get("lpips_vgg_test", float("nan")):.4f} '
          f'iters {log["iters_effective"]} ({secs:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
