#!/usr/bin/env python3
"""frame-0 fit of an N3DV scene with switches for the init/densification ablation (GPU, train venv).

The P6 frame-0 fit (e3.fit_densify first=True, COLMAP sparse init) reaches only 24.5 / 28.8 dB on the held-out cam00
while train views sit at 36-39 dB. This fitter keeps everything of e3/n3dv_train (renderer, loss, near plane, COLMAP
init, 3-NN scales, ext = 2 x 3DGS cameras_extent) and exposes the factors of the ablation:

  E3_GT=<gt_dir> python p12_init.py --out <dir> [options]
    --strategy default|absgrad|mcmc  densification: gsplat DefaultStrategy (P6), AbsGS gradients (grow 8e-4) or
                                     MCMCStrategy with --cap splats (3DGS-MCMC, Kheradmand et al. 2024)
    --grow <g>                       grow_grad2d threshold (default 2e-4 / absgrad 8e-4)
    --cap <N>                        MCMC: maximum number of splats (default 300000)
    --init colmap|colmap+random      COLMAP sparse points (P6) or those plus --n-random points sampled in the union of
                                     the train frusta between the LLFF near/far bounds (seen by >= 2 train cams)
    --n-random <N>                   default 100000
    --sh <d>                         SH degree (P6: 3)
    --iters <N>                      optimisation steps (P6: 20000)
    --cams <n>                       train cameras per step (P6: 4; 3DGS: 1)
    --lr-scale <f>                   factor on the means lr 1.6e-4*ext (P6: 1.0; 3DGS = 1.6e-4*cameras_extent -> 0.5)
    --refine-stop <f>                fraction of iters where densification stops (P6: 0.75; 3DGS 15000/30000 = 0.5)
    --eval-every <n>                 cam00 PSNR every n steps (curve in fit.json; default 2000)
    --seed <s>
Output: <out>/init.pt (same format as e7's cache: {'p', 'ext', 'secs'}, usable as E7_RUN/init.pt for the 64-frame run),
<out>/fit.json (config, cam00 PSNR/SSIM/LPIPS(alex, vgg; if the lpips package is importable), mean PSNR over the train
cams, N, time, PSNR curve), <out>/cam00.png (GT | render, half res; N3DV is CC BY-NC - stays on the GPU box).
"""
import argparse, functools, json, math, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
import n3dv_train as nt                                     # noqa: E402  patches e3 (COLMAP init, near plane)
import e3_track as e3                                       # noqa: E402
from gsplat.strategy import DefaultStrategy, MCMCStrategy   # noqa: E402

dev = e3.dev


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--strategy', default='default', choices=['default', 'absgrad', 'mcmc'])
    ap.add_argument('--grow', type=float, default=None)
    ap.add_argument('--cap', type=int, default=300000)
    ap.add_argument('--init', default='colmap', choices=['colmap', 'colmap+random'])
    ap.add_argument('--n-random', type=int, default=100000)
    ap.add_argument('--sh', type=int, default=3)
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--cams', type=int, default=4)
    ap.add_argument('--lr-scale', type=float, default=1.0)
    ap.add_argument('--refine-stop', type=float, default=0.75)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    # overfit countermeasures (the cam00 PSNR peaks after ~2000 steps and falls while train PSNR keeps rising)
    ap.add_argument('--scale-reg', type=float, default=0.0, help='weight of mean(exp(scales)) in the loss (all strategies)')
    ap.add_argument('--opa-reg', type=float, default=0.0, help='weight of mean(sigmoid(opacities)) in the loss (all strategies)')
    ap.add_argument('--reset-every', type=int, default=3000, help='opacity reset period (default strategy; 0 = never)')
    ap.add_argument('--prune-opa', type=float, default=0.005, help='default strategy: prune splats below this opacity')
    ap.add_argument('--sh-every', type=int, default=1000, help='steps per SH degree increment (P6: 1000)')
    ap.add_argument('--shn-lr', type=float, default=1.0, help='factor on the lr of the higher SH bands (P6: 1)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--eval-init', default=None, help='no fit: load this init.pt (e7 format) and write its metrics (reference row)')
    return ap.parse_args(argv)


def random_frustum_points(n, rng, colmap_xyz=None):
    """n points: random train cam, random pixel, depth log-uniform in the depth range the COLMAP points span in that
    cam (2nd..98th percentile; LLFF near/far bounds if no points given), kept if >= 2 train cams see the point (inside
    image, z > near). Colour = GT pixel of the generating cam at frame 0."""
    cams = e3.cams
    tr = e3.train
    pts, cols = [], []
    ims = {i: np.asarray(Image.open(f'{e3.GT}/f000/c{i:02d}.png'))[..., :3] for i in tr}
    zr = {}
    for i in tr:
        c = cams[i]; lo, hi = c['near'], c['far']
        if colmap_xyz is not None:
            Xc = colmap_xyz @ np.array(c['R']).T + np.array(c['t'])
            uu = c['fx'] * Xc[:, 0] / Xc[:, 2] + c['cx']; vv = c['fy'] * Xc[:, 1] / Xc[:, 2] + c['cy']
            inv = (Xc[:, 2] > 0) & (uu >= 0) & (uu < e3.W) & (vv >= 0) & (vv < e3.H)
            if inv.sum() > 50:
                lo, hi = np.percentile(Xc[inv, 2], [2, 98])
        zr[i] = (max(lo, c['near']), hi)
    want = n
    while sum(len(p) for p in pts) < want:
        m = 4 * n
        ci = rng.choice(tr, m)
        u = rng.uniform(0, e3.W, m); v = rng.uniform(0, e3.H, m)
        near = np.array([zr[i][0] for i in ci]); far = np.array([zr[i][1] for i in ci])
        z = np.exp(rng.uniform(np.log(near), np.log(far)))
        X = np.zeros((m, 3))
        for i in np.unique(ci):
            s = ci == i; c = cams[i]
            Xc = np.stack([(u[s] - c['cx']) / c['fx'] * z[s], (v[s] - c['cy']) / c['fy'] * z[s], z[s]], 1)
            X[s] = (Xc - np.array(c['t'])) @ np.array(c['R'])
        seen = np.zeros(m, int)
        for i in tr:
            c = cams[i]
            Xc = X @ np.array(c['R']).T + np.array(c['t'])
            uu = c['fx'] * Xc[:, 0] / Xc[:, 2] + c['cx']; vv = c['fy'] * Xc[:, 1] / Xc[:, 2] + c['cy']
            seen += ((Xc[:, 2] > c['near']) & (uu >= 0) & (uu < e3.W) & (vv >= 0) & (vv < e3.H)).astype(int)
        ok = seen >= 2
        col = np.zeros((int(ok.sum()), 3))
        ui = np.minimum(u[ok].astype(int), e3.W - 1); vi = np.minimum(v[ok].astype(int), e3.H - 1); cio = ci[ok]
        for i in np.unique(cio):
            s = cio == i
            col[s] = ims[i][vi[s], ui[s]]
        pts.append(X[ok]); cols.append(col / 255.0)
    pts, cols = np.concatenate(pts)[:n], np.concatenate(cols)[:n]
    return pts.astype(np.float32), cols.astype(np.float32)


def init_points(gt0, a, rng):
    p, ext = nt.init_points_colmap(gt0)                      # COLMAP sparse, 3-NN scales, ext = 2 x cameras_extent
    n_col = p['means'].shape[0]
    if a.init == 'colmap+random':
        pts, cols = random_frustum_points(a.n_random, rng, p['means'].cpu().numpy())
        allp = np.concatenate([p['means'].cpu().numpy(), pts]).astype(np.float32)
        sc = nt.knn_scale(allp)
        n = len(allp)
        q = {'means': allp.astype(np.float32), 'quats': np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
             'scales': np.repeat(sc[:, None], 3, 1).astype(np.float32), 'opacities': np.full((n,), -2.0, np.float32),
             'sh0': np.concatenate([p['sh0'].cpu().numpy(), ((cols - 0.5) / 0.28209479)[:, None]]).astype(np.float32),
             'shN': np.zeros((n, 15, 3), np.float32)}
        p = {k: torch.tensor(v, device=dev) for k, v in q.items()}
        print(f'init: {n_col} COLMAP + {len(pts)} random frustum points', flush=True)
    bands = (a.sh + 1) ** 2 - 1
    p['shN'] = torch.zeros((p['means'].shape[0], bands, 3), device=dev)
    return p, ext, n_col


def fit(p, gt0, ext, a, log):
    """frame-0 fit; returns the final params (shN padded to 15 bands so init.pt fits the SH3 codec path)."""
    if a.strategy == 'absgrad':
        e3.rasterization = functools.partial(e3.rasterization, absgrad=True)
    P = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in p.items()})
    lrs = e3.lr_table(ext, a.lr_scale)
    lrs['shN'] *= a.shn_lr
    opts = {k: torch.optim.Adam([P[k]], lr=lrs[k], eps=1e-15) for k in P}
    stop = int(a.iters * a.refine_stop)
    if a.strategy == 'mcmc':
        strat = MCMCStrategy(cap_max=a.cap, refine_stop_iter=stop, verbose=False)
    else:
        grow = a.grow if a.grow is not None else (0.0008 if a.strategy == 'absgrad' else 0.0002)
        strat = DefaultStrategy(refine_stop_iter=stop, reset_every=a.reset_every if a.reset_every > 0 else 10 ** 9,
                                prune_opa=a.prune_opa, grow_grad2d=grow, absgrad=a.strategy == 'absgrad', verbose=False)
    strat.check_sanity(P, opts)
    state = strat.initialize_state(scene_scale=ext / 2) if a.strategy != 'mcmc' else strat.initialize_state()
    sched = torch.optim.lr_scheduler.ExponentialLR(opts['means'], gamma=0.01 ** (1.0 / a.iters))
    test_gt = gt0[e3.test]
    ts = time.time()
    for it in range(a.iters):
        deg = min(a.sh, it // a.sh_every)
        cam = list(np.random.choice(e3.train, a.cams, replace=False))
        rgb, _, info = e3.render(P, cam, deg)
        strat.step_pre_backward(P, opts, state, it, info)
        loss = e3.loss_fn(rgb, gt0[cam])
        if a.strategy == 'mcmc':                               # 3DGS-MCMC regularisers (gsplat simple_trainer defaults)
            loss = loss + 0.01 * torch.sigmoid(P['opacities']).abs().mean() + 0.01 * torch.exp(P['scales']).abs().mean()
        if a.scale_reg > 0:
            loss = loss + a.scale_reg * torch.exp(P['scales']).mean()
        if a.opa_reg > 0:
            loss = loss + a.opa_reg * torch.sigmoid(P['opacities']).mean()
        loss.backward()
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        sched.step()
        if a.strategy == 'mcmc':
            strat.step_post_backward(P, opts, state, it, info, lr=sched.get_last_lr()[0])
        else:
            strat.step_post_backward(P, opts, state, it, info, packed=False)
        if it % 1000 == 0:
            print(f'  it {it} N={P["means"].shape[0]} loss {loss.item():.4f} {(it + 1) / (time.time() - ts):.1f} it/s', flush=True)
        if a.eval_every and ((it + 1) % a.eval_every == 0 or it + 1 == a.iters):
            with torch.no_grad():
                r, _, _ = e3.render(P, e3.test)
                ps = e3.psnr(r, test_gt)
            log['curve'].append({'it': it + 1, 'N': int(P['means'].shape[0]), 'psnr_test': round(ps, 3), 'loss': round(loss.item(), 5),
                                 'secs': round(time.time() - ts, 1)})
            print(f'  eval it {it + 1}: cam00 {ps:.2f} dB N={P["means"].shape[0]}', flush=True)
    out = {k: v.detach().contiguous() for k, v in P.items()}
    if out['shN'].shape[1] < 15:
        out['shN'] = torch.cat([out['shN'], torch.zeros((out['shN'].shape[0], 15 - out['shN'].shape[1], 3), device=dev)], 1)
    return out


def metrics(p, gt0, out_dir):
    with torch.no_grad():
        rgb, _, _ = e3.render(p, e3.test)
        rgb = torch.round(rgb * 255) / 255
        g = gt0[e3.test]
        m = {'psnr_test': e3.psnr(rgb, g), 'ssim_test': float(e3.ssim(rgb, g))}
        tr = [e3.psnr(e3.render(p, [c])[0], gt0[c:c + 1]) for c in e3.train]
        m['psnr_train_mean'] = float(np.mean(tr)); m['psnr_train_per_cam'] = [round(x, 2) for x in tr]
        try:
            import lpips
            a_ = rgb.permute(0, 3, 1, 2) * 2 - 1; b_ = g.permute(0, 3, 1, 2) * 2 - 1
            for n in ('alex', 'vgg'):
                net = lpips.LPIPS(net=n, verbose=False).to(dev).eval()
                m[f'lpips_{n}_test'] = float(net(a_, b_).mean())
                del net
        except ImportError:
            m['lpips_note'] = 'lpips not importable in this venv'
        im = torch.cat([g[0], rgb[0]], 1).cpu().numpy()
        Image.fromarray((im * 255).astype(np.uint8)).resize((im.shape[1] // 2, im.shape[0] // 2)).save(os.path.join(out_dir, 'cam00.png'))
    return m


def main(argv=None):
    a = parse(argv)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    gt0 = e3.load_frame(0)
    if a.eval_init:
        d = torch.load(a.eval_init, map_location=dev)
        p, ext = d['p'], d['ext']
        log = {'config': vars(a), 'gt': e3.GT, 'ext': ext, 'near': nt.NEAR, 'train_cams': len(e3.train),
               'test_cams': [e3.cams[i]['name'] for i in e3.test], 'curve': [], 'secs_fit_from_file': d.get('secs')}
        log.update(metrics(p, gt0, a.out), N=int(p['means'].shape[0]), secs=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(0))
        json.dump(log, open(os.path.join(a.out, 'fit.json'), 'w'), indent=1)
        print(f'P12 EVAL OK {a.out}: N={log["N"]} cam00 {log["psnr_test"]:.2f} dB SSIM {log["ssim_test"]:.4f} '
              f'train {log["psnr_train_mean"]:.2f} dB lpips_vgg {log.get("lpips_vgg_test", float("nan")):.4f}', flush=True)
        return
    p, ext, n_col = init_points(gt0, a, rng)
    log = {'config': vars(a), 'gt': e3.GT, 'ext': ext, 'near': nt.NEAR, 'n_init': int(p['means'].shape[0]), 'n_colmap': n_col,
           'train_cams': len(e3.train), 'test_cams': [e3.cams[i]['name'] for i in e3.test], 'curve': []}
    p = fit(p, gt0, ext, a, log)
    secs = time.time() - t0
    torch.save({'p': p, 'ext': ext, 'secs': secs}, os.path.join(a.out, 'init.pt'))
    log.update(metrics(p, gt0, a.out), N=int(p['means'].shape[0]), secs=round(secs, 1), gpu=torch.cuda.get_device_name(0))
    json.dump(log, open(os.path.join(a.out, 'fit.json'), 'w'), indent=1)
    print(f'P12 INIT OK {a.out}: N={log["N"]} cam00 {log["psnr_test"]:.2f} dB SSIM {log["ssim_test"]:.4f} '
          f'train {log["psnr_train_mean"]:.2f} dB lpips_vgg {log.get("lpips_vgg_test", float("nan")):.4f} ({secs:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
