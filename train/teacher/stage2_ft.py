#!/usr/bin/env python3
"""stage 2b: short fine-tuning of the fitted curves (P35 B: fit, then a short retraining with ids and
lifetimes fixed, no densify/prune), section by section, against the same targets as stage 1 (teacher PLY renders of the
frame blended with the real GT by P43_MIX; real GT for frames without a PLY; P43_MATTE=1: alpha target = GT matte, teacher
RGB scaled down where the teacher is more opaque than the matte = no teacher halo; P43_MATTE=2: teacher RGBA cut to zero
outside the GT matte dilated by P43_MATTE_DIL px = only the far halo/floaters go, the teacher's own edge stays), loss = rig_track.train_loss (random
background, alpha loss) + P43_FT_LPIPS x LPIPS-vgg on the matte bbox. Per step P43_FB random frames of the section x P43_CB random train cameras; every knot of every alive splat
is trainable (soft knot 0 = sections independent); learning rates = rig_track.lr_table(ext) x P43_FT_LR.
Real GT images are cached on the GPU as uint8 when first used.
  python stage2_ft.py <fit k dir> <out k dir>      env: P43_FT_ITS (per 32 frames, 3000), P43_FT_LR (0.3), P43_MIX (0),
       P43_FB (2), P43_CB (2), P43_PLYFRAMES (50), P8_GT, P8_ALPHA (0.3), P8_RANDBG (1), P8_RANDBG_P (0.5)
Writes <out>/curves.npz, <out>/tracks.npz (dense samples, shN = 0), <out>/ft.json (loss trace, PSNR vs GT test cams).
"""
import json
import os
import sys
import time

os.environ.setdefault('E3_SH', '0')
os.environ.setdefault('P8_ALPHA', '0.3')
os.environ.setdefault('P8_RANDBG', '1')
os.environ.setdefault('P8_RANDBG_P', '0.5')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rig'))
sys.path.insert(0, HERE)
import numpy as np                     # noqa: E402
import torch                           # noqa: E402
import torch.nn.functional as Fn       # noqa: E402
from PIL import Image                  # noqa: E402
import rig_track as e3                     # noqa: E402
import rig_lib as L                     # noqa: E402
from stage2_fit import hat_basis, ATTRS, GROUP  # noqa: E402

ROOT = os.environ.get('ROOT', 'work')
ITS32 = int(os.environ.get('P43_FT_ITS', '3000'))
LRS = float(os.environ.get('P43_FT_LR', '0.3'))
MIX = float(os.environ.get('P43_MIX', '0'))
FB = int(os.environ.get('P43_FB', '2'))
CB = int(os.environ.get('P43_CB', '2'))
PLYF = int(os.environ.get('P43_PLYFRAMES', '50'))
MATTE = int(os.environ.get('P43_MATTE', '0'))   # 1: target alpha = GT matte; 2: teacher cut outside the dilated matte
MATTE_DIL = int(os.environ.get('P43_MATTE_DIL', '3'))
FT_LPW = float(os.environ.get('P43_FT_LPIPS', '0'))        # LPIPS-vgg term against the target, 0 = off
FT_KEEP = os.environ.get('P43_FT_KEEP', '')                 # keep.npz (splat budget): train ONLY the kept ids (prune -> ft)
EXT = float(os.environ.get('P43_EXT', '2.0'))
PLY = os.environ.get('P43_TEACHER', '')                            # teacher PLY pattern with {:05d}
dev = e3.dev
_gt = {}
_ply = {}


def gt_img(t, c):
    if (t, c) not in _gt:
        a = np.asarray(Image.open(f'{e3.GT}/f{t:03d}/c{c:02d}.png'))[..., :4]
        _gt[(t, c)] = torch.tensor(a, device=dev)
    return _gt[(t, c)].float() / 255.0


def ply(t):
    if t not in _ply:
        p = L.ply_params(L.read_ply(PLY.format(t)))
        d = {k: torch.tensor(np.ascontiguousarray(v), device=dev).float() for k, v in p.items()}
        d['quats'] = torch.nn.functional.normalize(d['quats'], dim=-1)
        d['shN'] = torch.zeros((len(d['means']), 0, 3), device=dev)
        _ply[t] = d
    return _ply[t]


def target(t, cams):
    if t >= PLYF:
        return torch.stack([gt_img(t, c) for c in cams])
    with torch.no_grad():
        rgb, alpha, _ = e3.render(ply(t), cams, 0)
        out = torch.cat([rgb, alpha], -1)
    if MIX > 0:
        out = (1 - MIX) * out + MIX * torch.stack([gt_img(t, c) for c in cams])
    if MATTE == 1:                    # silhouette from the GT matte: no teacher halo outside, no holes inside
        ga = torch.stack([gt_img(t, c)[..., 3:] for c in cams])
        scale = (ga / out[..., 3:].clamp_min(1e-3)).clamp(max=1.0)
        out = torch.cat([out[..., :3] * scale, ga], -1)
    elif MATTE == 2:                  # only cut the teacher's halo/floaters outside the GT matte dilated by MATTE_DIL px
        ga = torch.stack([gt_img(t, c)[..., 3] for c in cams])
        m = Fn.max_pool2d((ga > 0.5).float()[:, None], 2 * MATTE_DIL + 1, stride=1, padding=MATTE_DIL)[:, 0, ..., None]
        out = out * m
    return out


_lp = None


def lpips_term(rgb, g):
    """mean LPIPS-vgg over the views on the target's alpha bbox (as p8_e7.lpips_term)."""
    global _lp
    if _lp is None:
        import lpips
        _lp = lpips.LPIPS(net='vgg', verbose=False).to(dev).eval()
        for q in _lp.parameters():
            q.requires_grad_(False)
    out = 0.0
    for c in range(len(rgb)):
        a, b = rgb[c], g[c, ..., :3]
        m = g[c, ..., 3] > 0.5
        ys, xs = torch.nonzero(m.any(1)).squeeze(1), torch.nonzero(m.any(0)).squeeze(1)
        if len(ys) >= 16 and len(xs) >= 16:
            y0, y1 = max(int(ys[0]) - 16, 0), min(int(ys[-1]) + 17, a.shape[0])
            x0, x1 = max(int(xs[0]) - 16, 0), min(int(xs[-1]) + 17, a.shape[1])
            a, b = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
        out = out + _lp(a.permute(2, 0, 1)[None].clamp(0, 1), b.permute(2, 0, 1)[None], normalize=True).mean()
    return out / len(rgb)


def at(P, A, alive_t, ti):
    w = torch.tensor(A[ti], device=dev, dtype=torch.float32)
    i = alive_t
    out = {k: torch.einsum('k,nkd->nd', w, P[k][i]) for k in ('means', 'quats', 'scales', 'sh0')}
    out['opacities'] = torch.einsum('k,nk->n', w, P['opacities'][i])
    out['sh0'] = out['sh0'][:, None]
    out['shN'] = torch.zeros((len(i), 0, 3), device=dev)
    return out


def main():
    kin, kout = sys.argv[1], sys.argv[2]
    os.makedirs(kout, exist_ok=True)
    t0 = time.time()
    cv = dict(np.load(os.path.join(kin, 'curves.npz')))
    tr = np.load(os.path.join(kin, 'tracks.npz'))
    birth, death = tr['birth'], tr['death']
    T, Ng = tr['means'].shape[:2]
    keep = np.load(FT_KEEP)['keep'].astype(bool) if FT_KEEP else np.ones(Ng, bool)
    if FT_KEEP:                        # dropped ids never live in the output tracks (the codec gets the same keep)
        death = np.where(keep, death, birth)
        print(f'P43_FT_KEEP: training {int(keep.sum())} of {Ng} ids', flush=True)
    Lsec, K = int(cv['L']), int(cv['K'])
    lr0 = e3.lr_table(EXT, LRS)
    dense = {k: np.full(tr[k].shape, np.nan, np.float32 if k in ('means', 'quats') else np.float16) for k in ATTRS}
    log = {'cfg': {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(('P43_', 'P8_', 'E3_'))},
           'sections': []}
    for gi, g0 in enumerate(range(0, T, Lsec)):
        g1 = min(T, g0 + Lsec)
        Lg = g1 - g0
        A = hat_basis(Lg, K)
        b = torch.tensor(cv[f'g{gi}_birth'], device=dev)
        d = torch.tensor(cv[f'g{gi}_death'], device=dev)
        kp_ = torch.tensor(keep[cv[f'g{gi}_ids']], device=dev)
        alive = [torch.nonzero((b <= t) & (d > t) & kp_).squeeze(1) for t in range(g0, g1)]
        P = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.tensor(cv[f'g{gi}_{k}'], device=dev)) for k in ATTRS})
        proj = []                      # adaptive K of stage2_fit: keep every splat on its nested K' curve while training
        for k in ATTRS:
            kk = f'g{gi}_p43k{GROUP[k]}'
            if kk not in cv:
                continue
            kp = cv[kk]
            for k2 in np.unique(kp):
                k2 = int(k2)
                if k2 >= K:
                    continue
                sel = torch.tensor(np.nonzero(kp == k2)[0], device=dev)
                if k2 == 1:
                    sub, H = [0], torch.ones((K, 1), device=dev)
                else:
                    st = (K - 1) // (k2 - 1)
                    sub, H = list(range(0, K, st)), torch.tensor(hat_basis(K, k2), device=dev, dtype=torch.float32)
                proj.append((k, sel, sub, H))

        def project():
            with torch.no_grad():
                for k, sel, sub, H in proj:
                    x = P[k].data[sel]
                    x = x[:, sub]
                    P[k].data[sel] = torch.einsum('jk,nk...->nj...', H, x)
        project()
        opt = torch.optim.Adam([{'params': [P[k]], 'lr': lr0[k]} for k in ATTRS], eps=1e-15)
        its = max(1, ITS32 * Lg // 32)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1 ** (1.0 / its))
        trace = []
        ts = time.time()
        for it in range(its):
            loss = 0.0
            for ti in np.random.randint(0, Lg, FB):
                cams = list(np.random.choice(e3.train, CB, replace=False))
                p = at(P, A, alive[ti], ti)
                rgb, alpha, _ = e3.render(p, cams, 0)
                tg = target(g0 + int(ti), cams)
                loss = loss + e3.train_loss(rgb, alpha, tg)
                if FT_LPW > 0:
                    loss = loss + FT_LPW * lpips_term(rgb, tg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            project()
            sched.step()
            if it % 250 == 0 or it == its - 1:
                trace.append((it, float(loss) / FB))
                print(f'  g{gi} it {it}/{its} loss {float(loss) / FB:.4f} ({time.time() - ts:.0f} s)', flush=True)
        ev = []
        with torch.no_grad():
            for ti in range(Lg):
                p = at(P, A, alive[ti], ti)
                rgb, _, _ = e3.render(p, e3.test, 0)
                g = torch.stack([gt_img(g0 + ti, c) for c in e3.test])[..., :3]
                ev.append(e3.psnr_fg(rgb, g))
                for k in ATTRS:
                    v = p[k]
                    ids = cv[f'g{gi}_ids'][alive[ti].cpu().numpy()]
                    slot = dense[k][g0 + ti, ids]
                    dense[k][g0 + ti, ids] = v.cpu().numpy().reshape(slot.shape)
        for k in ATTRS:
            cv[f'g{gi}_{k}'] = P[k].detach().cpu().numpy().astype(np.float32)
        log['sections'].append({'g0': g0, 'g1': g1, 'its': its, 'secs': round(time.time() - ts, 1), 'trace': trace,
                                'fg_psnr_gt_test_per_frame': ev})
        print(f'section {gi}: test fg-PSNR (GT, e3 mask) {np.mean(ev):.2f} dB', flush=True)
        del P, opt
        _gt.clear()
        _ply.clear()
        torch.cuda.empty_cache()
    np.savez(os.path.join(kout, 'curves.npz'), **cv)
    np.savez(os.path.join(kout, 'tracks.npz'), birth=birth, death=death, shN=np.zeros((Ng, 15, 3), np.float16), **dense)
    log['secs'] = round(time.time() - t0, 1)
    json.dump(log, open(os.path.join(kout, 'ft.json'), 'w'), indent=1)
    print('S2FT OK', kout, log['secs'], flush=True)


if __name__ == '__main__':
    main()
