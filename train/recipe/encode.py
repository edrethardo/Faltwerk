#!/usr/bin/env python3
"""standard-recipe encoder: ONE clean encode for every stream of every source (images or mesh).
The parameters come from rules on the curves themselves (train/recipe/recipe_rules.py), constants from
tools/recipe_standard.json:
  1. keep: dead (opacity < 1/255 at every knot), giant (sigma > giant_rel x R95), runaway (leaves runaway_rel x R99),
     AND an optional input keep (splat budget)
  2. deep floor (hat basis only, only when invisible knots exist): invisible stretches -> logit -12
  3. position step from the splat size (qstep_rel x sigma_fine), qpos = step / ext_kept, --ext ext_kept
  4. SH degree = highest band with energy; K list up to the curves' K (mesh K=33 -> 1..33)
  5. --cull-floor always; quat raw, ktol 1, key interval 4; prediction modes per source (mesh: spatial, Morton)
  encode.py <k dir> <out dir> [--keep in.npz] [--source images|mesh] [--cfg tools/recipe_standard.json]
            [--set qstep_rel=0.3 ...] [--out-name falt|csgs]
            -> <out>/{keep_clean.npz, rules.json, falt/, measure.json, falt.json} (runs before csgs/, csgs.json)
"""
import argparse, json, os, subprocess, sys, time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, ROOT)
from codec import prep                                    # noqa: E402
from codec.falt import Encoder, Decoder, stream_dir, stream_sizes, write_section  # noqa: E402
from codec.measure import parse_pred                      # noqa: E402
from train.recipe import recipe_rules as rr            # noqa: E402

CUT = float(np.log(1 / 255 / (1 - 1 / 255)))


def load_cfg(path, source, sets):
    cfg = json.load(open(path))
    c = dict(cfg['encode']['common']); c.update(cfg['encode'].get(source, {}))
    for kv in sets:
        k, v = kv.split('=', 1)
        try:
            c[k] = json.loads(v)
        except ValueError:
            c[k] = v
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kdir'); ap.add_argument('out')
    ap.add_argument('--keep'); ap.add_argument('--source', choices=('images', 'mesh'), default='images')
    ap.add_argument('--cfg', default=os.path.join(ROOT, 'tools', 'recipe_standard.json'))
    ap.add_argument('--set', action='append', default=[], help='override one constant (A/B only, never per scene)')
    ap.add_argument('--fps', type=float, default=None)
    ap.add_argument('--cams', help='images: cameras.json of the GT (position step = qstep_px pixels at the median depth)')
    ap.add_argument('--footprint', type=float, help='mesh: metres per pixel at the figures (train/recipe/mesh_std.py)')
    ap.add_argument('--matte', action='store_true', help='object scene with a GT matte: giant / runaway cut on (P40)')
    ap.add_argument('--out-name', default=None,
                    help='stream dir + summary name under <out>: default falt (falt/ + falt.json); an <out> that '
                         'already holds csgs/ keeps csgs; "csgs" = the pre-layout chains that read <out>/csgs expect')
    a = ap.parse_args()
    c = load_cfg(a.cfg, a.source, a.set)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    cv = os.path.join(a.kdir, 'curves.npz'); tr = os.path.join(a.kdir, 'tracks.npz')
    keep_in = np.load(a.keep)['keep'].astype(bool) if a.keep else None
    st = rr.curve_stats(cv, keep=keep_in)
    zt0 = np.load(tr, mmap_mode='r')
    if 'scales' in zt0.files and 'opacities' in zt0.files:  # per-frame tracks: exact sigma_max / opacity_max per id
        with np.errstate(invalid='ignore', over='ignore'):  # (curve knots overshoot between frames)
            smx = np.exp(np.nanmax(np.asarray(zt0['scales'], np.float32).max(-1), axis=0))
            omx = 1 / (1 + np.exp(-np.nanmax(np.asarray(zt0['opacities'], np.float32), axis=0)))
        n = min(len(st['S']), len(smx))
        st['S'][:n] = np.nan_to_num(smx[:n], nan=0.0); st['O'][:n] = np.nan_to_num(omx[:n], nan=0.0)
        st['vis'] = st['present'] & (st['O'] >= rr.DEFAULTS['visible_opa'])
        st['summary']['sigma_source'] = 'tracks'
    del zt0
    d = rr.derive_encoder(st, {k: c[k] for k in ('giant_rel', 'runaway_rel', 'qstep_rel') if k in c})
    d['step_sigma'] = d['step']                            # qstep_rel x sigma_fine (bound of the relaxed step)
    fp = a.footprint
    if a.footprint is not None and not a.footprint > 0:    # a locale-broken awk passed 0 -> silent sigma step
        raise SystemExit(f'--footprint {a.footprint}: must be > 0 (metres per pixel at the figures)')
    if fp is None and a.cams:
        fp = rr.footprint_from_cams(a.cams, st['C'][st['vis']])
    d['footprint'] = fp
    if fp and c.get('qstep_mode', 'px') in ('px', 'relaxed'):
        d['step_px'] = float(c['qstep_px']) * fp           # quality first: half a pixel at the median content depth
        d['step'] = d['step_px'] if c.get('qstep_mode', 'px') == 'px' else max(d['step_px'], d['step_sigma'])
    d['qpos'] = d['step'] / d['ext_kept']
    d['qstep_mode'] = c.get('qstep_mode', 'px') if fp else 'sigma'
    if c.get('step_fixed'):                                # A/B only: absolute position step (e.g. the old 0.5-px rule)
        d['step'] = float(c['step_fixed']); d['qpos'] = d['step'] / d['ext_kept']
    S, O, LO, HI = st['S'], st['O'], st['LO'], st['HI']
    med = np.asarray(st['summary']['median'])
    far = np.maximum(np.linalg.norm(LO - med, axis=1), np.linalg.norm(HI - med, axis=1))
    pres = st['present']
    dead = pres & (O < 1 / 255)
    giant = pres & (S > d['smax'])
    run = pres & (far > d['rmax'])
    if not a.matte:                                        # full-frame scene (no matte): far / large splats ARE the
        giant[:] = False; run[:] = False                   # background (N3DV flame: -2.3 dB without them) -> keep
        d['size_cuts'] = False
    else:
        d['size_cuts'] = True
    keep = pres & ~dead & ~giant & ~run
    ntr = int(np.load(tr, mmap_mode='r')['birth'].shape[0])
    kfull = np.zeros(ntr, bool); kfull[:len(keep)] = keep[:ntr]
    np.savez(os.path.join(a.out, 'keep_clean.npz'), keep=kfull)
    sn_deg = rr.sn_degree(tr) if c.get('sn_deg', 'auto') == 'auto' else int(c['sn_deg'])
    # deep floor (hat basis; only when invisible knots exist)
    z = np.load(cv)
    basis = str(z['basis']) if 'basis' in z.files else 'hat'
    nbelow = sum(int((z[k] < CUT).sum()) for k in z.files if k.startswith('g') and k.endswith('_opacities') and z[k].ndim == 2)
    K = int(z['K']); del z
    cvx = cv
    if basis == 'hat' and nbelow > 0 and c.get('deep_floor', True):
        cvx = os.path.join(a.out, 'curves_deep.npz')
        if not os.path.exists(cvx):
            subprocess.run([sys.executable, os.path.join(ROOT, 'train', 'recipe', 'deep_floor.py'), cv, cvx], check=True,
                           stdout=subprocess.DEVNULL)
    ks = tuple(k for k in (1, 2, 3, 5, 9, 17, 33) if k <= K)
    rules = {'encoder': 'p44-v2', 'source': a.source, 'cfg': c, 'stats': st['summary'], 'derived': d, 'sn_deg': sn_deg, 'ks': list(ks),
             'basis': basis, 'invisible_knots': nbelow, 'deep_floor': cvx != cv,
             'cut': {'input_keep': int(keep_in.sum()) if keep_in is not None else None, 'present': int(pres.sum()),
                     'dead': int(dead.sum()), 'giant': int(giant.sum()), 'runaway': int(run.sum()), 'kept': int(keep.sum())}}
    json.dump(rules, open(os.path.join(a.out, 'rules.json'), 'w'), indent=1)
    print('RULES', json.dumps({**d, 'sn_deg': sn_deg, **rules['cut']}), flush=True)
    pred = parse_pred(c.get('pred') or None)
    csd = os.path.join(a.out, a.out_name) if a.out_name else stream_dir(a.out)
    name = os.path.basename(csd)
    params, secs, extra = prep.sections_from_curves(cvx, tr, d['qpos'], ks=ks, key_interval=int(c['key_interval']),
                                                    quat='raw', tol=int(c['ktol']), sn_deg=sn_deg, pred=pred,
                                                    knn_k=int(c.get('knn_k', 0)), ext=d['ext_kept'], cull_floor=True,
                                                    newborn_order='morton' if c.get('morton') else 'id', keep=kfull)
    sp = extra.pop('section_params', None) or [params] * len(secs)
    os.makedirs(csd, exist_ok=True)
    enc, dec = Encoder(params), Decoder()
    rows, total = [], 0
    for si, s in enumerate(secs):
        enc.params = sp[si]
        buf = enc.encode(s)
        write_section(csd, s.index, buf)
        ok = dec.decode(buf) == s
        total += len(buf)
        rows.append({'index': s.index, 'splats': len(s.ids), 'n_new': len(s.ids) - s.n_cont, 'bytes': len(buf),
                     'streams': stream_sizes(buf), 'roundtrip_ok': ok})
        print(s.index, len(s.ids), len(buf), ok, flush=True)
    T = secs[-1].g0 + secs[-1].nframes
    fps = a.fps or float(c.get('fps', 30.0))
    zt = np.load(tr, mmap_mode='r'); b, dth = zt['birth'][kfull], zt['death'][kfull]
    alive_mean = float(np.mean([((b <= t) & (dth > t)).sum() for t in range(T)]))
    res = {'kdir': a.kdir, 'qpos': d['qpos'], 'ext': d['ext_kept'], 'pos_step': extra.get('pos_step'), 'ks': list(ks),
           'T': T, 'fps': fps, 'bytes': total, 'mbit_s': total * 8 / (T / fps) / 1e6, 'sections': rows,
           'roundtrip_all_ok': all(r['roundtrip_ok'] for r in rows), 'secs': round(time.time() - t0, 1), 'rules': rules}
    json.dump(res, open(os.path.join(a.out, f'{name}.json'), 'w'), indent=1, default=str)
    # codec.measure-style measure.json (fields the evaluations read)
    khist = {g: {} for g, _ in prep.GROUPS}
    for s in secs:
        for gi, (g, _) in enumerate(prep.GROUPS):
            for k in s.k[gi]:
                khist[g][int(k)] = khist[g].get(int(k), 0) + 1
    json.dump({'total_bytes': {'rans': total}, 'T': T, 'alive_mean': alive_mean, 'roundtrip_all_ok': res['roundtrip_all_ok'],
               'version': 3 if pred else 1, 'basis': basis, 'qpos': d['qpos'], 'ext': d['ext_kept'], 'csgs_dir': csd,
               'extra': {'k_hist': khist},
               'note': 'written by train/recipe/encode.py'},
              open(os.path.join(a.out, 'measure.json'), 'w'), indent=1)
    print('P44ENC OK', csd, f"{res['mbit_s']:.2f} Mbit/s at {fps} fps", 'roundtrip', res['roundtrip_all_ok'], flush=True)


if __name__ == '__main__':
    main()
