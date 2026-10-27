"""measurement: .csgs/rANS vs zstd-19 on the SAME symbols, bit-exact roundtrip, timings (numpy + zstandard).

  python -m codec.measure --source tracks --tracks <tracks.npz> [--qpos 0.001 --tol 1] --out <json>
  python -m codec.measure --source curves --curves <curves.npz> --tracks <tracks.npz> --out <json>

Per section and stream: rANS bytes (the stream inside the .csgs section) vs zstd-19 over the very integers the rANS
stream codes (smallest fixed-width little-endian array that holds them, one zstd frame per stream and section) and
vs zstd-19 over all streams of a section concatenated (one frame per section, like e6). For --source tracks also
the e6 byte layout (codec.prep reproduces e6.encode_curves) as a cross-check of the published e6 number.
"""
import argparse, json, os, time

import numpy as np
import zstandard

from .csgs import PRED_TEMPORAL, Decoder, Encoder, Params, encode_section, stream_sizes, stream_names
from . import prep

FPS = 25.0


def packed(vals):
    a = np.asarray(vals, dtype=np.int64)
    if a.size == 0:
        return b''
    lo, hi = int(a.min()), int(a.max())
    for dt in (np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64):
        info = np.iinfo(dt)
        if info.min <= lo and hi <= info.max:
            return a.astype(dt).tobytes()


def with_pred(p, modes, knn_k=None):
    """copy of Params p with other motion-prediction modes (same quantisation table, ks, groups)."""
    return Params(p.L, p.ks, p.groups, p.n_sn, p.offsets, p.scales, p.shared_knot, p.key_interval, sn_vq=p.sn_vq,
                  sn_zero=p.sn_zero, pred=modes, knn_k=p.knn_k if knn_k is None else knn_k, bspline=p.bspline)


def parse_pred(spec, groups=prep.GROUPS):
    """"P=1,R=3" / "all=1" -> one mode per group (codec.csgs.Params.pred); None/'' -> None (v1)."""
    if not spec or spec == 'auto':
        return None
    modes = [0] * len(groups)
    for part in spec.split(','):
        k, v = part.split('=')
        if k == 'all':
            modes = [int(v)] * len(groups)
        else:
            modes[[g for g, _ in groups].index(k)] = int(v)
    return modes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=('tracks', 'curves'), required=True)
    ap.add_argument('--tracks', required=True)
    ap.add_argument('--curves')
    ap.add_argument('--qpos', type=float, default=0.001)
    ap.add_argument('--tol', type=float, default=1.0)
    ap.add_argument('--key-interval', type=int, default=4)
    ap.add_argument('--max-sections', type=int)
    ap.add_argument('--out', required=True)
    ap.add_argument('--quat', choices=('norm', 'raw'), default='norm', help='curves: quaternion knots (prep)')
    ap.add_argument('--ktol', type=int, default=0, help='curves: K reduction tolerance in quantisation steps')
    ap.add_argument('--ktol-group', default=None,
                    help='curves (P22): per-group K tolerance overriding --ktol/plan ktol, e.g. "D=2,O=1" (default: none)')
    ap.add_argument('--no-follow', action='store_true', help='curves: send dormant position knots explicitly (no follow flag)')
    ap.add_argument('--shared-knot', choices=('auto', '0', '1'), default='auto', help='curves: shared boundary knot')
    ap.add_argument('--sn-deg', type=int, default=3, choices=(0, 1, 2, 3), help='curves: highest SH degree sent (higher bands = 0)')
    ap.add_argument('--sn-bits', type=int, default=8, help='curves: quantisation bits of the sent SH scalars')
    ap.add_argument('--write-csgs', help='directory for the .csgs section files (gitignored data only)')
    ap.add_argument('--sn-vq', help='curves: SN-VQ levels "deg:size[+size..],..." e.g. "1:1024,3:4096"')
    ap.add_argument('--sn-deg-q', help='SN-VQ: fraction of splats per degree below the highest, e.g. "0.2,0.3" for degrees 0,1')
    ap.add_argument('--sn-deg-thr', help='SN-VQ: absolute weighted-energy thresholds per degree below the highest')
    ap.add_argument('--sn-weight', choices=('none', 'importance'), default='none', help='SN-VQ: k-means / rule weight')
    ap.add_argument('--sn-iters', type=int, default=25)
    ap.add_argument('--sn-seed', type=int, default=0)
    ap.add_argument('--sn-vq-from', help='SN-VQ: reuse a saved assignment .npz (degrees, indices, codebooks, SN grid)')
    ap.add_argument('--sn-vq-save', help='SN-VQ: save the assignment .npz (for the retraining with fixed SN)')
    ap.add_argument('--ext', type=float, default=None, help='curves: fixed scene extent for the position step (P10; default: min/max box)')
    ap.add_argument('--cull-floor', action='store_true', help='curves: opacity knots below 1/255 rounded down (never visible after decoding; P10)')
    ap.add_argument('--newborn-order', choices=('id', 'morton'), default='id',
                    help='curves (S2): renumber the newborns of a section along a Morton curve (stream stays v1/v2)')
    ap.add_argument('--pred', default=None,
                    help='curves (S2): .csgs v3 motion-prediction modes per group, e.g. "P=1,R=1" or "all=3"; '
                         'bits 1 temporal, 2 spatial (k-NN), 4 velocity for knot 0 of continuing splats; '
                         '"auto" = temporal per section and group where it shrinks the stream; combinable with --sn-vq')
    ap.add_argument('--knn-k', type=int, default=0, help='curves (S2): neighbours for spatial prediction')
    ap.add_argument('--basis', choices=('auto', 'hat', 'bspline'), default='auto',
                    help='curves (P16): curve basis; auto = the trainer\'s curves.npz field (E7_BASIS), bspline = flag 64 / version 6')
    ap.add_argument('--plan', default=None,
                    help='curves (S7 plan in codec.measure, ): JSON list (or file) with one dict per section '
                         '{qpos, attr_levels, rot_levels, ktol, sn_levels}; combinable with --sn-vq*/--newborn-order/--pred')
    ap.add_argument('--uniform', default=None,
                    help='curves: \"qpos,attr,rot,ktol[,sn]\" = the same plan entry in every section (S7 candidate)')
    ap.add_argument('--keep', help='curves: npz with a bool `keep` per global id (experiments/james/matte_cull.py)')
    ap.add_argument('--keep-frac', type=float, help='curves: with --keep, recompute keep from its stats at this centre-out fraction')
    a = ap.parse_args()
    keep = None
    if a.keep:
        kz = np.load(a.keep)
        keep = kz['keep']
        if a.keep_frac is not None:
            keep = ~((kz['centre_out'] > a.keep_frac) | (kz['ends_out'] > max(a.keep_frac, 0.6)))
        print(f'keep {int(keep.sum())} of {len(keep)} splats ({a.keep})', flush=True)
    plan = None
    if a.plan:
        plan = json.load(open(a.plan)) if os.path.exists(a.plan) else json.loads(a.plan)
    elif a.uniform:
        u = a.uniform.split(',')
        one = {'qpos': float(u[0]), 'attr_levels': int(u[1]), 'rot_levels': int(u[2]), 'ktol': int(u[3]),
               'sn_levels': int(u[4]) if len(u) > 4 else (1 << a.sn_bits) - 1}
        nsec_all = sum(1 for f in np.load(a.curves).files if f.startswith('g') and f.endswith('_ids'))
        plan = [dict(one) for _ in range(nsec_all)]
    if plan is not None:                                   # the plan's first entry rules the stream-wide defaults
        a.qpos, a.ktol = float(plan[0].get('qpos', a.qpos)), int(plan[0].get('ktol', a.ktol))
    Z = zstandard.ZstdCompressor(level=19)
    t0 = time.time()
    sn_vq = None
    if a.sn_vq or a.sn_vq_from:
        from . import snvq
        sn_vq = {'weight': a.sn_weight, 'iters': a.sn_iters, 'seed': a.sn_seed, 'save': a.sn_vq_save}
        if a.sn_vq_from:
            sn_vq['from'] = snvq.Assignment.load(a.sn_vq_from); sn_vq['from_path'] = a.sn_vq_from
            sn_vq['levels'] = sn_vq['from'].levels
        else:
            sn_vq['levels'] = snvq.parse_levels(a.sn_vq)
        if a.sn_deg_q:
            sn_vq['q'] = [float(x) for x in a.sn_deg_q.split(',')]
        if a.sn_deg_thr:
            sn_vq['thr'] = [float(x) for x in a.sn_deg_thr.split(',')]
    if a.source == 'tracks':
        params, secs, extra = prep.sections_from_tracks(a.tracks, a.qpos, a.tol, key_interval=a.key_interval,
                                                        max_sections=a.max_sections)
        T, alive_mean = extra['T'], extra['alive_mean']
    else:
        sk = None if a.shared_knot == 'auto' else a.shared_knot == '1'
        params, secs, extra = prep.sections_from_curves(a.curves, a.tracks, a.qpos, key_interval=a.key_interval,
                                                        max_sections=a.max_sections, quat=a.quat, tol=a.ktol,
                                                        shared_knot=sk, sn_vq=sn_vq, ext=a.ext, cull_floor=a.cull_floor,
                                                        newborn_order=a.newborn_order, pred=parse_pred(a.pred),
                                                        knn_k=a.knn_k, follow=not a.no_follow,
                                                        basis=None if a.basis == 'auto' else a.basis, plan=plan,
                                                        sn_deg=a.sn_deg, sn_bits=a.sn_bits, keep=keep,
                                                        ktol_groups={kv.split('=')[0]: int(kv.split('=')[1])
                                                                     for kv in a.ktol_group.split(',')} if a.ktol_group else None)
        tr = np.load(a.tracks)
        b, dth = tr['birth'], tr['death']
        if keep is not None:
            b, dth = b[keep], dth[keep]
        T = int(tr['means'].shape[0])
        al = (np.arange(T)[:, None] >= b[None]) & (np.arange(T)[:, None] < dth[None])
        alive_mean = float(al.sum(1).mean())
    t_prep = time.time() - t0
    print(f'prep {t_prep:.1f} s, {len(secs)} sections, T={T}, alive/frame {alive_mean:.0f}', flush=True)
    names = stream_names(params)
    sp = extra.pop('section_params', None) or [params] * len(secs)   # S7 plan: one table per section (else all equal)
    enc = Encoder(params)
    rows, bufs = [], []
    auto = a.source == 'curves' and a.pred == 'auto'
    for si, s in enumerate(secs):
        col = {}
        t = time.time()
        enc.params = sp[si]
        if auto:                                           # per section and group: temporal prediction only where it
            key = enc.state is None or s.index % params.key_interval == 0   # shrinks the stream (header carries the modes)
            cand = [encode_section(with_pred(sp[si], [m] * len(params.groups)), s, key, enc.state) for m in (0, PRED_TEMPORAL)]
            szs = [stream_sizes(b) for b in cand]
            modes = [PRED_TEMPORAL if szs[1][f'knots_{tg}'] < szs[0][f'knots_{tg}'] else 0 for tg, _ in params.groups]
            enc.params = with_pred(sp[si], modes)
        buf = enc.encode(s, collect=col); te = time.time() - t
        bufs.append(buf)
        sz = stream_sizes(buf)
        zs = {nm: len(Z.compress(packed(col.get(nm, [])))) if col.get(nm) else 0 for nm in names}
        nsym = {nm: len(col.get(nm, [])) for nm in names}
        blob = b''.join(packed(col[nm]) for nm in names if col.get(nm))
        row = {'index': s.index, 'key': s.index % params.key_interval == 0, 'splats': len(s.ids), 'n_cont': s.n_cont,
               'n_new': len(s.ids) - s.n_cont, 'bytes_csgs': len(buf), 'rans': sz, 'zstd19_per_stream': zs,
               'zstd19_concat': len(Z.compress(blob)), 'nsym': nsym, 'secs_encode': round(te, 2),
               'pred_modes': enc.params.pred, 'version': buf[4], 'flags': buf[5]}
        if a.write_csgs:
            os.makedirs(a.write_csgs, exist_ok=True)
            open(os.path.join(a.write_csgs, f's{s.index:04d}.csgs'), 'wb').write(buf)
        rows.append(row)
        print({k: v for k, v in row.items() if k not in ('nsym',)}, flush=True)
    dec = Decoder()
    for s, buf, row in zip(secs, bufs, rows):
        t = time.time(); out = dec.decode(buf); td = time.time() - t
        row['secs_decode'] = round(td, 2)
        row['roundtrip_ok'] = out == s
        if not row['roundtrip_ok']:
            row['roundtrip_first_diff'] = out.first_difference(s)
        print(f'section {s.index}: decode {td:.1f} s, roundtrip {row["roundtrip_ok"]}', flush=True)
    # key-section variant of every non-key section (cost of a random-access point) + its stand-alone roundtrip
    for i, s in enumerate(secs):
        if rows[i]['key']:
            continue
        ks = s
        if params.sn_vq is not None:                       # a stand-alone key version must carry its interval's codebooks
            import copy
            ks = copy.copy(s)
            ks.sn_cb = next(x.sn_cb for x in reversed(secs[:i]) if x.sn_cb is not None)
        kb = Encoder(sp[i]).encode(ks, key=True)
        rows[i]['bytes_as_key'] = len(kb)
        rows[i]['key_roundtrip_ok'] = Decoder().decode(kb) == ks
        print(f'section {s.index} as key section: {len(kb)} bytes, stand-alone roundtrip {rows[i]["key_roundtrip_ok"]}',
              flush=True)
    secs_dur = T / FPS
    tot = {'rans': sum(r['bytes_csgs'] for r in rows),
           'rans_streams_only': sum(sum(v for k, v in r['rans'].items() if k != 'header') for r in rows),
           'zstd19_per_stream': sum(sum(r['zstd19_per_stream'].values()) for r in rows),
           'zstd19_concat': sum(r['zstd19_concat'] for r in rows)}
    per_stream = {nm: {'rans': sum(r['rans'][nm] for r in rows), 'zstd19': sum(r['zstd19_per_stream'][nm] for r in rows),
                       'nsym': sum(r['nsym'][nm] for r in rows)} for nm in names}
    for nm, v in per_stream.items():
        v['saving_pct'] = round(100 * (1 - v['rans'] / v['zstd19']), 1) if v['zstd19'] else None
    best_z = min(tot['zstd19_per_stream'], tot['zstd19_concat'])
    sn_streams = [nm for nm in names if nm.startswith('sn')]
    sn_bytes = sum(per_stream[nm]['rans'] for nm in sn_streams)
    res = {'source': a.source, 'tracks': a.tracks, 'curves': a.curves, 'qpos': a.qpos, 'tol': a.tol,
           'quat': a.quat, 'ktol': a.ktol, 'ktol_group': a.ktol_group, 'csgs_dir': a.write_csgs,
           'sn_vq': {k: v for k, v in (sn_vq or {}).items() if k not in ('from',)} if sn_vq else None,
           'sn_streams_bytes': sn_bytes, 'sn_share_pct': round(100 * sn_bytes / tot['rans'], 2) if tot['rans'] else None,
           'newborn_order': a.newborn_order, 'plan': plan,
           'pred': a.pred if a.pred == 'auto' else parse_pred(a.pred), 'knn_k': a.knn_k,
           'version': 6 if params.bspline else 4 if extra.get('follow_splats') else 3 if a.pred else 2 if sn_vq else 1,
           'basis': 'bspline' if params.bspline else 'hat',
           'sn_deg': a.sn_deg, 'sn_bits': a.sn_bits,
           'keep': a.keep, 'keep_frac': a.keep_frac, 'kept': None if keep is None else int(keep.sum()),
           'key_interval': a.key_interval, 'T': T, 'fps': FPS, 'alive_mean': alive_mean, 'ks': params.ks,
           'shared_knot': params.shared_knot, 'secs_prep': round(t_prep, 1), 'extra': extra, 'sections': rows,
           'per_stream': per_stream, 'total_bytes': tot,
           'mbit_s': {k: round(v * 8 / secs_dur / 1e6, 2) for k, v in tot.items()},
           'bytes_per_splat_frame_rans': round(tot['rans'] / (alive_mean * T), 3),
           'rans_vs_best_zstd_saving_pct': round(100 * (1 - tot['rans'] / best_z), 1),
           'roundtrip_all_ok': all(r['roundtrip_ok'] for r in rows),
           'secs_encode': round(sum(r['secs_encode'] for r in rows), 1),
           'secs_decode': round(sum(r['secs_decode'] for r in rows), 1)}
    if a.source == 'tracks':
        e6b = sum(extra['zstd_bytes_per_gop'])
        res['e6_layout_zstd19_bytes'] = e6b
        res['mbit_s']['e6_layout_zstd19'] = round(e6b * 8 / secs_dur / 1e6, 2)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('total_bytes', 'mbit_s', 'rans_vs_best_zstd_saving_pct', 'roundtrip_all_ok',
                                          'secs_encode', 'secs_decode', 'per_stream')}, indent=1))
    print('P4 OK', a.out)


if __name__ == '__main__':
    main()
