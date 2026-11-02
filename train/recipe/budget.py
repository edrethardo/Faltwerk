#!/usr/bin/env python3
"""standard recipe: splat budget from a DELIVERY rate target instead of a hand row count.
Ranking = contrib.npz of train/recipe/contrib.py (alpha*T contribution averaged over the frames a splat lives, scored on
EVERY frame); budget B = max rows per .csgs section like contrib.py. B is found by encoding (train/recipe/encode.py,
the standard encode) and a secant search on the rate until 0.93 x aim <= rate <= aim, aim = 0.97 x target (the
prune -> fine-tune that follows moves the rate by a few %).
  budget.py <contrib.npz> <k dir> <out dir> --target 50 [--source images]  -> <out>/{keep.npz, budget.json, try_<B>/}
"""
import argparse, json, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def keep_for(score, in_sec, B):
    order = np.sort(score)
    lo, hi = 0, len(order) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if max(int((m & (score >= order[mid])).sum()) for m in in_sec) <= B:
            hi = mid
        else:
            lo = mid + 1
    return score >= order[lo]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('contrib'); ap.add_argument('kdir'); ap.add_argument('out')
    ap.add_argument('--target', type=float, required=True); ap.add_argument('--source', default='images')
    ap.add_argument('--cams'); ap.add_argument('--set', action='append', default=[]); ap.add_argument('--matte', action='store_true')
    a = ap.parse_args()
    xargs = (['--cams', a.cams] if a.cams else []) + [x for s in a.set for x in ('--set', s)] + (['--matte'] if a.matte else [])
    os.makedirs(a.out, exist_ok=True)
    z = np.load(a.contrib)
    score, birth, death = z['score'], z['birth'], z['death']
    L = int(np.load(os.path.join(a.kdir, 'curves.npz'))['L']); T = int(death.max())
    in_sec = [(birth < min(T, g0 + L)) & (death > g0) for g0 in range(0, T, L)]
    rows_full = max(int(m.sum()) for m in in_sec)
    aim = 0.97 * a.target
    tried = {}

    def rate(B):
        if B in tried:
            return tried[B]
        keep = keep_for(score, in_sec, B)
        d = os.path.join(a.out, f'try_{B}'); os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, 'keep_in.npz'), keep=keep)
        js = [os.path.join(d, n) for n in ('falt.json', 'csgs.json')]     # encode.py summary (csgs.json before )
        if not any(os.path.exists(j) for j in js):
            subprocess.run([sys.executable, '-u', os.path.join(HERE, 'encode.py'), a.kdir, d, '--keep', os.path.join(d, 'keep_in.npz'),
                            '--source', a.source] + xargs, check=True, stdout=open(os.path.join(d, 'encode.log'), 'w'), stderr=subprocess.STDOUT)
        r = json.load(open(next(j for j in js if os.path.exists(j))))['mbit_s']
        tried[B] = r
        print(f'B {B}: {r:.2f} Mbit/s', flush=True)
        return r

    B = rows_full; r = rate(B)
    if r <= a.target:
        best = B
    else:
        b0, r0 = B, r
        B = int(b0 * aim / r0)
        best = None
        for _ in range(5):
            r = rate(B)
            if 0.93 * aim <= r <= aim:
                best = B; break
            b1, r1 = B, r
            B = int(b1 + (aim - r1) * (b1 - b0) / (r1 - r0)) if r1 != r0 else int(b1 * aim / r1)
            b0, r0 = b1, r1
        if best is None:                                    # the largest tried budget under the target
            ok = [b for b, x in tried.items() if x <= a.target]
            best = max(ok) if ok else min(tried)
    keep = keep_for(score, in_sec, best)
    np.savez(os.path.join(a.out, 'keep.npz'), keep=keep)
    res = {'target': a.target, 'aim': aim, 'rows_full': rows_full, 'tried': {str(k): v for k, v in sorted(tried.items())},
           'budget': best, 'rate_before_ft': tried.get(best), 'kept_ids': int(keep.sum())}
    json.dump(res, open(os.path.join(a.out, 'budget.json'), 'w'), indent=1)
    print(f'BUDGET {best} rows/section -> {tried.get(best):.2f} Mbit/s (target {a.target}), kept {int(keep.sum())} ids')


if __name__ == '__main__':
    main()
