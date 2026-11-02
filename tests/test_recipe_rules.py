"""standard recipe: rules on synthetic data + the recipe generator holds no scene values."""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from train.recipe import recipe_rules as rr  # noqa: E402
from train.recipe.budget import keep_for  # noqa: E402


def _curves(path, n=400, K=3, seed=0):
    rng = np.random.default_rng(seed)
    d = {'L': np.array(32), 'K': np.array(K), 'knots': np.array('hat'), 'soft_knot0': np.array(True), 'basis': np.array('hat')}
    for g in range(2):
        ids = np.arange(n, dtype=np.int32)
        m = rng.normal(0, 1, (n, 1, 3)).repeat(K, 1).astype(np.float32)
        m[0] = 500.0                                            # one runaway
        s = np.full((n, K, 3), np.log(0.01), np.float32)
        s[1] = np.log(3.0)                                      # one giant
        o = np.full((n, K), 3.0, np.float32)
        o[2] = -9.0                                             # one dead (never >= 1/255)
        d.update({f'g{g}_ids': ids, f'g{g}_means': m, f'g{g}_scales': s, f'g{g}_opacities': o,
                  f'g{g}_quats': np.zeros((n, K, 4), np.float32), f'g{g}_sh0': np.zeros((n, K, 3), np.float32)})
    np.savez(path, **d)


def test_encoder_rules(tmp_path):
    p = tmp_path / 'curves.npz'; _curves(p)
    st = rr.curve_stats(str(p))
    d = rr.derive_encoder(st)
    assert st['summary']['n_ids'] == 400
    assert d['giant_n'] == 1 and d['runaway_n'] == 1
    assert abs(d['sigma_fine'] - 0.01) < 1e-6
    assert abs(d['step'] - rr.DEFAULTS['qstep_rel'] * 0.01) < 1e-9
    assert d['ext_kept'] < 20                                   # the runaway does not blow up the position grid
    assert st['O'][2] < 1 / 255                                 # dead splat detected


def test_footprint(tmp_path):
    cams = {'cameras': [{'R': np.eye(3).tolist(), 't': [0, 0, 5.0], 'fx': 500.0, 'split': 'test'}]}
    (tmp_path / 'cameras.json').write_text(json.dumps(cams))
    fp = rr.footprint_from_cams(str(tmp_path / 'cameras.json'), np.zeros((10, 3)))
    assert abs(fp - 0.01) < 1e-9


def test_budget_keep():
    score = np.array([5, 4, 3, 2, 1], np.float32)
    in_sec = [np.array([1, 1, 1, 1, 1], bool), np.array([0, 0, 1, 1, 1], bool)]
    keep = keep_for(score, in_sec, 3)
    assert keep.tolist() == [True, True, True, False, False]


def test_recipe_has_no_scene_values(tmp_path):
    sc = {'name': 'x', 'gt': '/nonexistent/gt', 'frames': 64, 'kind': 'llff'}
    (tmp_path / 's.json').write_text(json.dumps(sc))
    out = tmp_path / 'run'
    subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'faltwerk_recipe.py'), '--source', 'images', '--scene',
                    str(tmp_path / 's.json'), '--out', str(out)], check=True, capture_output=True)
    sh = (out / 'recipe.sh').read_text()
    cfg = json.load(open(os.path.join(ROOT, 'tools', 'recipe_standard.json')))
    assert cfg['images']['init'] in sh and 'E7_STATIC_LR=0.3' in sh
    for bad in ('flame', 'coffee', 'sear', 'scene_a', 'scene_b'):  # the generator never branches on a scene name
        assert bad not in sh.lower()
    subprocess.run(['bash', '-n', str(out / 'recipe.sh')], check=True)
