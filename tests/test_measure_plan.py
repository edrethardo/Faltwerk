"""(P19): codec.measure --uniform/--plan (S7 tables per section) combined with S1/S2 options.
A uniform plan equal to the defaults writes byte-identical sections; a plan that changes the table per section sets
RESCALE (flag 32 / v5) in the delta sections, composes with --newborn-order morton + --pred auto, and roundtrips."""
import json
import os
import subprocess
import sys

from test_prep_s2 import synth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def measure(tmp_path, name, *args):
    cp, tp = str(tmp_path / 'curves.npz'), str(tmp_path / 'tracks.npz')
    out, d = tmp_path / f'{name}.json', tmp_path / name
    subprocess.run([sys.executable, '-m', 'codec.measure', '--source', 'curves', '--curves', cp, '--tracks', tp,
                    '--quat', 'raw', '--key-interval', '4', '--write-csgs', str(d), '--out', str(out), *args],
                   cwd=ROOT, check=True, capture_output=True)
    res = json.load(open(out))
    files = sorted(os.listdir(d))
    return res, [open(d / f, 'rb').read() for f in files]


def test_uniform_default_plan_is_byte_identical(tmp_path):
    synth(tmp_path)
    r0, b0 = measure(tmp_path, 'plain', '--qpos', '0.001', '--ktol', '1')
    r1, b1 = measure(tmp_path, 'uni', '--uniform', '0.001,255,127,1')
    assert b0 == b1
    assert r1['plan'] is not None and r0['plan'] is None
    assert r1['roundtrip_all_ok']


def test_plan_per_section_rescale_with_morton_and_pred_auto(tmp_path):
    synth(tmp_path)
    plan = json.dumps([{'qpos': 0.001, 'attr_levels': 255, 'rot_levels': 127, 'ktol': 1},
                       {'qpos': 0.0005, 'attr_levels': 511, 'rot_levels': 255, 'ktol': 0}])
    r, bufs = measure(tmp_path, 'plan', '--plan', plan, '--newborn-order', 'morton', '--pred', 'auto')
    assert r['roundtrip_all_ok']
    assert bufs[0][5] & 32 == 0 and bufs[1][5] & 32 == 32 and bufs[1][4] >= 5
    assert [s['flags'] for s in r['sections']] == [b[5] for b in bufs]
    ru, _ = measure(tmp_path, 'uni2', '--uniform', '0.001,255,127,1', '--newborn-order', 'morton', '--pred', 'auto')
    assert ru['roundtrip_all_ok'] and all(s['flags'] & 32 == 0 for s in ru['sections'])
