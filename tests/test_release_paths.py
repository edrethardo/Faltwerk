"""CPU regressions for camera metadata and the encoder's optional opacity floor."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('meta,expected', [
    ({'width': 48, 'height': 32}, (48, 32)),
    ({'res': 24}, (24, 24)),
    ({'width': 48, 'height': 32, 'res': 24}, (48, 32)),
])
def test_camera_dimensions(meta, expected):
    # Execute the real dimension assignments, without importing the CUDA trainer.
    tree = ast.parse((ROOT / 'train/e3_track.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in ('W', 'H') for t in n.targets)]
    assert len(nodes) == 2
    scope = {'meta': meta}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<dimensions>', 'exec'), scope)
    assert (scope['W'], scope['H']) == expected


def encoder_floor_path():
    # Resolve the helper expression actually passed to subprocess.run by encode.py.
    tree = ast.parse((ROOT / 'train/recipe/encode.py').read_text())
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'join' and any(
                    isinstance(a, ast.Constant) and a.value == 'deep_floor.py' for a in n.args)):
            import os
            return Path(eval(compile(ast.Expression(n), '<helper path>', 'eval'),
                             {'os': os, 'ROOT': str(ROOT)}))
    raise AssertionError('encoder does not reference its floor helper')


@pytest.mark.parametrize('soft', [False, True])
def test_encoder_floor_helper(tmp_path, soft):
    helper = encoder_floor_path()
    assert helper.is_file(), 'encoder calls an unexported helper'
    opacity = np.array([[-7., -7., -7., -7., -7.],
                        [-7., -7., 0., -7., -7.],
                        [-15., -15., -15., -15., -15.]], dtype=np.float32)
    src, dst = tmp_path / 'curves.npz', tmp_path / 'deep.npz'
    np.savez(src, basis='hat', soft_knot0=soft, g0_opacities=opacity,
             g0_means=np.arange(9, dtype=np.float32).reshape(3, 3))
    p = subprocess.run([sys.executable, str(helper), str(src), str(dst)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    with np.load(dst) as z:
        expected = opacity.copy()
        if soft:
            expected[0] = -12
            expected[1, [0, 4]] = -12
        else:
            expected[0, 1:4] = -12
        np.testing.assert_array_equal(z['g0_opacities'], expected)
        np.testing.assert_array_equal(z['g0_means'], np.arange(9).reshape(3, 3))
    stats = json.loads(dst.with_suffix('.json').read_text())
    assert stats['deepened'] == (7 if soft else 3)
