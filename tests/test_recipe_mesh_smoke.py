"""End-to-end smoke of the standard recipe's mesh source on a synthetic scene (CPU only, no third-party data):
tools/make_synthetic_mesh.py -> tools/faltwerk_recipe.py --source mesh -> recipe.sh (mesh rules, mesh2gs, standard
encode) -> the written .falt sections decode with the reference decoder."""
import json
import os
import shlex
import subprocess
import sys

from codec.falt import Decoder, section_files

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def test_mesh_recipe_end_to_end(tmp_path):
    sc = tmp_path / 'scene'
    subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'make_synthetic_mesh.py'), str(sc), '--frames', '36',
                    '--res', '10'], check=True, capture_output=True)
    out = tmp_path / 'run'
    subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'faltwerk_recipe.py'), '--source', 'mesh', '--scene',
                    str(sc / 'scene.json'), '--out', str(out), '--py', shlex.quote(sys.executable)], check=True,
                   capture_output=True)
    r = subprocess.run(['bash', str(out / 'recipe.sh')], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    status = (out / 'status').read_text()
    assert 'FAIL' not in status and status.rstrip().endswith('done'), status
    rules = json.load(open(out / 'rules_p0.json'))
    assert rules['n_backdrop_meshes'] == 1                    # the far static wall gets its own stream
    for part, n_min in (('fg', 2), ('bd', 1)):
        files = section_files(str(out / f'shaded_p0_{part}' / 'enc' / 'falt'))
        assert len(files) >= n_min and all(f.endswith('.falt') for f in files)
        dec = Decoder()
        secs = [dec.decode(open(f, 'rb').read()) for f in files]
        assert sum(s.nframes for s in secs) == 36 or part == 'bd'
        assert all(len(s.ids) > 0 for s in secs)
