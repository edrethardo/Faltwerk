"""web/make_scene.py: served root for the single-scene player from a section directory (CPU, no browser)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'web' / 'test' / 'fixtures'


def test_make_scene_reads_section_headers(tmp_path):
    src = tmp_path / 'falt'
    src.mkdir()
    files = sorted(FIX.glob('synth_shared_s000*.falt')) or sorted(FIX.glob('synth_shared_s000*.csgs'))
    assert len(files) >= 2
    for f in files:
        shutil.copy2(f, src / f.name)
    root = tmp_path / 'root'
    p = subprocess.run([sys.executable, str(ROOT / 'web' / 'make_scene.py'), str(root), str(src), '--no-dist',
                        '--dist', '6'], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    scene = json.loads((root / 'scene.json').read_text())
    secs = scene['sections']
    assert [s['url'] for s in secs] == [f'data/{f.name}' for f in files]
    assert all((root / s['url']).is_file() for s in secs)
    assert secs[0]['g0'] == 0 and all(s['nframes'] > 0 for s in secs)
    for a, b in zip(secs, secs[1:]):
        assert b['g0'] == a['g0'] + a['nframes']        # contiguous timeline
    assert scene['frames'] == sum(s['nframes'] for s in secs)
    assert scene['orbit']['dist'] == 6 and scene['orbit']['up'] == [0.0, 1.0, 0.0] and scene['testcams'] == []


def test_make_scene_rejects_foreign_files(tmp_path):
    src = tmp_path / 'falt'
    src.mkdir()
    (src / 's0000.falt').write_bytes(b'GLTF' + bytes(40))
    p = subprocess.run([sys.executable, str(ROOT / 'web' / 'make_scene.py'), str(tmp_path / 'root'), str(src),
                        '--no-dist'], capture_output=True, text=True)
    assert p.returncode != 0 and 'not a Faltwerk section' in (p.stderr + p.stdout)
