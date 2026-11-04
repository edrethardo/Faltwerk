#!/usr/bin/env python3
"""Build a served root for the single-scene player (index.html) from a directory of Faltwerk sections.

  python3 web/make_scene.py <root> <falt_dir> [--center X,Y,Z] [--up X,Y,Z] [--front X,Y,Z]
                            [--dist D] [--elev DEG] [--fovy DEG] [--fps N] [--no-dist]

Copies web/dist/* (run `npm run build` first) and the sections (*.falt, legacy *.csgs) into <root>, reads each section's
first frame and frame count from its header and writes <root>/scene.json. Serve it with
`python3 web/serve_viewer.py <root>` and open http://127.0.0.1:8790/index.html. Orbit defaults fit
tools/make_synthetic_mesh.py (y up, object near the origin).
"""
import argparse
import json
import shutil
from pathlib import Path

WEB = Path(__file__).resolve().parent


def vec(s):
    v = [float(x) for x in s.split(',')]
    if len(v) != 3:
        raise argparse.ArgumentTypeError('expected X,Y,Z')
    return v


def section_header(path):
    h = path.read_bytes()[:20]
    if h[:4] not in (b'FALT', b'CSGS'):
        raise SystemExit(f'{path}: not a Faltwerk section (magic {h[:4]!r})')
    return int.from_bytes(h[10:14], 'little'), int.from_bytes(h[16:18], 'little')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', type=Path)
    ap.add_argument('falt_dir', type=Path)
    ap.add_argument('--center', type=vec, default=[0.0, 0.2, 0.0])
    ap.add_argument('--up', type=vec, default=[0.0, 1.0, 0.0])
    ap.add_argument('--front', type=vec, default=[0.0, 0.0, 1.0], help='direction from the center to the camera')
    ap.add_argument('--dist', type=float, default=4.0)
    ap.add_argument('--elev', type=float, default=8.0)
    ap.add_argument('--fovy', type=float, default=40.0)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--no-dist', action='store_true', help='do not copy web/dist (root already holds the viewer)')
    a = ap.parse_args()

    files = sorted(p for p in a.falt_dir.iterdir() if p.suffix in ('.falt', '.csgs'))
    if not files:
        raise SystemExit(f'no .falt/.csgs sections in {a.falt_dir}')
    data = a.root / 'data'
    data.mkdir(parents=True, exist_ok=True)
    if not a.no_dist:
        dist = WEB / 'dist'
        if not (dist / 'index.html').exists():
            raise SystemExit('web/dist missing: run `npm ci && npm run build` in web/ first')
        for f in dist.iterdir():
            if f.is_file():
                shutil.copy2(f, a.root / f.name)
    sections = []
    for f in files:
        shutil.copy2(f, data / f.name)
        g0, n = section_header(f)
        sections.append({'url': f'data/{f.name}', 'g0': g0, 'nframes': n})
    scene = {'fps': a.fps, 'frames': sum(s['nframes'] for s in sections), 'sections': sections,
             'orbit': {'center': a.center, 'up': a.up, 'front': a.front, 'dist': a.dist,
                       'elev_deg': a.elev, 'fovy_deg': a.fovy},
             'testcams': []}
    (a.root / 'scene.json').write_text(json.dumps(scene, indent=1))
    print(f"{a.root / 'scene.json'}: {len(sections)} sections, {scene['frames']} frames")


if __name__ == '__main__':
    main()
