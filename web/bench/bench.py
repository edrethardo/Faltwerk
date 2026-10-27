#!/usr/bin/env python3
"""P5 benchmark: headless Chromium (WebGPU, Vulkan) on the GPU box plays .csgs sections with web/dist.

Runs on the GPU host in the sdk venv (Playwright). Needs the GPU lock (taken by run_bench.sh on the build host, owner project).
  python bench.py <serve_dir> <csgs_dir> <cameras.json> <out_dir>
Serves <serve_dir> (index.html, main.js, worker.js) over http://127.0.0.1:<port>/, writes <serve_dir>/data/scene.json
(+ symlinks to the .csgs sections), then per scenario a fresh page: time to first frame, rAF-paced fps, uncapped
throughput with per-pass GPU timestamps, GPU memory (own allocations + nvidia-smi delta), screenshots (desktop,
iPhone viewport 390x844) and test-camera renders (raw RGB, compared against GT/gsplat by ref_render.py).
"""
import functools
import http.server
import json
import math
import os
import subprocess
import sys
import threading
import time

from playwright.sync_api import sync_playwright

CHROME = os.environ.get('P5_CHROME', os.path.expanduser('~/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'))
ARGS = ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=vulkan', '--use-vulkan',
        '--ignore-gpu-blocklist', '--enable-gpu', '--disable-vulkan-surface', '--enable-webgpu-developer-features']
if os.environ.get('P5_SWIFTSHADER') == '1':      # CPU WebGPU for functional tests without the GPU lock
    ARGS = ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-webgpu-adapter=swiftshader',
            '--use-angle=swiftshader', '--enable-features=WebGPU', '--enable-webgpu-developer-features']
TEST_FRAMES = [0, 20, 40, 63]
SMOKE = os.environ.get('P5_SMOKE') == '1'


def gpu_mem_mib():
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def make_scene(serve, csgs_dir, cams_path):
    meta = json.load(open(cams_path))
    data = os.path.join(serve, 'data')
    os.makedirs(data, exist_ok=True)
    files = sorted(f for f in os.listdir(csgs_dir) if f.endswith('.csgs'))
    sections, g0 = [], 0
    for f in files:
        dst = os.path.join(data, f)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.join(csgs_dir, f), dst)
        with open(os.path.join(csgs_dir, f), 'rb') as fh:
            h = fh.read(20)
        sg0 = int.from_bytes(h[10:14], 'little'); nf = int.from_bytes(h[16:18], 'little')
        sections.append(dict(url=f, g0=sg0, nframes=nf))
        g0 = sg0 + nf
    cams = meta['cameras']
    test = [(i, c) for i, c in enumerate(cams) if c['split'] == 'test']
    center, up = meta['center'], meta['up']
    nu = math.sqrt(sum(u * u for u in up)); up = [u / nu for u in up]
    R, t = test[0][1]['R'], test[0][1]['t']
    pos = [-(R[0][c] * t[0] + R[1][c] * t[1] + R[2][c] * t[2]) for c in range(3)]
    v = [pos[i] - center[i] for i in range(3)]
    dist = math.sqrt(sum(x * x for x in v))
    vu = sum(v[i] * up[i] for i in range(3))
    front = [v[i] - vu * up[i] for i in range(3)]
    nf = math.sqrt(sum(x * x for x in front)); front = [x / nf for x in front]
    elev = math.degrees(math.asin(max(-1, min(1, vu / dist))))
    c0 = test[0][1]
    scene = dict(fps=meta['fps'], frames=g0, sections=sections,
                 orbit=dict(center=center, up=up, front=front, dist=dist, elev_deg=elev,
                            fovy_deg=math.degrees(2 * math.atan(0.5 * c0['h'] / c0['fy']))),
                 testcams=[dict(name=f'c{i:02d}', R=c['R'], t=c['t'], fx=c['fx'], fy=c['fy'], cx=c['cx'], cy=c['cy'],
                                w=c['w'], h=c['h']) for i, c in test])
    json.dump(scene, open(os.path.join(data, 'scene.json'), 'w'))
    return scene


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


def main():
    serve, csgs_dir, cams_path, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    scene = make_scene(serve, csgs_dir, cams_path)
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=serve))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{port}/index.html?scene=data/scene.json'
    res = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), chrome=CHROME, args=ARGS, scene=dict(
        sections=scene['sections'], fps=scene['fps'], frames=scene['frames']), gpu_mem_idle_mib=gpu_mem_mib(),
        scenarios=[])
    print('idle GPU MiB', res['gpu_mem_idle_mib'], flush=True)
    raf_s, nunc = (1, int(os.environ.get('P5_NUNC', '60'))) if SMOKE else (5, 600)
    scen = [
        dict(name='desktop_1280x720_real', vp=(1280, 720), n=0, shot='desktop_1280x720.png', tests=True),
        dict(name='desktop_1920x1080_real', vp=(1920, 1080), n=0),
        dict(name='desktop_1280x720_150k', vp=(1280, 720), n=150000),
        dict(name='desktop_1920x1080_150k', vp=(1920, 1080), n=150000),
        dict(name='desktop_1920x1080_300k', vp=(1920, 1080), n=300000),
        dict(name='desktop_1920x1080_600k', vp=(1920, 1080), n=600000),
        dict(name='iphone_390x844_dpr3_real', vp=(390, 844), n=0, mobile=True, dpr=3, shot='iphone_390x844.png'),
        dict(name='iphone_390x844_dpr3_150k', vp=(390, 844), n=150000, mobile=True, dpr=3),
    ]
    if SMOKE:
        scen = scen[:1] + scen[6:7]
    if os.environ.get('P5_SCENARIOS'):
        keep = os.environ['P5_SCENARIOS'].split(',')
        scen = [s for s in scen if s['name'] in keep]
    with sync_playwright() as pw:
        br = pw.chromium.launch(executable_path=CHROME, headless=True, args=ARGS)
        res['browser_version'] = br.version
        for sc in scen:
            print('==', sc['name'], flush=True)
            dpr = sc.get('dpr', 1)
            ctx = br.new_context(viewport=dict(width=sc['vp'][0], height=sc['vp'][1]), device_scale_factor=dpr,
                                 is_mobile=sc.get('mobile', False), has_touch=sc.get('mobile', False))
            page = ctx.new_page()
            logs = []
            page.on('console', lambda m: logs.append(f'{m.type}: {m.text}'))
            page.on('pageerror', lambda e: logs.append(f'pageerror: {e}'))
            extra = os.environ.get('P5_URL_EXTRA', '')
            for attempt in range(2):
                url = base + f'&n={sc["n"]}&dpr={dpr}' + extra
                t0 = time.time()
                page.goto(url)
                r = dict(name=sc['name'], viewport=sc['vp'], dpr=dpr, target_n=sc['n'], url=url)
                try:
                    page.wait_for_function('window.__p5 && (window.__p5.error || window.__p5.marks.first_frame_ms)',
                                           timeout=int(os.environ.get('P5_LOAD_TIMEOUT_MS', '120000')))
                    err = page.evaluate('window.__p5.error || null')
                except Exception as e:      # noqa: BLE001 - record and continue with the next scenario
                    err = f'no first frame: {e}'.splitlines()[0] + ' | hud: ' + page.evaluate("document.getElementById('stats').textContent")
                if err and 'offscreen=1' not in extra and attempt == 0:
                    print('  canvas path failed, retry without swap chain:', err, flush=True)
                    res.setdefault('canvas_failures', []).append(dict(name=sc['name'], error=err, console=logs[-10:]))
                    extra += '&offscreen=1'
                    continue
                break
            if err:
                r['error'] = err; r['console'] = logs[-30:]
                res['scenarios'].append(r); print('ERROR', err, flush=True); ctx.close(); continue
            r['wall_to_first_frame_s'] = round(time.time() - t0, 3)
            r['adapter'] = page.evaluate('window.__p5.info')
            r['marks_ms'] = page.evaluate('window.__p5.marks')
            r['raf'] = page.evaluate(f'window.__p5.rafFps({raf_s})')
            r['gpu_mem_used_mib'] = gpu_mem_mib()
            r['uncapped'] = page.evaluate(f'window.__p5.uncapped({nunc}, true)')
            r['uncapped_no_timestamps'] = page.evaluate(f'window.__p5.uncapped({nunc}, false)')
            r['state'] = page.evaluate('window.__p5.state()')
            r['section_timings'] = page.evaluate('window.__p5.timings')
            if sc.get('shot'):
                page.evaluate('window.__p5.setView({t: 20, playing: false})')
                page.wait_for_timeout(1000)
                # headless screenshots do not reliably capture a WebGPU canvas (mobile emulation came out white):
                # draw the renderer's own float target (same clamp as the blit) into a 2D canvas first
                r['screenshot_via'] = 'readback of the render target into a 2D canvas'
                page.evaluate('window.__p5.snapshot()')
                page.screenshot(path=os.path.join(out, sc['shot']), scale='css')
                r['screenshot'] = sc['shot']
            if sc.get('tests'):
                r['test_renders'] = []
                for ci, cam in enumerate(scene['testcams']):
                    for f in TEST_FRAMES:
                        try:
                            tr = page.evaluate(f'window.__p5.renderTest({ci}, {f})')
                        except Exception as e:  # noqa: BLE001
                            r['test_renders'].append(dict(cam=cam['name'], frame=f, error=str(e)[:300]))
                            continue
                        import base64
                        fn = f'browser_{cam["name"]}_f{f:03d}.rgb'
                        open(os.path.join(out, fn), 'wb').write(base64.b64decode(tr.pop('rgb_b64')))
                        tr.update(cam=cam['name'], frame=f, file=fn)
                        r['test_renders'].append(tr)
            r['console'] = logs[-20:]
            res['scenarios'].append(r)
            u = r['uncapped']
            print(f"  ttff {r['marks_ms'].get('first_frame_ms', 0):.0f} ms, raf {r['raf']['fps']:.1f} fps, uncapped {u['fps']:.1f} fps, "
                  f"no-ts {r['uncapped_no_timestamps']['fps']:.1f} fps, n={r['state']['visible']} visible, "
                  f"own GPU {r['state']['gpu_bytes'] / 2**20:.0f} MiB, nvsmi {r['gpu_mem_used_mib']} MiB", flush=True)
            ctx.close()
        br.close()
    httpd.shutdown()
    res['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(res, open(os.path.join(out, 'bench.json'), 'w'), indent=1)
    print('wrote', os.path.join(out, 'bench.json'), flush=True)


if __name__ == '__main__':
    main()
