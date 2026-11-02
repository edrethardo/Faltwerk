#!/usr/bin/env python3
"""compact table of a compare_bench.json (load, fps per pane, sync, PSNR)."""
import json, sys
r = json.load(open(sys.argv[1]))
print('browser', r.get('browser_version'), 'swiftshader', r.get('swiftshader'))
for s in r['scenarios']:
    print(f"\n== {s['name']} vp={s['viewport']} dpr={s['dpr']}  error={s.get('error')}")
    if 'adapter' in s:
        a = s['adapter']; print('  adapter', a.get('vendor'), a.get('architecture'), a.get('description'), 'ts', a.get('timestamps'))
    print('  wall: first frames all panes', s.get('wall_first_frames_s'), 's, all data', s.get('wall_all_loaded_s'), 's')
    for k, v in (s.get('load') or {}).items():
        print(f"  load {k:9s} first {v['first_frame_ms']:6d} ms  all {v['all_loaded_ms']:6d} ms  {v['bytes'] / 2**20:7.1f} MiB  {v['splats']:7d} splats  canvas {v['canvas']}")
    raf = s.get('raf')
    if raf:
        print(f"  rAF {raf['raf_fps']:.1f} Hz, stalls {raf['stalls']}")
        for p in raf['panes']:
            g = p['gpu'] or {}
            print(f"    {p['id']:9s} {p['fps']:5.1f} fps  GPU median {g.get('gpu_total', {}).get('median', float('nan')):.2f} ms (raster {g.get('raster', {}).get('median', float('nan')):.2f}, sort {g.get('sort', {}).get('median', float('nan')):.2f})")
    for k, u in (s.get('uncapped') or {}).items():
        print(f"  uncapped {k:9s} {u['fps']:6.1f} fps  GPU median {u['gpu']['gpu_total']['median']:.2f} ms" if u.get('gpu') else f"  uncapped {k} {u['fps']:.1f}")
    for k in ('sync_playing', 'sync_after_orbit', 'sync_after_step', 'sync_snap'):
        if k in s:
            print(f"  {k}: {s[k]['samples']} samples, {s[k]['mismatches']} mismatches, last {s[k]['last']['frames']}")
    if 'interaction' in s:
        print('  interaction', s['interaction']['view_after'], 'rendered changed', s['interaction']['rendered_camera_changed'])
    if 'psnr' in s:
        for k, v in s['psnr']['summary'].items():
            print(f"  fg-PSNR {k:9s} browser {v['fg_psnr_browser_mean']:.2f} dB vs P8 gsplat {v['fg_psnr_p8_gsplat_mean']} (mean |diff| {v['mean_abs_diff']} dB, n={v['n']}), GPU-vs-CPU eval max rel err {v['eval_check_max_rel_err']:.1e}")
    print('  shots', s.get('screenshot_orbit'), s.get('screenshot_snap'))
