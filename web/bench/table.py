#!/usr/bin/env python3
"""markdown tables from a P5 bench.json (+ psnr.json next to it). usage: python table.py <bench.json>"""
import json
import os
import sys


def f(v, nd=1):
    return '–' if v is None else f'{v:.{nd}f}'


def main():
    path = sys.argv[1]
    b = json.load(open(path))
    rows = []
    hdr = ('| Szenario | Splats (sichtbar) | rAF-fps | ungedrosselt fps | GPU/Frame ms (Median) | Kurven-Eval ms | '
           'Projektion+SH ms | Sort ms | Raster ms | eigene GPU-Puffer MiB | nvidia-smi Δ MiB | erster Frame ms |')
    rows.append(hdr)
    rows.append('|' + '|'.join(['---'] * 12) + '|')
    idle = b.get('gpu_mem_idle_mib')
    for s in b['scenarios']:
        if s.get('error'):
            rows.append(f"| {s['name']} | Fehler: {s['error'][:80]} |")
            continue
        g = (s['uncapped'].get('gpu_timings') or {})
        med = lambda k: (g.get(k) or {}).get('median')
        st = s['state']
        n = s['section_timings'][0]['n_coded'] if not s['target_n'] else s['target_n']
        mem = s.get('gpu_mem_used_mib')
        rows.append(f"| {s['name']} | {max(n, s['target_n'] or 0):,} ({st['visible']:,}) | {f(s['raf']['fps'])} | "
                    f"{f(s['uncapped_no_timestamps']['fps'])} | {f(med('gpu_total'), 2)} | {f(med('eval'), 3)} | "
                    f"{f(med('project'), 3)} | {f(med('sort'), 3)} | {f(med('raster'), 3)} | "
                    f"{st['gpu_bytes'] / 2**20:.0f} | {mem - idle if mem is not None and idle is not None else '–'} | "
                    f"{f(s['marks_ms'].get('first_frame_ms'), 0)} |")
    print('\n'.join(rows))
    s0 = next((s for s in b['scenarios'] if not s.get('error')), None)
    if s0:
        print('\n| Abschnitt | Bytes | Schlüssel | Splats | Frames | Laden ms | Dekodieren ms | ms/Frame | Packen ms | GPU-Upload ms |')
        print('|---|---|---|---|---|---|---|---|---|---|')
        for t in s0['section_timings'][:4]:
            print(f"| {t['index']} (seq {t['seq']}) | {t['bytes']:,} | {'ja' if t['key'] else 'nein'} | {t['n_coded']:,} | "
                  f"{t['nframes']} | {f(t['fetch_ms'])} | {f(t['decode_ms'])} | {f(t['decode_ms'] / t['nframes'], 2)} | "
                  f"{f(t['pack_ms'])} | {f(t.get('upload_ms'))} |")
    pj = os.path.join(os.path.dirname(path), 'psnr.json')
    if os.path.exists(pj):
        p = json.load(open(pj))
        print('\n| Kamera | Frame | fg-PSNR Browser/GT | fg-PSNR gsplat/GT | fg-PSNR Browser/gsplat | max |Δ| 8 bit |')
        print('|---|---|---|---|---|---|')
        for r in p['rows']:
            print(f"| {r['cam']} | {r['frame']} | {r['browser_vs_gt']['fg_psnr']:.2f} | {r['gsplat_vs_gt']['fg_psnr']:.2f} | "
                  f"{r['browser_vs_gsplat']['fg_psnr']:.2f} | {r['browser_vs_gsplat']['max_abs_8bit']} |")
        print('\n' + json.dumps(p['summary']))


if __name__ == '__main__':
    main()
