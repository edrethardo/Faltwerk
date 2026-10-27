#!/usr/bin/env python3
"""probe: download throughput of the viewer data via the tailnet HTTPS URL (from the build host itself, 4 parallel requests)."""
import concurrent.futures as cf, sys, time, urllib.request
BASE = sys.argv[1] if len(sys.argv) > 1 else 'https://the build host.localhost:8453/'
urls = [f'{BASE}james/4danyone/{i:05d}.splq' for i in range(50)]
t0 = time.time()
with cf.ThreadPoolExecutor(4) as ex:
    n = sum(ex.map(lambda u: len(urllib.request.urlopen(u).read()), urls))
d = time.time() - t0
print(f'{len(urls)} files, {n / 2**20:.0f} MiB in {d:.1f} s = {n / 2**20 / d:.0f} MiB/s via {BASE}')
