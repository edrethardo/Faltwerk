// where does the spatial-mode decode time go? times causalKnn on the real the demo capture section's knot-0 positions.
// usage: node web/bench/probe_knn_time.ts <dir with s0000.csgs>
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Decoder } from '../src/csgs.ts';
import { causalKnn } from '../src/knn.ts';

const dir = process.argv[2];
const d = new Decoder();
const sec = d.decode(new Uint8Array(readFileSync(join(dir, 's0000.csgs'))));
const gp = d.params!.groups.findIndex(([t]) => t === 'P');
const pos = new Int32Array(sec.n * 3);
for (let i = 0; i < sec.n; i++) for (let c = 0; c < 3; c++) pos[i * 3 + c] = sec.knots[gp][sec.knotOff[gp][i] + c];
for (let r = 0; r < 5; r++) {
  const t0 = performance.now();
  causalKnn(pos, sec.n, d.params!.knnK || 4);
  console.log(`causalKnn n=${sec.n} k=${d.params!.knnK || 4}: ${(performance.now() - t0).toFixed(0)} ms`);
}
