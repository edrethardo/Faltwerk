// Probe: TS decode + pack time of P8-plyq single-frame .csgs blobs and of our sections.
// usage: node web/test/probe_decode_speed.ts <dir> [maxFiles]   (not part of `npm test`: needs data/)
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { Decoder } from '../src/csgs.ts';

const dir = process.argv[2];
const max = Number(process.argv[3] ?? 5);
const files = readdirSync(dir).filter((f) => f.endsWith('.csgs')).sort().slice(0, max);
const dec = new Decoder();
for (const f of files) {
  const b = new Uint8Array(readFileSync(join(dir, f)));
  const t0 = performance.now();
  const sec = dec.decode(b);
  const t1 = performance.now();
  const p = dec.params!;
  console.log(JSON.stringify({ f, bytes: b.length, n: sec.n, nframes: sec.nframes, nSn: p.nSn, groups: p.groups, ks: p.ks,
    decode_ms: +(t1 - t0).toFixed(1) }));
}
