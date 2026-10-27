// decode time per section in node (TS decoder; key section + delta section, plus pack for the GPU upload).
// usage: node web/bench/probe_decode_s6.ts <out.json> <label=dir> [<label=dir> ...]   (dir holds s0000.csgs, s0001.csgs; needs data/)
// Each stream is decoded REPS times with a fresh Decoder (first rep = cold JIT, reported separately); the JSON holds
// cold / median / min of decode_ms and pack_ms per section, plus the stream facts (version, flags, n, bytes).
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { cpus } from 'node:os';
import { Decoder } from '../src/csgs.ts';
import { packSection } from '../src/pack.ts';

const REPS = Number(process.env.REPS ?? 7);
const [out, ...specs] = process.argv.slice(2);
const med = (a: number[]) => [...a].sort((x, y) => x - y)[a.length >> 1];
const res: Record<string, unknown> = { host: cpus()[0].model, node: process.version, reps: REPS, streams: {} };
for (const spec of specs) {
  const [label, dir] = spec.split('=');
  const bufs = ['s0000', 's0001'].map((n) => new Uint8Array(readFileSync(join(dir, `${n}.csgs`))));
  const dec: number[][] = [[], []], pk: number[][] = [[], []];
  let info: Record<string, unknown> = {};
  for (let r = 0; r < REPS; r++) {
    const d = new Decoder();
    bufs.forEach((b, i) => {
      const t0 = performance.now();
      const sec = d.decode(b);
      const t1 = performance.now();
      packSection(sec, d.params!);
      const t2 = performance.now();
      dec[i].push(t1 - t0); pk[i].push(t2 - t1);
      if (r === 0) info = { ...info, [`s000${i}`]: { bytes: b.length, version: b[4], flags: b[5], n: sec.n, nframes: sec.nframes, key: sec.key,
        pred_modes: d.params!.pred ? Array.from(d.params!.pred) : null } };
    });
  }
  const f = (a: number[]) => ({ cold_ms: +a[0].toFixed(1), median_ms: +med(a.slice(1)).toFixed(1), min_ms: +Math.min(...a.slice(1)).toFixed(1) });
  const secs = bufs.map((_, i) => ({ ...(info as Record<string, any>)[`s000${i}`], decode: f(dec[i]), pack: f(pk[i]) }));
  (res.streams as Record<string, unknown>)[label] = { dir, sections: secs, section_duration_s: 1.28 };
  console.log(label, JSON.stringify(secs.map((s) => [s.bytes, s.decode, s.pack])));
}
writeFileSync(out, JSON.stringify(res, null, 1));
