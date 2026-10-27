// node bench/decode_bench.ts <dir with sNNNN.csgs> [reps]: wall time of the TS decoder + packSection per section.
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { Decoder } from '../src/csgs.ts';
import { packSection } from '../src/pack.ts';

const dir = process.argv[2];
const reps = Number(process.argv[3] ?? 3);
const files = readdirSync(dir).filter((f) => f.endsWith('.csgs')).sort();
const bufs = files.map((f) => new Uint8Array(readFileSync(join(dir, f))));
for (let r = 0; r < reps; r++) {
  const dec = new Decoder();
  const row: string[] = [];
  for (let i = 0; i < bufs.length; i++) {
    const t0 = performance.now();
    const sec = dec.decode(bufs[i]);
    const t1 = performance.now();
    packSection(sec, dec.params!);
    const t2 = performance.now();
    row.push(`${files[i]} n=${sec.n} decode ${(t1 - t0).toFixed(0)} ms pack ${(t2 - t1).toFixed(0)} ms`);
  }
  console.log(`rep ${r}: ${row.join(' | ')}`);
}
