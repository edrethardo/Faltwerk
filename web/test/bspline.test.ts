// (P16): .csgs v6 / flag BSPLINE in the TS decoder. Fixtures: web/test/make_fixture.py (synth_bsp_*, synth_bsp_weights.json).
//  - weights: bit-identical to codec.csgs.bspline_weights (IEEE double hex patterns), local form within 1e-12
//  - decode: integers of a v6 stream (key/delta/key, shared knot, K up to 9) equal the Python decoder (CRC32 of every array)
//  - curve evaluation (pack.ts evalSplat, v6 branch) vs the numpy oracle of dump_ref.py
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Decoder, crc32, VERSION_BSPLINE } from '../src/csgs.ts';
import { bsplineWeights, bsplineLocal } from '../src/bspline.ts';
import { packSection, evalSplat } from '../src/pack.ts';

const FIX = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');
const hex = (x: number) => Buffer.from(new Float64Array([x]).buffer).toString('hex');

test('B-spline weights: bit-identical to the Python reference, local form equal to 1e-12, partition of unity', () => {
  const rows = JSON.parse(readFileSync(join(FIX, 'synth_bsp_weights.json'), 'utf8'));
  assert.ok(rows.length >= 100);
  for (const r of rows) {
    const w = bsplineWeights(r.K, r.u);
    assert.deepEqual(w.map(hex), r.bits, `K=${r.K} u=${r.u}`);
    const loc = bsplineLocal(r.K, r.u);
    const full = new Array(r.K).fill(0);
    loc.w.forEach((x, i) => { full[loc.s0 + i] = x; });
    full.forEach((x, i) => assert.ok(Math.abs(x - w[i]) < 1e-12, `local K=${r.K} u=${r.u} i=${i}: ${x} vs ${w[i]}`));
    assert.ok(Math.abs(w.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  }
  // dense sweep (no oracle): local == full everywhere, ends interpolate the end control points
  for (const K of [2, 3, 4, 5, 8, 9, 13]) {
    for (let i = 0; i <= 200; i++) {
      const u = i / 200, w = bsplineWeights(K, u), loc = bsplineLocal(K, u);
      loc.w.forEach((x, j) => assert.ok(Math.abs(x - w[loc.s0 + j]) < 1e-12));
    }
    assert.equal(bsplineWeights(K, 0)[0], 1); assert.equal(bsplineWeights(K, 1)[K - 1], 1);
  }
});

function i32crc(a: Int32Array): number { return crc32(new Uint8Array(a.buffer, a.byteOffset, a.byteLength)); }

test('v6 synthetic stream (key/delta/key, shared knot, K up to 9): integers equal the Python decoder, curve evaluation matches the oracle', () => {
  const dec = new Decoder();
  for (const nm of ['synth_bsp_s0000', 'synth_bsp_s0001', 'synth_bsp_s0002']) {
    const buf = new Uint8Array(readFileSync(join(FIX, `${nm}.csgs`)));
    assert.equal(buf[4], VERSION_BSPLINE); assert.ok(buf[5] & 64);
    const ref = JSON.parse(readFileSync(join(FIX, `${nm}.ref.json`), 'utf8'));
    const sec = dec.decode(buf);
    assert.equal(dec.params!.bspline, true);
    assert.equal(sec.n, ref.n);
    const mine: Record<string, Int32Array> = {
      ids: new Int32Array(sec.ids.buffer.slice(0)), birth: Int32Array.from(sec.birth), death: Int32Array.from(sec.death),
    };
    dec.params!.groups.forEach(([t], g) => { mine[`k_${t}`] = Int32Array.from(sec.k[g]); mine[`knots_${t}`] = sec.knots[g]; });
    mine.sn = Int32Array.from(sec.sn);
    for (const a of ref.arrays) assert.equal(i32crc(mine[a.name]), a.crc32, `${nm}.${a.name}`);
    assert.ok(Math.max(...Array.from(sec.k[0])) >= 5, 'K >= 5 occurs');
    const ps = packSection(sec, dec.params!);
    assert.ok(ps.hdr[2] >>> 31, 'bit 31 of the K word marks the B-spline basis');
    let maxErr = 0;
    for (const s of ref.curve.splats) {
      ref.curve.times.forEach((t: number, ti: number) => {
        const { vals } = evalSplat(ps, s.i, t);
        ['P', 'R', 'S', 'O', 'D'].forEach((tag, g) => {
          (s.vals[tag][ti] as number[]).forEach((w, c) => { maxErr = Math.max(maxErr, Math.abs(vals[g][c] - w) / Math.max(1, Math.abs(w))); });
        });
      });
    }
    assert.ok(maxErr < 1e-5, `${nm}: curve eval max rel err ${maxErr}`);       // f32 knots; cubic weights amplify the synthetic huge jumps a bit more than the hat (2e-6)
  }
});

test('BSPLINE flag and version must agree', () => {
  const bad = new Uint8Array(readFileSync(join(FIX, 'synth_bsp_s0000.csgs')));
  bad[4] = 1;
  assert.throws(() => new Decoder().decode(bad), /version and flags/);
  const bad2 = new Uint8Array(readFileSync(join(FIX, 'synth_res_s0000.csgs')));
  bad2[5] |= 64;
  assert.throws(() => new Decoder().decode(bad2), /version and flags/);
});
