// Decoder tests against the Python reference (codec/csgs.py) — run: npm test  (node --test, type stripping).
// Synthetic fixtures (web/test/fixtures, made by make_fixture.py) always run; the real P4 sections run when
// CSGS_REF_DIR points at a directory with s0000.csgs … and their *.ref.{json,bin} (dump_ref.py; not in the repo).
// v2 (SN_VQ, ): synthetic fixture synth_vq_*; the real S1 sections (ft_d3_cb256_i6000_lr0.3) run when
// CSGS_S1_REF_DIR (default data/run/s1) holds s0000.csgs, s0001.csgs and their refs.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Decoder, crc32, maxStages, type Params, type Section } from '../src/csgs.ts';
import { causalKnn } from '../src/knn.ts';
import { packSection, evalSplat } from '../src/pack.ts';

const FIX = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');
const REAL = process.env.CSGS_REF_DIR ?? '';
const REAL_S1 = process.env.CSGS_S1_REF_DIR ?? '';
// the demo capture S1+S2 (v3 pred auto + SN_VQ + Morton), dump by web/bench/make_s6_refs.sh
const REAL_S6 = process.env.CSGS_S6_REF_DIR ?? '';

function arraysOf(sec: Section, p: Params): Record<string, Int32Array> {
  const out: Record<string, Int32Array> = {
    ids: new Int32Array(sec.ids.buffer.slice(0)), birth: Int32Array.from(sec.birth), death: Int32Array.from(sec.death),
  };
  p.groups.forEach(([t], g) => { out[`k_${t}`] = Int32Array.from(sec.k[g]); out[`knots_${t}`] = sec.knots[g]; });
  out.sn = Int32Array.from(sec.sn);
  if (p.snVq) {          // same flat order as dump_ref.py: per splat the indices of its level's stages; codebooks level/stage/entry/scalar
    out.sn_deg = Int32Array.from(sec.snDeg!);
    const ms = maxStages(p), idx: number[] = [];
    for (let i = 0; i < sec.n; i++) {
      const dg = sec.snDeg![i];
      const ns = dg ? p.snVq.find((l) => l.deg === dg)!.sizes.length : 0;
      for (let s = 0; s < ns; s++) idx.push(sec.snIdx![i * ms + s]);
    }
    out.sn_idx = Int32Array.from(idx);
    if (sec.snCb) out.sn_cb = Int32Array.from(sec.snCb.flatMap((lvl) => lvl.flatMap((st) => Array.from(st))));
  }
  return out;
}

function i32crc(a: Int32Array): number {
  return crc32(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
}

function checkStream(dir: string, refDir: string, names: string[], tol = 2e-6) {
  const dec = new Decoder();
  for (const nm of names) {
    const buf = new Uint8Array(readFileSync(join(dir, `${nm}.csgs`)));
    const ref = JSON.parse(readFileSync(join(refDir, `${nm}.ref.json`), 'utf8'));
    const bin = readFileSync(join(refDir, `${nm}.ref.bin`));
    const t0 = performance.now();
    const sec = dec.decode(buf);
    const ms = performance.now() - t0;
    assert.equal(sec.n, ref.n, `${nm}: n`);
    assert.equal(sec.nCont, ref.n_cont);
    assert.equal(sec.key, ref.key);
    assert.equal(sec.g0, ref.g0);
    assert.equal(sec.nframes, ref.nframes);
    const mine = arraysOf(sec, dec.params!);
    let off = 0;
    for (const a of ref.arrays) {
      const got = mine[a.name];
      assert.ok(got, `missing ${a.name}`);
      assert.equal(got.length, a.len, `${nm}.${a.name} length`);
      if (i32crc(got) !== a.crc32) {
        const want = new Int32Array(bin.buffer.slice(bin.byteOffset + off * 4, bin.byteOffset + (off + a.len) * 4));
        const i = got.findIndex((v, j) => v !== want[j]);
        assert.fail(`${nm}.${a.name} differs at ${i}: ${got[i]} != ${want[i]}`);
      }
      off += a.len;
    }
    // curve oracle (numpy, float64) vs CPU mirror of the shader (float32 knots)
    const ps = packSection(sec, dec.params!);
    let maxErr = 0;
    for (const s of ref.curve.splats) {
      ref.curve.times.forEach((t: number, ti: number) => {
        const { vals } = evalSplat(ps, s.i, t);
        ['P', 'R', 'S', 'O', 'D'].forEach((tag, g) => {
          const want: number[] = s.vals[tag][ti];
          want.forEach((w, c) => {
            const e = Math.abs(vals[g][c] - w) / Math.max(1, Math.abs(w));
            maxErr = Math.max(maxErr, e);
          });
        });
      });
      const sn8 = new Uint8Array(ps.sn.buffer, s.i * 48, 45);
      s.sn.forEach((w: number, c: number) => {
        const v = ps.snOffset[c] + sn8[c] * ps.snScale[c];
        maxErr = Math.max(maxErr, Math.abs(v - w) / Math.max(1, Math.abs(w)));
      });
    }
    assert.ok(maxErr < tol, `${nm}: curve eval max rel err ${maxErr}`);
    console.log(`# ${nm}: n=${sec.n} decode ${ms.toFixed(1)} ms (python ${ref.python_decode_s} s), curve err ${maxErr.toExponential(1)}`);
  }
}

test('synthetic stream, shared knot, key/delta/key, short last section', () => {
  checkStream(FIX, FIX, ['synth_shared_s0000', 'synth_shared_s0001', 'synth_shared_s0002']);
});

test('synthetic stream, residual first knot', () => {
  checkStream(FIX, FIX, ['synth_res_s0000', 'synth_res_s0001']);
});

test('random access: key section alone works, delta alone is refused', () => {
  const read = (nm: string) => new Uint8Array(readFileSync(join(FIX, `${nm}.csgs`)));
  assert.throws(() => new Decoder().decode(read('synth_shared_s0001')), /seek to a key section/);
  const sec = new Decoder().decode(read('synth_shared_s0002'));
  assert.equal(sec.key, true);
});

test('v2 SN_VQ synthetic stream (key/delta/key, residual VQ, fresh codebooks per key) matches the Python decoder', () => {
  checkStream(FIX, FIX, ['synth_vq_s0000', 'synth_vq_s0001', 'synth_vq_s0002']);
  const read = (nm: string) => new Uint8Array(readFileSync(join(FIX, `${nm}.csgs`)));
  const dec = new Decoder();
  const s0 = dec.decode(read('synth_vq_s0000'));
  assert.equal(s0.key, true);
  assert.ok(s0.snCb && s0.snCb.length === 2 && s0.snCb[1].length === 2, 'key section carries the codebooks (2 levels, level 2 with 2 stages)');
  assert.ok(dec.params!.snVq && dec.params!.snZero, 'params carry the v2 block');
  const s1 = dec.decode(read('synth_vq_s0001'));
  assert.equal(s1.snCb, null, 'delta section has no codebooks of its own');
  assert.ok(s1.snDeg!.some((d) => d === 3) && s1.snDeg!.some((d) => d === 0), 'degrees 0 and 3 both occur');
  // v1 sections must not be mistaken for v2: version byte 1 with the SN_VQ flag set is refused
  const bad = read('synth_res_s0000'); bad[5] |= 4;
  assert.throws(() => new Decoder().decode(bad), /version and flags/);
  assert.throws(() => new Decoder().decode(read('synth_vq_s0001')), /seek to a key section/);
});

test('version/flag consistency and unknown flags / versions are refused with a clear error', () => {
  const read = (nm: string) => new Uint8Array(readFileSync(join(FIX, `${nm}.csgs`)));
  // flags that need a newer version than the version byte says
  for (const [nm, flag] of [['synth_res_s0000', 8], ['synth_res_s0000', 16], ['synth_vq_s0000', 8], ['synth_pred_s0000', 16],
    ['synth_res_s0000', 64], ['synth_pred_s0000', 64]] as const) {
    const bad = read(nm); bad[5] |= flag;
    assert.throws(() => new Decoder().decode(bad), /version and flags/, `${nm} + flag ${flag}`);
  }
  const unk = read('synth_res_s0000'); unk[5] |= 128;                 // 64 = BSPLINE (v6) since the P16 merge
  assert.throws(() => new Decoder().decode(unk), /unsupported \.csgs flags/);
  const v7 = read('synth_vq_s0000'); v7[4] = 7;
  assert.throws(() => new Decoder().decode(v7), /unsupported \.csgs version 7/);
  const v6 = read('synth_vq_s0000'); v6[4] = 6;                       // version 6 without the BSPLINE flag
  assert.throws(() => new Decoder().decode(v6), /version and flags/);
  const v3 = read('synth_vq_s0000'); v3[4] = 3;                       // SN_VQ flag with version 3 but no pred flag
  assert.throws(() => new Decoder().decode(v3), /version and flags/);
});

// ---- v3 pred / v4 follow: synthetic streams made with the Python encoder, all mode combinations
test('v3 pred, all modes (temporal / spatial k-NN / vel0), modes change per section, matches the Python decoder', () => {
  checkStream(FIX, FIX, ['synth_pred_s0000', 'synth_pred_s0001', 'synth_pred_s0002', 'synth_pred_s0003']);
  const d = new Decoder();
  const rd = (nm: string) => new Uint8Array(readFileSync(join(FIX, `${nm}.csgs`)));
  d.decode(rd('synth_pred_s0000'));
  assert.deepEqual(Array.from(d.params!.pred!), [7, 3, 6, 0, 5]);
  assert.equal(d.params!.knnK, 4);
  d.decode(rd('synth_pred_s0001'));
  assert.deepEqual(Array.from(d.params!.pred!), [0, 1, 7, 2, 4]);
});

test('v3 pred with shared knot 0', () => {
  checkStream(FIX, FIX, ['synth_pred_shared_s0000', 'synth_pred_shared_s0001', 'synth_pred_shared_s0002']);
});

test('v3 pred + v2 SN_VQ combined (flags 8|4, version 3)', () => {
  checkStream(FIX, FIX, ['synth_pred_vq_s0000', 'synth_pred_vq_s0001', 'synth_pred_vq_s0002']);
  const buf = new Uint8Array(readFileSync(join(FIX, 'synth_pred_vq_s0001.csgs')));
  assert.equal(buf[4], 3); assert.equal(buf[5] & 12, 12);
});

test('v4 follow alone (flag 16, version 4)', () => {
  checkStream(FIX, FIX, ['synth_follow_s0000', 'synth_follow_s0001', 'synth_follow_s0002']);
  const dec = new Decoder();
  const sec = dec.decode(new Uint8Array(readFileSync(join(FIX, 'synth_follow_s0000.csgs'))));
  assert.ok(sec.follow && sec.follow.some((m) => m !== 0), 'the section carries follow masks');
});

test('v4 follow + v3 pred + v2 SN_VQ (flags 16|8|4) and follow + pred + shared knot', () => {
  checkStream(FIX, FIX, ['synth_follow_all_s0000', 'synth_follow_all_s0001', 'synth_follow_all_s0002', 'synth_follow_all_s0003']);
  checkStream(FIX, FIX, ['synth_follow_shared_s0000', 'synth_follow_shared_s0001', 'synth_follow_shared_s0002']);
  const buf = new Uint8Array(readFileSync(join(FIX, 'synth_follow_all_s0001.csgs')));
  assert.equal(buf[4], 4); assert.equal(buf[5] & 28, 28);
});

test('v6 BSPLINE combined with SN_VQ + pred + follow + rescale (flags 64|32|16|8|4) matches the Python decoder', () => {
  const nm = ['synth_bsp_all_s0000', 'synth_bsp_all_s0001', 'synth_bsp_all_s0002', 'synth_bsp_all_s0003'];
  checkStream(FIX, FIX, nm);
  const b1 = new Uint8Array(readFileSync(join(FIX, 'synth_bsp_all_s0001.csgs')));
  assert.equal(b1[4], 6); assert.equal(b1[5] & 124, 124);
  const dec = new Decoder();
  dec.decode(new Uint8Array(readFileSync(join(FIX, 'synth_bsp_all_s0000.csgs'))));
  assert.equal(dec.params!.bspline, true);
});

test('causalKnn equals brute force (ties by index, clusters, outliers, duplicates)', () => {
  let seed = 12345;
  const rnd = (n: number) => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed % n; };
  for (const [n, k, spread] of [[300, 4, 40], [500, 1, 6], [400, 8, 100000], [200, 255, 20], [64, 3, 0]] as const) {
    const pos = new Int32Array(n * 3);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) pos[i * 3 + c] = rnd(2 * spread + 1) - spread + (rnd(40) === 0 ? rnd(2_000_000) - 1_000_000 : 0);
    const got = causalKnn(pos, n, k);
    for (let i = 0; i < n; i++) {
      const cand: [number, number][] = [];
      for (let j = 0; j < i; j++) {
        const dx = pos[j * 3] - pos[i * 3], dy = pos[j * 3 + 1] - pos[i * 3 + 1], dz = pos[j * 3 + 2] - pos[i * 3 + 2];
        cand.push([dx * dx + dy * dy + dz * dz, j]);
      }
      cand.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      const want = Array.from({ length: k }, (_, t) => (t < cand.length ? cand[t][1] : -1));
      assert.deepEqual(Array.from(got.subarray(i * k, (i + 1) * k)), want, `n=${n} k=${k} spread=${spread} i=${i}`);
    }
  }
});

test('v5 RESCALE (S7): knot table changes between sections, with pred (vel0, temporal, spatial), follow, SN_VQ', () => {
  const rd = (nm: string) => new Uint8Array(readFileSync(join(FIX, `${nm}.csgs`)));
  checkStream(FIX, FIX, ['synth_rescale_s0000', 'synth_rescale_s0001', 'synth_rescale_s0002', 'synth_rescale_s0003']);
  checkStream(FIX, FIX, ['synth_rescale_all_s0000', 'synth_rescale_all_s0001', 'synth_rescale_all_s0002', 'synth_rescale_all_s0003'], 2e-5);   // integer arrays are exact; only the float32 curve mirror gets a looser bound (random wide tables)
  assert.equal(rd('synth_rescale_s0001')[4], 5); assert.ok(rd('synth_rescale_s0001')[5] & 32);
  assert.equal(rd('synth_rescale_s0003')[5] & 32, 0, 'key sections never carry the flag');
  assert.equal(rd('synth_rescale_all_s0001')[4], 5);
  // flag without a table change / table change without flag / flag on a key section are refused
  const fixCrc = (b: Uint8Array) => new DataView(b.buffer).setUint32(b.length - 4, crc32(b, 0, b.length - 4), true);
  const d = new Decoder(); d.decode(rd('synth_pred_s0000'));
  const noChange = rd('synth_pred_s0001'); noChange[4] = 5; noChange[5] |= 32; fixCrc(noChange);
  assert.throws(() => d.decode(noChange), /RESCALE flag disagree/);
  const k = rd('synth_pred_s0000'); k[4] = 5; k[5] |= 32; fixCrc(k);
  assert.throws(() => new Decoder().decode(k), /RESCALE flag on a key section/);
  const d2 = new Decoder(); d2.decode(rd('synth_rescale_s0000'));
  const noFlag = rd('synth_rescale_s0001'); noFlag[4] = 3; noFlag[5] &= ~32; fixCrc(noFlag);
  assert.throws(() => d2.decode(noFlag), /RESCALE flag disagree/);
});

test('corruption is detected by the CRC', () => {
  const buf = new Uint8Array(readFileSync(join(FIX, 'synth_res_s0000.csgs')));
  buf[buf.length >> 1] ^= 0x40;
  assert.throws(() => new Decoder().decode(buf), /CRC mismatch/);
});

test('real P4 sections (grow4 K=5 P1 curves) match the Python decoder', { skip: !existsSync(join(REAL, 'ref', 's0000.ref.json')) }, () => {
  checkStream(REAL, join(REAL, 'ref'), ['s0000', 's0001']);
});

test('real S1 sections (v2 SN_VQ, ft_d3_cb256_i6000_lr0.3) match the Python decoder', { skip: !existsSync(join(REAL_S1, 'ref', 's0000.ref.json')) }, () => {
  checkStream(REAL_S1, join(REAL_S1, 'ref'), ['s0000', 's0001']);
});

test('real the demo capture S1+S2 sections (v3 pred auto + SN_VQ, Morton, 15.98 Mbit/s) match the Python decoder', { skip: !existsSync(join(REAL_S6, 'ref', 's0000.ref.json')) }, () => {
  checkStream(REAL_S6, join(REAL_S6, 'ref'), ['s0000', 's0001']);
  const b = new Uint8Array(readFileSync(join(REAL_S6, 's0001.csgs')));
  assert.equal(b[4], 3);
});

// every v3 mode on every group (temporal | spatial k-NN, knn_k 4 | vel0) on the real 100K-splat sections (make_s6_spatial.sh)
const REAL_S6_ALL = process.env.CSGS_S6_ALL_DIR ?? '';
test('real the demo capture sections with all pred modes (spatial k-NN + vel0 on 100K splats) match the Python decoder', { skip: !existsSync(join(REAL_S6_ALL, 'ref', 's0000.ref.json')) }, () => {
  checkStream(REAL_S6_ALL, join(REAL_S6_ALL, 'ref'), ['s0000', 's0001']);
  const d = new Decoder();
  d.decode(new Uint8Array(readFileSync(join(REAL_S6_ALL, 's0000.csgs'))));
  assert.deepEqual(Array.from(d.params!.pred!), [7, 7, 7, 7, 7]);
  assert.equal(d.params!.knnK, 4);
});
