// the renderer header packs K per curve group in 6 bits (mesh streams use K up to 33 knots per
// 33-frame section; the old 4-bit packing rejected them with "K<=15").
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Params, Section } from '../src/falt.ts';
import { packSection, evalSplat, GROUP_DIMS } from '../src/pack.ts';

test('pack/eval with K = 33 (hat basis, one knot per frame)', () => {
  const K = 33, n = 2, G = 5;
  const dims = GROUP_DIMS;
  const ks = [1, 2, 3, 5, 9, 17, 33];
  const nq = dims.reduce((a, b) => a + b, 0) + 45;
  const p: Params = { L: 33, ks, groups: [['P', 3], ['R', 4], ['S', 3], ['O', 1], ['D', 3]], nSn: 45,
    offsets: new Float32Array(nq), scales: new Float32Array(nq).fill(1), sharedKnot: false, keyInterval: 4,
    snVq: null, snZero: null, pred: null, knnK: 0, bspline: false };
  const kPer = [K, 17, 5, 1, K];                         // mixed K per group, splat 1 uses them, splat 0 all K=33
  const k: Uint8Array[] = [], knotOff: Uint32Array[] = [], knots: Int32Array[] = [];
  for (let g = 0; g < G; g++) {
    const kk = new Uint8Array([K, kPer[g]]);
    const off = new Uint32Array(n + 1);
    for (let i = 0; i < n; i++) off[i + 1] = off[i] + kk[i] * dims[g];
    const v = new Int32Array(off[n]);
    for (let i = 0; i < n; i++) for (let j = 0; j < kk[i]; j++) for (let c = 0; c < dims[g]; c++) v[off[i] + j * dims[g] + c] = 100 * i + 10 * j + c;
    k.push(kk); knotOff.push(off); knots.push(v);
  }
  const sec: Section = { index: 0, g0: 0, nframes: 33, key: true, n, nCont: 0, ids: new Uint32Array([0, 1]),
    birth: new Uint16Array([0, 0]), death: new Uint16Array([34, 34]), k, knotOff, knots, sn: new Uint8Array(n * 45),
    snDeg: null, snIdx: null, snCb: null, follow: null };
  const ps = packSection(sec, p);
  for (const t of [0, 7, 20, 32]) {
    const a = evalSplat(ps, 0, t).vals;                  // K = 33 over 33 frames: knot j = frame j
    for (let g = 0; g < G; g++) if (g !== 1) for (let c = 0; c < dims[g]; c++) assert.equal(a[g][c], 10 * t + c);  // R is nlerp-normalised
    const b = evalSplat(ps, 1, t).vals;
    assert.equal(b[3][0], 100);                            // K = 1: constant
    assert.equal(b[0][0], 100 + 10 * t);                   // K = 33 in group P of splat 1
  }
});
