// .splq: one quantised splat frame (P8-plyq values without the rANS layer), GPU-ready, 16 bytes per splat.
// Written by web/bench/make_viewer_data.py (splq). Little-endian layout:
//   header 256 B: 'SPLQ', u32 version (1), u32 n, u32 flags (bit 0: similarity), f32 off[14], f32 sc[14]
//                 (P3 R4 S3 O1 D3, value = off + q * sc), f32 A[12] (3x4 rows, s*R | t), f32 qR[4] (wxyz), f32 logS
//   n x 4 u32:    w0 = Px | Py<<10 | Pz<<20 (10 bit each), w1 = R0..R3 (int8), w2 = S0 S1 S2 O (u8), w3 = D0 D1 D2 (u8)
// Under the similarity (MLSharp registration, p8_lib.apply_similarity): p' = A p, q' = qR * normalize(q), s' = s + logS.
// optional blocks (flags), in this order after the splat words:
//   bit 2 (P16): n x 2 u32 = Px | Py<<16, Pz (16 bit each); then P = off + q16 * sc and w0 is unused (large N3DV extents)
//   bit 1 (SN):  u32 m (SH scalars present, <= 45), f32 off[45], f32 sc[45], n x ceil4(m) u8 = higher SH bands
//                (coefficient-major like pack.ts SN; missing bands are 0) -> rendered with the bands instead of SH0 only
export const SPLQ_HDR = 256;
export const SPLQ_FLAG_SIM = 1;
export const SPLQ_FLAG_SN = 2;
export const SPLQ_FLAG_P16 = 4;

export interface SplqFrame {
  n: number; flags: number; header: Uint8Array; data: Uint32Array;
  run: Uint32Array | null;                       // n * 2 (flag P16)
  snM: number; snq: Float32Array | null; sn: Uint8Array | null; snStride: number;   // flag SN
}

export function parseSplq(buf: ArrayBuffer): SplqFrame {
  const u8 = new Uint8Array(buf);
  if (u8.length < SPLQ_HDR || u8[0] !== 0x53 || u8[1] !== 0x50 || u8[2] !== 0x4c || u8[3] !== 0x51) throw new Error('not a .splq frame');
  const dv = new DataView(buf);
  const ver = dv.getUint32(4, true);
  if (ver !== 1) throw new Error(`unsupported .splq version ${ver}`);
  const n = dv.getUint32(8, true), flags = dv.getUint32(12, true);
  let o = SPLQ_HDR + n * 16;
  let run: Uint32Array | null = null, snq: Float32Array | null = null, sn: Uint8Array | null = null, snM = 0, snStride = 0;
  if (flags & SPLQ_FLAG_P16) {
    if (u8.length < o + n * 8) throw new Error(`.splq size ${u8.length} < ${o + n * 8} (P16)`);
    run = new Uint32Array(buf, o, n * 2); o += n * 8;
  }
  if (flags & SPLQ_FLAG_SN) {
    if (u8.length < o + 364) throw new Error(`.splq size ${u8.length} < ${o + 364} (SN)`);
    snM = dv.getUint32(o, true);
    if (snM < 1 || snM > 45) throw new Error(`.splq SN count ${snM}`);
    snq = new Float32Array(buf.slice(o + 4, o + 364)); o += 364;
    snStride = (snM + 3) & ~3;
    if (u8.length < o + n * snStride) throw new Error(`.splq size ${u8.length} < ${o + n * snStride} (SN)`);
    sn = u8.subarray(o, o + n * snStride); o += n * snStride;
  }
  if (u8.length !== o) throw new Error(`.splq size ${u8.length} != ${o}`);
  return { n, flags, header: u8.subarray(0, SPLQ_HDR), data: new Uint32Array(buf, SPLQ_HDR, n * 4), run, snM, snq, sn, snStride };
}

/** SN bands in the renderer's layout (pack.ts: 12 u32 = 48 B per splat, scalar k at byte k). */
export function splqSnWords(fr: SplqFrame): Uint32Array | null {
  if (!fr.sn) return null;
  const out = new Uint8Array(fr.n * 48), m = fr.snM, st = fr.snStride, src = fr.sn;
  for (let i = 0; i < fr.n; i++) out.set(src.subarray(i * st, i * st + m), i * 48);
  return new Uint32Array(out.buffer);
}

/** dequantised SN scalar k (0..44) of splat i (CPU mirror of PROJECT_WGSL shn). */
export function splqSn(fr: SplqFrame, i: number, k: number): number {
  if (!fr.sn || !fr.snq) return 0;
  const b = k < fr.snM ? fr.sn[i * fr.snStride + k] : 0;
  return Math.fround(fr.snq[k] + Math.fround(b * fr.snq[45 + k]));
}

function qmul(a: number[], b: number[]): number[] {
  const [aw, ax, ay, az] = a, [bw, bx, by, bz] = b;
  return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw];
}

/** CPU mirror of EVALQ_WGSL: world attributes of splat i -> [P(3), R(4, unit), S(3), O(1), D(3)]. */
export function evalSplq(fr: SplqFrame, i: number): number[][] {
  const h = new Float32Array(fr.header.buffer, fr.header.byteOffset, 64);
  const off = (k: number) => h[4 + k], sc = (k: number) => h[18 + k];
  const w0 = fr.data[i * 4], w1 = fr.data[i * 4 + 1], w2 = fr.data[i * 4 + 2], w3 = fr.data[i * 4 + 3];
  const f = Math.fround;
  let p = [w0 & 1023, (w0 >>> 10) & 1023, (w0 >>> 20) & 1023].map((q, c) => f(off(c) + f(q * sc(c))));
  if (fr.run) {
    const a = fr.run[i * 2], b = fr.run[i * 2 + 1];
    p = [a & 65535, a >>> 16, b & 65535].map((q, c) => f(off(c) + f(q * sc(c))));
  }
  const r8 = [0, 8, 16, 24].map((s) => ((w1 >>> s) & 255) << 24 >> 24);
  let q = r8.map((v, c) => f(off(3 + c) + f(v * sc(3 + c))));
  const nq = Math.hypot(...q) || 1;
  q = q.map((v) => v / nq);
  let s = [0, 8, 16].map((sh, c) => f(off(7 + c) + f(((w2 >>> sh) & 255) * sc(7 + c))));
  const o = f(off(10) + f((w2 >>> 24) * sc(10)));
  const d = [0, 8, 16].map((sh, c) => f(off(11 + c) + f(((w3 >>> sh) & 255) * sc(11 + c))));
  if (fr.flags & SPLQ_FLAG_SIM) {
    const A = (r: number, c: number) => h[32 + r * 4 + c];
    p = [0, 1, 2].map((r) => A(r, 0) * p[0] + A(r, 1) * p[1] + A(r, 2) * p[2] + A(r, 3));
    q = qmul([h[44], h[45], h[46], h[47]], q);
    s = s.map((v) => v + h[48]);
  }
  return [p, q, s, [o], d];
}
