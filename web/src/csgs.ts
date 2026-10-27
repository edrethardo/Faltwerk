// .csgs section decoder — 1:1 port of codec/csgs.py (decode side), layout in docs/format-csgs.md.
// Output is struct-of-arrays (typed arrays) instead of Python lists; values are identical.
// v2 (flag SN_VQ, docs §7, ): SH codebooks in the key section, per splat an SH degree + one codebook index per
// residual stage; `sn` is still the reconstructed 0..255 row, so everything downstream (pack.ts, shaders) is unchanged.
// v3 (flag pred, docs §8, ): per-group knot prediction modes (temporal / spatial k-NN / vel0), v4 (flag FOLLOW, §9):
// masked knots of group P are predicted from the nearest unmasked splats (apply_follow). Flags combine freely, the
// version is the highest one a set flag needs. All of it is integer arithmetic (doubles, exact below 2^53) like Python.
import { AdaptiveModel, IDX_INC, NTOK, RansDecoder } from './rans.ts';
import { causalKnn } from './knn.ts';

export const VERSION = 1;
export const VERSION_SN_VQ = 2;
export const VERSION_PRED = 3;
export const VERSION_FOLLOW = 4;
export const VERSION_RESCALE = 5;
export const VERSION_BSPLINE = 6;                  // flag 64, knots are control points of a clamped cubic B-spline
const FLAG_KEY = 1, FLAG_SHARED_KNOT = 2, FLAG_SN_VQ = 4, FLAG_PRED = 8, FLAG_FOLLOW = 16, FLAG_RESCALE = 32, FLAG_BSPLINE = 64;
export const PRED_TEMPORAL = 1, PRED_SPATIAL = 2, PRED_VEL0 = 4;
export const FOLLOW_K = 4, FOLLOW_CELL = 8;
const ST_K = 3;
const SN_CH = 3;
const MAX_CB = 16384;

export interface VqLevel { deg: number; sizes: number[] }

export interface Params {
  L: number;
  ks: number[];
  groups: [string, number][];   // tag, dims
  nSn: number;
  offsets: Float32Array;        // nq = sum(dims) + nSn, groups in order then SN
  scales: Float32Array;
  sharedKnot: boolean;
  keyInterval: number;
  snVq: VqLevel[] | null;       // v2: levels (degree + codebook size per residual stage), null = v1
  snZero: Uint8Array | null;    // v2: grid point of 0.0 per SN scalar (truncated bands)
  pred: Uint8Array | null;      // v3: mode bits per group (PRED_*), null = v1/v2 knot coding
  knnK: number;                 // v3: neighbours of the spatial mode
  bspline: boolean;             // v6 (flag 64): curve basis = clamped cubic B-spline (src/bspline.ts), else hat/linear
}

export interface Header {
  follow: boolean; rescale: boolean;
  key: boolean; index: number; g0: number; nframes: number; nCont: number; nNew: number; idBase: number;
  snPred: Uint8Array;
}

/** v2 codebooks: [level][stage] flat entries (size * snDim(deg) ints; stage 0 in 0..255, residual stages -255..255) */
export type Codebooks = Int16Array[][];

export interface Section {
  index: number; g0: number; nframes: number; key: boolean;
  n: number; nCont: number;
  ids: Uint32Array;            // n
  birth: Uint16Array;          // nNew (birth frame - g0)
  death: Uint16Array;          // n (death code, nframes+1 = continues)
  k: Uint8Array[];             // [G][n] K per splat
  knotOff: Uint32Array[];      // [G][n+1] start of splat i in knots[g] (in ints)
  knots: Int32Array[];         // [G] flat, knot-major per splat (K*dims ints)
  sn: Uint8Array;              // n * nSn (v2: the VQ reconstruction)
  snDeg: Uint8Array | null;    // v2: n, SH degree per splat (0 or one of the level degrees)
  snIdx: Uint16Array | null;   // v2: n * maxStages, codebook index per stage of the splat's level (rest 0)
  snCb: Codebooks | null;      // v2 key sections only
  follow: Uint32Array | null;  // v4: per splat bit mask over the knots of group P (bit j = knot j predicted); null = none
}

export interface State {
  ids: Uint32Array; last: Int32Array[]; sn: Uint8Array; k0: Uint8Array; nextIndex: number;
  vel: Int32Array[];                                                       // v3 vel0: per group n*d end velocity (empty otherwise)
  params: Params | null;                                                   // quantisation table the integers live on (v5 RESCALE)
  snLevel: Uint8Array; snIdx: Uint16Array; snCb: Codebooks | null;      // v2 (empty / null in v1)
}

export function snDim(deg: number): number { return SN_CH * ((deg + 1) ** 2 - 1); }
export function maxStages(p: Params): number { return p.snVq ? Math.max(1, ...p.snVq.map((l) => l.sizes.length)) : 0; }

// ---------------------------------------------------------------- CRC32 (zlib/PNG)
const CRC_TABLES = (() => {                       // slicing-by-4
  const t = new Uint32Array(4 * 256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  for (let n = 0; n < 256; n++) for (let k = 1; k < 4; k++) t[k * 256 + n] = (t[(k - 1) * 256 + n] >>> 8) ^ t[t[(k - 1) * 256 + n] & 255];
  return t;
})();

export function crc32(b: Uint8Array, start = 0, end = b.length): number {
  const T = CRC_TABLES;
  let c = 0xffffffff, i = start;
  for (; i + 4 <= end; i += 4) {
    c ^= b[i] | (b[i + 1] << 8) | (b[i + 2] << 16) | (b[i + 3] << 24);
    c = T[768 + (c & 255)] ^ T[512 + ((c >>> 8) & 255)] ^ T[256 + ((c >>> 16) & 255)] ^ T[c >>> 24];
  }
  for (; i < end; i++) c = T[(c ^ b[i]) & 255] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// ---------------------------------------------------------------- header
export function unpackHeader(buf: Uint8Array): { params: Params; h: Header; lens: number[]; o: number } {
  if (buf.length < 4 || buf[0] !== 0x43 || buf[1] !== 0x53 || buf[2] !== 0x47 || buf[3] !== 0x53)
    throw new Error('not a .csgs section');
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let o = 4;
  const ver = buf[o++];
  if (ver < VERSION || ver > VERSION_BSPLINE) throw new Error(`unsupported .csgs version ${ver}`);
  const flags = buf[o++];
  if (flags & ~(FLAG_KEY | FLAG_SHARED_KNOT | FLAG_SN_VQ | FLAG_PRED | FLAG_FOLLOW | FLAG_RESCALE | FLAG_BSPLINE))
    throw new Error(`unsupported .csgs flags 0x${flags.toString(16)}`);
  const vq = !!(flags & FLAG_SN_VQ), predF = !!(flags & FLAG_PRED), followF = !!(flags & FLAG_FOLLOW), rescaleF = !!(flags & FLAG_RESCALE);
  const bspline = !!(flags & FLAG_BSPLINE);
  if (ver !== (bspline ? VERSION_BSPLINE : rescaleF ? VERSION_RESCALE : followF ? VERSION_FOLLOW : predF ? VERSION_PRED : vq ? VERSION_SN_VQ : VERSION))
    throw new Error('version and flags (SN_VQ / pred / follow / rescale / bspline) disagree');
  const index = dv.getUint32(o, true); o += 4;
  const g0 = dv.getUint32(o, true); o += 4;
  const L = dv.getUint16(o, true); o += 2;
  const nframes = dv.getUint16(o, true); o += 2;
  const nCont = dv.getUint32(o, true); o += 4;
  const nNew = dv.getUint32(o, true); o += 4;
  const idBase = dv.getUint32(o, true); o += 4;
  const keyInterval = buf[o++];
  const G = buf[o++];
  const groups: [string, number][] = [];
  for (let g = 0; g < G; g++) { groups.push([String.fromCharCode(buf[o]), buf[o + 1]]); o += 2; }
  const nk = buf[o++];
  const ks = Array.from(buf.subarray(o, o + nk)); o += nk;
  let pred: Uint8Array | null = null, knnK = 0;
  if (predF) {                                    // v3 block: knn_k and one mode byte per group
    knnK = buf[o++];
    pred = buf.slice(o, o + G); o += G;
    if (pred.some((mo) => mo > 7)) throw new Error('pred needs one mode 0..7 per group');
    const spatial = pred.map((mo) => mo & PRED_SPATIAL);
    if (spatial.some((x) => x)) {
      const gp = groups.findIndex(([t, d]) => t === 'P' && d === 3);
      if (knnK < 1) throw new Error('knn_k 1..255 for spatial prediction');
      if (gp < 0) throw new Error("spatial prediction needs a group 'P' with 3 dims");
      if (spatial.slice(0, gp).some((x) => x)) throw new Error("group 'P' must be coded before any spatially predicted group");
    } else if (knnK) throw new Error('knn_k only with spatial prediction');
  }
  const nSn = dv.getUint16(o, true); o += 2;
  const snPred = buf.slice(o, o + nSn); o += nSn;
  let snVq: VqLevel[] | null = null, snZero: Uint8Array | null = null;
  if (vq) {                                       // v2 block: zero grid point, levels (degree + stage sizes)
    snZero = buf.slice(o, o + nSn); o += nSn;
    const nl = buf[o++];
    snVq = [];
    for (let l = 0; l < nl; l++) {
      const deg = buf[o++], ns = buf[o++];
      const sizes: number[] = [];
      for (let s = 0; s < ns; s++) { sizes.push(dv.getUint16(o, true)); o += 2; }
      snVq.push({ deg, sizes });
    }
    const degs = snVq.map((l) => l.deg);
    if (!degs.length || degs.some((d, i) => d < 1 || d > 3 || (i > 0 && d <= degs[i - 1])) || snDim(degs[degs.length - 1]) > nSn)
      throw new Error('sn_vq levels: strictly increasing degrees in 1..3 with sn_dim(deg) <= n_sn');
    if (snVq.some((l) => !l.sizes.length || l.sizes.some((s) => s < 1 || s > MAX_CB)))
      throw new Error(`sn_vq stages: 1..255 stages of 1..${MAX_CB} entries`);
  }
  const nq = groups.reduce((a, [, d]) => a + d, 0) + nSn;
  const offsets = new Float32Array(nq), scales = new Float32Array(nq);
  for (let i = 0; i < nq; i++) { offsets[i] = dv.getFloat32(o, true); o += 4; }
  for (let i = 0; i < nq; i++) { scales[i] = dv.getFloat32(o, true); o += 4; }
  const ns = buf[o++];
  const lens: number[] = [];
  for (let i = 0; i < ns; i++) { lens.push(dv.getUint32(o, true)); o += 4; }
  if (!(L >= 1 && L <= 253) || !ks.length) throw new Error('bad L or ks');
  const params: Params = { L, ks, groups, nSn, offsets, scales, sharedKnot: !!(flags & FLAG_SHARED_KNOT), keyInterval, snVq, snZero, pred, knnK, bspline };
  const h: Header = { key: !!(flags & FLAG_KEY), follow: followF, rescale: rescaleF, index, g0, nframes, nCont, nNew, idBase, snPred };
  return { params, h, lens, o };
}

// ---------------------------------------------------------------- models
class Models {
  birth: AdaptiveModel[]; death: AdaptiveModel[]; ids: AdaptiveModel; k: AdaptiveModel[][];
  first: AdaptiveModel[][][]; diff: AdaptiveModel[][][][]; sn: AdaptiveModel[][];
  cbRes: AdaptiveModel[][] = []; deg: AdaptiveModel[] = []; idx: AdaptiveModel[][] = [];
  follow: AdaptiveModel[][];                           // [kidx of group P][bit of the previous knot]
  constructor(p: Params, nframes: number) {
    const nk = p.ks.length;
    this.follow = Array.from({ length: nk }, () => [new AdaptiveModel(2), new AdaptiveModel(2)]);
    this.birth = Array.from({ length: nframes }, () => new AdaptiveModel(nframes));
    this.death = [new AdaptiveModel(p.L + 2), new AdaptiveModel(p.L + 2)];
    this.ids = new AdaptiveModel(NTOK);
    this.k = p.groups.map(() => Array.from({ length: nk + 1 }, () => new AdaptiveModel(nk)));
    this.first = p.groups.map(([, d]) => [0, 1].map(() => Array.from({ length: d }, () => new AdaptiveModel(NTOK))));
    this.diff = p.groups.map(([, d]) => Array.from({ length: nk }, () =>
      Array.from({ length: d }, () => Array.from({ length: 4 }, () => new AdaptiveModel(NTOK)))));
    this.sn = Array.from({ length: p.nSn }, () => Array.from({ length: 6 }, () => new AdaptiveModel(NTOK)));
    if (p.snVq) {
      const nl = p.snVq.length + 1;
      // residual-stage codebook entries: [scalar c][bucket6 of the previous coefficient's entry, same channel]
      this.cbRes = Array.from({ length: p.nSn }, () => Array.from({ length: 6 }, () => new AdaptiveModel(NTOK)));
      this.deg = Array.from({ length: nl }, () => new AdaptiveModel(nl));                 // context: previous splat's level
      this.idx = p.snVq.map((l) => l.sizes.map((s) => new AdaptiveModel(s, IDX_INC)));  // [level-1][stage]
    }
  }
}

function bucket6(v: number): number {
  if (v < 0) v = -v;
  return v <= 2 ? v : v <= 4 ? 3 : v <= 8 ? 4 : 5;
}

function bucket(v: number): number {
  if (v < 0) v = -v;
  return v >= 2 ? 3 : v + 1;
}

/** SN row of one splat into out[o..o+nSn): level 0 -> zero row; else codebook sum over the stages (clipped 0..255) for
 *  the first snDim(deg) scalars, zero row beyond. Integer-only (codec/csgs.py vq_row). idxs[io..] = the splat's indices. */
export function vqRow(out: Uint8Array, o: number, p: Params, level: number, idxs: Uint16Array, io: number, cb: Codebooks): void {
  out.set(p.snZero!, o);
  if (level === 0) return;
  const stages = cb[level - 1];
  const d = snDim(p.snVq![level - 1].deg);
  const base = stages[0], b0 = idxs[io] * d;
  for (let c = 0; c < d; c++) {
    let v = base[b0 + c];
    for (let s = 1; s < stages.length; s++) v += stages[s][idxs[io + s] * d + c];
    out[o + c] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
}

// ---------------------------------------------------------------- motion prediction (v3) helpers
/** knot step of the last knot interval scaled to ONE frame (codec/csgs.py end_velocity); 0 for K = 1 or a 1-frame section. */
export function endVelocity(row: Int32Array, o: number, K: number, d: number, nframes: number, out: Int32Array, po: number): void {
  if (K < 2 || nframes < 2) { for (let c = 0; c < d; c++) out[po + c] = 0; return; }
  const den = nframes - 1;
  for (let c = 0; c < d; c++)
    out[po + c] = Math.floor((2 * (row[o + (K - 1) * d + c] - row[o + (K - 2) * d + c]) * (K - 1) + den) / (2 * den));
}

/** value of dim c of a Kn-knot hat curve at the frame of knot j of a Ki-knot curve (resample_knot): linear, round half up. */
function resampleAt(row: Int32Array, o: number, Kn: number, d: number, j: number, Ki: number, c: number): number {
  if (Kn === 1 || Ki === 1) return row[o + c];
  const num = j * (Kn - 1), den = Ki - 1;
  let q = Math.floor(num / den);
  const rem = num - q * den;
  if (rem === 0 || q >= Kn - 1) return row[o + Math.min(q, Kn - 1) * d + c];
  const a = row[o + q * d + c], b = row[o + (q + 1) * d + c];
  return a + Math.floor((2 * (b - a) * rem + den) / (2 * den));
}

/** predict_diffs: mean over the neighbour curves of their knot differences resampled to a Ki-knot grid ((Ki-1)*d, diff-major). */
function predictDiffs(out: Float64Array, row: Int32Array, off: Uint32Array, kk: Uint8Array, nbs: Int32Array, nb0: number, cnt: number, d: number, Ki: number): void {
  const len = (Ki - 1) * d;
  out.fill(0, 0, len);
  for (let t = 0; t < cnt; t++) {
    const j0 = nbs[nb0 + t], o = off[j0], Kn = kk[j0];
    for (let c = 0; c < d; c++) {
      let prev = resampleAt(row, o, Kn, d, 0, Ki, c);
      for (let j = 1; j < Ki; j++) {
        const cur = resampleAt(row, o, Kn, d, j, Ki, c);
        out[(j - 1) * d + c] += cur - prev;
        prev = cur;
      }
    }
  }
  for (let t = 0; t < len; t++) out[t] = Math.floor((2 * out[t] + cnt) / (2 * cnt));
}

/** knots of one group in `pred` mode (docs §8): phase 1 = knot 0 of every splat, phase 2 = the knot differences. */
function decodeKnotsPred(p: Params, d: RansDecoder, key: boolean, st: State | null, m: Models, g: number, ki: Uint8Array[],
                        k: Uint8Array[], nc: number, n: number, off: Uint32Array, row: Int32Array,
                        gpRow: Int32Array | null, gpOff: Uint32Array | null, knn: { v: Int32Array | null }): void {
  const dm = p.groups[g][1], mode = p.pred![g];
  const firstRes = m.first[g][0], firstAbs = m.first[g][1], mdiff = m.diff[g];
  const prevAbs = new Int32Array(dm);
  const last = st ? st.last[g] : null, vel = st ? st.vel[g] : null;
  const vel0 = (mode & PRED_VEL0) !== 0;
  for (let i = 0; i < n; i++) {                                  // phase 1
    const b = off[i];
    if (!key && i < nc) {
      if (p.sharedKnot) for (let c = 0; c < dm; c++) row[b + c] = last![i * dm + c];
      else if (vel0) for (let c = 0; c < dm; c++) row[b + c] = last![i * dm + c] + vel![i * dm + c] + d.getSint(firstRes[c]);
      else for (let c = 0; c < dm; c++) row[b + c] = last![i * dm + c] + d.getSint(firstRes[c]);
    } else {
      for (let c = 0; c < dm; c++) { row[b + c] = prevAbs[c] + d.getSint(firstAbs[c]); prevAbs[c] = row[b + c]; }
    }
  }
  const kg = k[g];
  let nbs: Int32Array | null = null;
  if (mode & PRED_SPATIAL) {
    if (!knn.v) {                                                // once per section, from knot 0 of group P
      const pr = gpRow ?? row, po = gpOff ?? off;
      const pos = new Int32Array(n * 3);
      for (let i = 0; i < n; i++) { const b = po[i]; pos[i * 3] = pr[b]; pos[i * 3 + 1] = pr[b + 1]; pos[i * 3 + 2] = pr[b + 2]; }
      knn.v = causalKnn(pos, n, p.knnK);
    }
    nbs = knn.v;
  }
  const knnK = p.knnK;
  const temporal = (mode & PRED_TEMPORAL) !== 0;
  let maxK = 1;
  for (let i = 0; i < n; i++) if (kg[i] > maxK) maxK = kg[i];
  const pred = new Float64Array((maxK - 1) * dm);
  for (let i = 0; i < n; i++) {                                  // phase 2
    const K = kg[i], b = off[i];
    let hasPred = false;
    if (nbs && K > 1) {
      let cnt = 0;
      while (cnt < knnK && nbs[i * knnK + cnt] >= 0) cnt++;     // padding (-1) only at the end
      if (cnt) { predictDiffs(pred, row, off, kg, nbs, i * knnK, cnt, dm, K); hasPred = true; }
    }
    const mk = mdiff[ki[g][i]];
    for (let c = 0; c < dm; c++) {
      const mc = mk[c];
      let pr = 0, has = false, prevV = 0;
      for (let j = 1; j < K; j++) {
        const r = d.getSint(mc[has ? bucket(pr) : 0]);
        let v = r;
        if (hasPred) {
          if (temporal && j >= 2) v = r + prevV + (pred[(j - 1) * dm + c] - pred[(j - 2) * dm + c]);
          else v = r + pred[(j - 1) * dm + c];
        } else if (temporal && j >= 2) v = r + prevV;
        row[b + j * dm + c] = row[b + (j - 1) * dm + c] + v;
        pr = r; has = true; prevV = v;
      }
    }
  }
}

// ---------------------------------------------------------------- follow mode (v4, docs §9)
function followDen(p: Params): number {
  const D = Math.max(...p.ks) - 1;
  if (D < 1 || p.ks.some((k) => k > 1 && D % (k - 1))) throw new Error('follow mode needs every K-1 to divide max(ks)-1');
  return D;
}

/** D x value of dim c of a 3-dim knot row (K knots at row[o..]) at grid position u in 0..D (exact integer, codec/csgs.py _at). */
function followAt(row: Int32Array, o: number, K: number, u: number, D: number, c: number): number {
  if (K === 1) return D * row[o + c];
  const s = D / (K - 1);
  const seg = Math.floor(u / s), rem = u - seg * s;
  let v = row[o + seg * 3 + c] * (s - rem);
  if (rem) v += row[o + (seg + 1) * 3 + c] * rem;
  return v * (D / s);
}

/** Replaces every masked knot of group P (in place) by its prediction — codec/csgs.py apply_follow. The neighbour set
 *  (FOLLOW_K nearest unmasked splats by squared distance of knot 0, ties: lower index) is the exact kNN, so the search
 *  details (grid, brute-force fallback) cannot change the result. */
export function applyFollow(p: Params, sec: Section): void {
  const mask = sec.follow;
  if (!mask) return;
  const gP = p.groups.findIndex(([t]) => t === 'P');
  const D = followDen(p);
  const row = sec.knots[gP], off = sec.knotOff[gP], K = sec.k[gP], n = sec.n;
  const cand: number[] = [];
  for (let i = 0; i < n; i++) if (mask[i] === 0) cand.push(i);
  let F = FOLLOW_CELL;                  // the cell size only affects speed, never the result (exact kNN)
  const spanOf = () => {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const i of cand) for (let c = 0; c < 3; c++) { const v = Math.floor(row[off[i] + c] / F); if (v < lo[c]) lo[c] = v; if (v > hi[c]) hi[c] = v; }
    return { lo, hi };
  };
  let { lo, hi } = spanOf();
  while (cand.length && (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1) >= 2 ** 52) { F *= 2; ({ lo, hi } = spanOf()); }
  const cellOf = (v: number) => Math.floor(v / F);
  const ny = cand.length ? hi[1] - lo[1] + 1 : 1, nz = cand.length ? hi[2] - lo[2] + 1 : 1, nx = cand.length ? hi[0] - lo[0] + 1 : 1;
  const cells = new Map<number, number[]>();
  for (const i of cand) {
    const key = ((cellOf(row[off[i]]) - lo[0]) * ny + (cellOf(row[off[i] + 1]) - lo[1])) * nz + (cellOf(row[off[i] + 2]) - lo[2]);
    const l = cells.get(key);
    if (l) l.push(i); else cells.set(key, [i]);
  }
  const d2 = (i: number, q0: number, q1: number, q2: number) => {
    const o = off[i], a = row[o] - q0, b = row[o + 1] - q1, c = row[o + 2] - q2;
    return a * a + b * b + c * c;
  };
  const byDist = (a: [number, number], b: [number, number]) => a[0] - b[0] || a[1] - b[1];
  const neighbours = (q0: number, q1: number, q2: number, span: number): number[] => {
    const cq0 = cellOf(q0), cq1 = cellOf(q1), cq2 = cellOf(q2);
    let found: [number, number][] = [];
    for (let r = 0; r <= span; r++) {
      if ((2 * r + 1) ** 3 - (2 * r - 1) ** 3 > cells.size) {   // sparse: exact brute force is cheaper
        found = cand.map((i) => [d2(i, q0, q1, q2), i] as [number, number]).sort(byDist);
        return found.slice(0, FOLLOW_K).map((x) => x[1]);
      }
      for (let dx = -r; dx <= r; dx++) {
        const ax = cq0 + dx - lo[0];
        if (ax < 0 || ax >= nx) continue;
        for (let dy = -r; dy <= r; dy++) {
          const ay = cq1 + dy - lo[1];
          if (ay < 0 || ay >= ny) continue;
          const edgeXY = Math.abs(dx) === r || Math.abs(dy) === r;
          const step = edgeXY ? 1 : 2 * r;
          for (let dz = -r; dz <= r; dz += step) {
            const az = cq2 + dz - lo[2];
            if (az < 0 || az >= nz) continue;
            const l = cells.get((ax * ny + ay) * nz + az);
            if (l) for (const i of l) found.push([d2(i, q0, q1, q2), i]);
          }
        }
      }
      if (found.length >= FOLLOW_K) {
        found.sort(byDist);
        if (found[FOLLOW_K - 1][0] <= (r * F) * (r * F)) return found.slice(0, FOLLOW_K).map((x) => x[1]);
      }
    }
    found.sort(byDist);
    return found.slice(0, FOLLOW_K).map((x) => x[1]);
  };
  for (let i = 0; i < n; i++) {
    const m = mask[i];
    if (!m) continue;
    const Ki = K[i], o = off[i];
    if ((m & 1) || Ki < 2 || Ki > 31 || (m >>> Ki)) throw new Error('follow mask: knot 0 or a knot beyond K masked');
    let nb: number[] = [];
    if (cand.length) {
      const qx = cellOf(row[o]), qy = cellOf(row[o + 1]), qz = cellOf(row[o + 2]);
      const span = Math.max(Math.abs(qx - lo[0]), Math.abs(qx - hi[0]), Math.abs(qy - lo[1]), Math.abs(qy - hi[1]), Math.abs(qz - lo[2]), Math.abs(qz - hi[2]));
      nb = neighbours(row[o], row[o + 1], row[o + 2], span);
    }
    const st = D / (Ki - 1);
    let a = 0;
    for (let j = 1; j < Ki; j++) {
      if (!((m >>> j) & 1)) { a = j; continue; }
      for (let c = 0; c < 3; c++) {
        if (nb.length) {
          let S = 0;
          for (const nn of nb) S += followAt(row, off[nn], K[nn], j * st, D, c) - followAt(row, off[nn], K[nn], a * st, D, c);
          const q = D * nb.length;
          row[o + j * 3 + c] = row[o + a * 3 + c] + Math.floor((S + Math.floor(q / 2)) / q);
        } else row[o + j * 3 + c] = row[o + a * 3 + c];
      }
    }
  }
}

// ---------------------------------------------------------------- quantisation change between sections (v5, docs §10)
const sameTable = (a: Float32Array, b: Float32Array, from: number, to: number) => {
  for (let i = from; i < to; i++) if (a[i] !== b[i]) return false;
  return true;
};

/** copy of the state with `last` and the end velocities moved to the new section's table (codec/csgs.py rescale_state):
 *  last := floor((off_o + q*sc_o - off_n)/sc_n + 0.5), vel := floor(v*sc_o/sc_n + 0.5), IEEE doubles in this order. */
export function rescaleState(st: State, pOld: Params, pNew: Params): State {
  const out: State = { ...st, params: pNew, last: [], vel: [] };
  let q0 = 0;
  pNew.groups.forEach(([, d], g) => {
    const last = new Int32Array(st.last[g].length), vel = new Int32Array(st.vel[g].length);
    const m = last.length / d;
    for (let i = 0; i < m; i++) {
      for (let c = 0; c < d; c++) {
        const oo = pOld.offsets[q0 + c], so = pOld.scales[q0 + c], on = pNew.offsets[q0 + c], sn = pNew.scales[q0 + c];
        last[i * d + c] = Math.floor((oo + st.last[g][i * d + c] * so - on) / sn + 0.5);
        vel[i * d + c] = Math.floor(st.vel[g][i * d + c] * so / sn + 0.5);
      }
    }
    q0 += d;
    out.last.push(last); out.vel.push(vel);
  });
  return out;
}

// ---------------------------------------------------------------- decoder
/** prev: the state before `sec` (carries the key section's codebooks through the delta sections). */
export function stateAfter(sec: Section, p: Params, prev: State | null = null): State {
  const cont = sec.nframes + 1;
  let m = 0;
  for (let i = 0; i < sec.n; i++) if (sec.death[i] === cont) m++;
  const kidx = new Map(p.ks.map((k, i) => [k, i]));
  const vq = !!p.snVq, ms = maxStages(p);
  const degs = vq ? p.snVq!.map((l) => l.deg) : [];
  const needVel = true;                  // modes may change from section to section (pred auto): always keep the end velocities
  const st: State = {
    ids: new Uint32Array(m), sn: new Uint8Array(m * p.nSn), k0: new Uint8Array(m),
    last: p.groups.map(([, d]) => new Int32Array(m * d)), nextIndex: sec.index + 1,
    vel: needVel ? p.groups.map(([, d]) => new Int32Array(m * d)) : [], params: p,
    snLevel: new Uint8Array(vq ? m : 0), snIdx: new Uint16Array(vq ? m * ms : 0),
    snCb: vq ? (sec.snCb ?? prev?.snCb ?? null) : null,
  };
  let j = 0;
  for (let i = 0; i < sec.n; i++) {
    if (sec.death[i] !== cont) continue;
    st.ids[j] = sec.ids[i];
    st.sn.set(sec.sn.subarray(i * p.nSn, (i + 1) * p.nSn), j * p.nSn);
    st.k0[j] = kidx.get(sec.k[0][i])!;
    p.groups.forEach(([, d], g) => {
      const endOff = sec.knotOff[g][i + 1];
      st.last[g].set(sec.knots[g].subarray(endOff - d, endOff), j * d);
      if (needVel) endVelocity(sec.knots[g], sec.knotOff[g][i], sec.k[g][i], d, sec.nframes, st.vel[g], j * d);
    });
    if (vq) {
      const dg = sec.snDeg![i];
      st.snLevel[j] = dg ? degs.indexOf(dg) + 1 : 0;
      st.snIdx.set(sec.snIdx!.subarray(i * ms, (i + 1) * ms), j * ms);
    }
    j++;
  }
  return st;
}

export function decodeSection(buf: Uint8Array, st: State | null): { params: Params; sec: Section } {
  if (buf.length < 8) throw new Error('CRC mismatch');
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (crc32(buf, 0, buf.length - 4) !== dv.getUint32(buf.length - 4, true)) throw new Error('CRC mismatch');
  const { params: p, h, lens, o: o0 } = unpackHeader(buf);
  const { key, nCont: nc, nNew: nn, nframes: nf } = h;
  const vq = !!p.snVq;
  if (lens.length !== 5 + (vq ? 3 : 0) + (h.follow ? 1 : 0) + p.groups.length || o0 + lens.reduce((a, b) => a + b, 0) + 4 !== buf.length)
    throw new Error('stream table does not match the section size');
  if (!key && (!st || nc !== st.ids.length)) throw new Error('n_cont differs from the decoder state');
  if (key && h.rescale) throw new Error('RESCALE flag on a key section');
  if (!key && st!.params) {                     // quantisation change (S7): the state moves to this section's table
    const nKnot = p.groups.reduce((a, [, dd]) => a + dd, 0), old = st!.params;
    if (!sameTable(old.offsets, p.offsets, nKnot, p.offsets.length) || !sameTable(old.scales, p.scales, nKnot, p.scales.length))
      throw new Error('the SN quantisation table changed inside a key interval');
    const changed = !sameTable(old.offsets, p.offsets, 0, nKnot) || !sameTable(old.scales, p.scales, 0, nKnot);
    if (changed !== h.rescale) throw new Error('quantisation table change and the RESCALE flag disagree');
    if (h.rescale) st = rescaleState(st!, old, p);
  }
  const n = nc + nn, G = p.groups.length, nk = p.ks.length;
  const m = new Models(p, nf);
  const starts: [number, number][] = [];
  let o = o0;
  for (const ln of lens) { starts.push([o, o + ln]); o += ln; }
  const dec = (i: number) => new RansDecoder(buf, starts[i][0], starts[i][1]);
  const end = (d: RansDecoder) => {
    if (!(d.done() || (d.x === 0 && d.pos === d.end))) throw new Error('stream not consumed exactly');
  };

  let d = dec(0);
  const birth = new Uint16Array(nn);
  let pb = 0;
  for (let i = 0; i < nn; i++) { pb = d.get(m.birth[pb]); birth[i] = pb; }
  end(d);

  d = dec(1);
  const death = new Uint16Array(n);
  for (let i = 0; i < n; i++) death[i] = d.get(m.death[i < nc ? 0 : 1]);
  end(d);

  d = dec(2);
  const ids = new Uint32Array(n);
  if (key) {
    let prev = 0;
    for (let i = 0; i < nc; i++) { prev += d.getUint(m.ids); ids[i] = prev; }
  } else {
    ids.set(st!.ids);
  }
  end(d);
  for (let i = 0; i < nn; i++) ids[nc + i] = h.idBase + i;

  d = dec(ST_K);
  const ki = p.groups.map(() => new Uint8Array(n));
  for (let i = 0; i < n; i++) {
    const c0 = !key && i < nc ? st!.k0[i] : nk;
    const k0 = d.get(m.k[0][c0]);
    ki[0][i] = k0;
    for (let g = 1; g < G; g++) ki[g][i] = d.get(m.k[g][k0]);
  }
  end(d);
  const k = ki.map((row) => row.map((x) => p.ks[x]));

  const knotOff: Uint32Array[] = [], knots: Int32Array[] = [];
  const knnCache: { v: Int32Array | null } = { v: null };
  for (let g = 0; g < G; g++) {
    const dm = p.groups[g][1];
    const off = new Uint32Array(n + 1);
    for (let i = 0; i < n; i++) off[i + 1] = off[i] + k[g][i] * dm;
    const row = new Int32Array(off[n]);
    d = dec(ST_K + 1 + g);
    if (p.pred) {
      const gp = p.pred[g] & PRED_SPATIAL ? p.groups.findIndex(([t, dd]) => t === 'P' && dd === 3) : -1;
      decodeKnotsPred(p, d, key, st, m, g, ki, k, nc, n, off, row, gp >= 0 && gp !== g ? knots[gp] : null,
                      gp >= 0 && gp !== g ? knotOff[gp] : null, knnCache);
      end(d);
      knotOff.push(off); knots.push(row);
      continue;
    }
    const firstRes = m.first[g][0], firstAbs = m.first[g][1], mdiff = m.diff[g];
    const prevAbs = new Int32Array(dm);
    const last = st ? st.last[g] : null;
    for (let i = 0; i < n; i++) {
      const K = k[g][i], b = off[i];
      if (!key && i < nc) {
        if (p.sharedKnot) for (let c = 0; c < dm; c++) row[b + c] = last![i * dm + c];
        else for (let c = 0; c < dm; c++) row[b + c] = last![i * dm + c] + d.getSint(firstRes[c]);
      } else {
        for (let c = 0; c < dm; c++) { row[b + c] = prevAbs[c] + d.getSint(firstAbs[c]); prevAbs[c] = row[b + c]; }
      }
      const mk = mdiff[ki[g][i]];
      for (let c = 0; c < dm; c++) {
        const mc = mk[c];
        let pd = 0, has = false;
        for (let j = 1; j < K; j++) {
          const v = d.getSint(mc[has ? bucket(pd) : 0]);
          row[b + j * dm + c] = row[b + (j - 1) * dm + c] + v;
          pd = v; has = true;
        }
      }
    }
    end(d);
    knotOff.push(off); knots.push(row);
  }

  d = dec(ST_K + 1 + G);
  const ns = p.nSn;
  const sn = new Uint8Array(n * ns);
  if (!key) sn.set(st!.sn);
  const res = new Int32Array(ns);
  const snPred = h.snPred;
  let snDeg: Uint8Array | null = null, snIdx: Uint16Array | null = null, snCb: Codebooks | null = null;
  if (!vq) {
    for (let i = key ? 0 : nc; i < n; i++) {
      for (let c = 0; c < ns; c++) {
        const v = d.getSint(m.sn[c][c >= SN_CH ? bucket6(res[c - SN_CH]) : 0]);
        res[c] = v;
        const val = snPred[c] + v;
        if (val < 0 || val > 255) throw new Error('SN value out of range');
        sn[i * ns + c] = val;
      }
    }
    end(d);
  } else {
    end(d);                                             // sn stream is empty in SN-VQ sections
    const levels = p.snVq!, degs = levels.map((l) => l.deg), ms = maxStages(p);
    d = dec(ST_K + 2 + G);                              // sn_cb (key sections only)
    let cb: Codebooks;
    if (key) {
      snCb = [];
      for (const { deg, sizes } of levels) {
        const dim = snDim(deg), lvl: Int16Array[] = [];
        sizes.forEach((size, si) => {
          const entries = new Int16Array(size * dim);
          const mod = si === 0 ? m.sn : m.cbRes;
          for (let e = 0; e < size; e++) {
            for (let c = 0; c < dim; c++) {
              const v = d.getSint(mod[c][c >= SN_CH ? bucket6(res[c - SN_CH]) : 0]);
              res[c] = v;
              const val = si === 0 ? snPred[c] + v : v;
              if ((si === 0 && (val < 0 || val > 255)) || (si > 0 && (v < -255 || v > 255))) throw new Error('codebook entry out of range');
              entries[e * dim + c] = val;
            }
          }
          lvl.push(entries);
        });
        snCb.push(lvl);
      }
      cb = snCb;
    } else {
      if (!st!.snCb) throw new Error('delta section without codebooks in the decoder state');
      cb = st!.snCb;
    }
    end(d);
    d = dec(ST_K + 3 + G);                              // sn_deg: level per coded splat, context previous level
    const lv = new Uint8Array(n);
    if (!key) lv.set(st!.snLevel);
    let prevLv = 0;
    for (let i = key ? 0 : nc; i < n; i++) { prevLv = d.get(m.deg[prevLv]); lv[i] = prevLv; }
    end(d);
    d = dec(ST_K + 4 + G);                              // sn_idx: one index per stage, model per (level, stage)
    snDeg = new Uint8Array(n);
    snIdx = new Uint16Array(n * ms);
    if (!key) snIdx.set(st!.snIdx);
    for (let i = 0; i < n; i++) snDeg[i] = lv[i] ? degs[lv[i] - 1] : 0;
    for (let i = key ? 0 : nc; i < n; i++) {
      const l = lv[i];
      if (l) { const mi = m.idx[l - 1]; for (let si = 0; si < mi.length; si++) snIdx[i * ms + si] = d.get(mi[si]); }
      vqRow(sn, i * ns, p, l, snIdx, i * ms, cb);
    }
    end(d);
  }
  const sec: Section = { index: h.index, g0: h.g0, nframes: nf, key, n, nCont: nc, ids, birth, death, k, knotOff, knots, sn, snDeg, snIdx, snCb, follow: null };
  if (h.follow) {
    const gP = p.groups.findIndex(([t]) => t === 'P');
    if (gP < 0) throw new Error("follow mode needs a group 'P'");
    d = dec(lens.length - 1);                           // follow masks: always the last stream
    const mask = new Uint32Array(n);
    let any = false;
    for (let i = 0; i < n; i++) {
      let mk = 0, pb = 0;
      const KP = k[gP][i], mm = m.follow[ki[gP][i]];
      if (KP > 32) throw new Error('follow masks support at most 32 knots');
      for (let j = 1; j < KP; j++) { pb = d.get(mm[pb]); mk |= pb << j; }
      mask[i] = mk >>> 0;
      if (mk) any = true;
    }
    end(d);
    if (any) { sec.follow = mask; applyFollow(p, sec); }
  }
  return { params: p, sec };
}

/** sequential decoder; may start at any key section (random access). */
export class Decoder {
  state: State | null = null;
  params: Params | null = null;

  decode(buf: Uint8Array): Section {
    const { h } = unpackHeader(buf);
    let st: State | null = null;
    if (!h.key) {
      if (!this.state || this.state.nextIndex !== h.index)
        throw new Error('non-key section without its previous section (seek to a key section)');
      st = this.state;
    }
    const { params, sec } = decodeSection(buf, st);
    this.params = params;
    this.state = stateAfter(sec, params, this.state);
    return sec;
  }
}
