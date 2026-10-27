// Section -> GPU layout (dequantised knots) + CPU reference of the WGSL curve evaluation (used by tests and by the
// in-browser GPU/CPU cross-check). Convention (codec/prep.py, gsplat): P position, R quaternion wxyz (nlerp),
// S log-scale, O opacity logit, D sh0 (DC), SN 15x3 higher SH bands (coefficient-major, static per splat).
import type { Params, Section } from './csgs.ts';
import { bsplineWeights } from './bspline.ts';

export const GROUP_TAGS = ['P', 'R', 'S', 'O', 'D'];
export const GROUP_DIMS = [3, 4, 3, 1, 3];
export const HDR_U32 = 8;       // birthAbs, deathAbs, kpack (4 bit per group), knot offset per group (floats)
export const SN_U32 = 12;       // 45 bytes padded to 48

export interface PackedSection {
  index: number; g0: number; nframes: number;
  n: number;                    // splats in the GPU buffers (incl. replicas)
  nCoded: number;               // splats in the bitstream section
  hdr: Uint32Array;             // n * HDR_U32
  knots: Float32Array;          // dequantised knots of all groups
  sn: Uint32Array;              // n * SN_U32 (u8 packed)
  snOffset: Float32Array;       // 45
  snScale: Float32Array;        // 45
}

export function checkParams(p: Params): void {
  const ok = p.groups.length === 5 && p.groups.every(([t, d], g) => t === GROUP_TAGS[g] && d === GROUP_DIMS[g]);
  if (!ok || p.nSn !== 45 || p.ks.some((k) => k > 15))
    throw new Error(`renderer expects groups P3 R4 S3 O1 D3, 45 SN, K<=15; got ${JSON.stringify(p.groups)} n_sn=${p.nSn}`);
}

/** replicate>n adds copies of the first splats with a small position offset (stress test for splat count). */
export function packSection(sec: Section, p: Params, targetN = 0): PackedSection {
  checkParams(p);
  const nc = sec.n, n = Math.max(nc, targetN | 0);
  const hdr = new Uint32Array(n * HDR_U32);
  let total = 0;
  for (let g = 0; g < 5; g++) total += sec.knots[g].length;
  // replicas reuse the source's knots of R S O D and get their own, shifted P knots
  const knots = new Float32Array(total + countReplicaP(sec, n));
  const base = new Uint32Array(5);
  let q0 = 0, o = 0;
  for (let g = 0; g < 5; g++) {
    base[g] = o;
    const d = GROUP_DIMS[g], src = sec.knots[g];
    for (let i = 0; i < src.length; i++) {
      const c = i % d;
      knots[o + i] = p.offsets[q0 + c] + src[i] * p.scales[q0 + c];
    }
    o += src.length; q0 += d;
  }
  for (let i = 0; i < nc; i++) {
    const h = i * HDR_U32;
    hdr[h] = i < sec.nCont ? 0 : sec.g0 + sec.birth[i - sec.nCont];
    hdr[h + 1] = sec.g0 + sec.death[i];
    let kp = 0;
    for (let g = 0; g < 5; g++) {
      kp |= sec.k[g][i] << (4 * g);
      hdr[h + 3 + g] = base[g] + sec.knotOff[g][i];
    }
    hdr[h + 2] = p.bspline ? (kp | 0x80000000) >>> 0 : kp;           // bit 31: B-spline basis (v6), WGSL eval_curves reads it
  }
  // replicas: copy header, give each its own P knots shifted by a deterministic offset
  for (let r = nc, src = 0; r < n; r++, src = (src + 1) % nc) {
    const copy = Math.floor((r - nc) / nc) + 1;
    hdr.copyWithin(r * HDR_U32, src * HDR_U32, (src + 1) * HDR_U32);
    const kP = sec.k[0][src];
    const from = base[0] + sec.knotOff[0][src];
    hdr[r * HDR_U32 + 3] = o;
    const dx = 0.006 * copy, dy = -0.004 * copy, dz = 0.005 * copy;
    for (let j = 0; j < kP; j++) {
      knots[o++] = knots[from + 3 * j] + dx;
      knots[o++] = knots[from + 3 * j + 1] + dy;
      knots[o++] = knots[from + 3 * j + 2] + dz;
    }
  }
  const sn = new Uint32Array(n * SN_U32);
  const sn8 = new Uint8Array(sn.buffer);
  for (let i = 0; i < n; i++) {
    const s = i < nc ? i : (i - nc) % nc;
    sn8.set(sec.sn.subarray(s * 45, s * 45 + 45), i * SN_U32 * 4);
  }
  const qs = 14;
  return {
    index: sec.index, g0: sec.g0, nframes: sec.nframes, n, nCoded: nc, hdr, knots, sn,
    snOffset: p.offsets.slice(qs, qs + 45), snScale: p.scales.slice(qs, qs + 45),
  };
}

function countReplicaP(sec: Section, n: number): number {
  let c = 0;
  for (let r = sec.n, src = 0; r < n; r++, src = (src + 1) % sec.n) c += sec.k[0][src] * 3;
  return c;
}

/** CPU mirror of eval_curves in shaders.ts (f32 math emulated with Math.fround where it matters). */
export function evalSplat(ps: PackedSection, i: number, t: number): { alive: boolean; vals: number[][] } {
  const h = i * HDR_U32;
  const fi = Math.floor(t);
  const alive = ps.hdr[h] <= fi && fi < ps.hdr[h + 1];
  const span = Math.max(ps.nframes - 1, 1);
  const vals: number[][] = [];
  for (let g = 0; g < 5; g++) {
    const K = (ps.hdr[h + 2] >>> (4 * g)) & 15, d = GROUP_DIMS[g], off = ps.hdr[h + 3 + g];
    const v: number[] = [];
    if (ps.hdr[h + 2] >>> 31) {                          // v6: clamped B-spline, weights exactly as codec.csgs.bspline_weights
      const u = Math.min(Math.max((t - ps.g0) / span, 0), 1);
      const wts = bsplineWeights(K, u);
      for (let c = 0; c < d; c++) {
        let acc = 0;
        for (let j = 0; j < K; j++) if (wts[j] !== 0) acc += wts[j] * ps.knots[off + j * d + c];
        v.push(acc);
      }
    } else {
      let j = 0, w = 0;
      if (K > 1) {
        const tau = Math.min(Math.max(((t - ps.g0) * (K - 1)) / span, 0), K - 1);
        j = Math.min(Math.floor(tau), K - 2);
        w = tau - j;
      }
      for (let c = 0; c < d; c++) {
        const a = ps.knots[off + j * d + c];
        const b = K > 1 ? ps.knots[off + (j + 1) * d + c] : a;
        v.push(a + (b - a) * w);
      }
    }
    if (g === 1) {
      const nrm = Math.hypot(v[0], v[1], v[2], v[3]) || 1;
      for (let c = 0; c < 4; c++) v[c] /= nrm;
    }
    vals.push(v);
  }
  return { alive, vals };
}
