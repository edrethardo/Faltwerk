// Clamped uniform B-spline weights of .csgs version 6 / flag BSPLINE (docs/format-csgs.md §11, ).
// `bsplineWeights` is a 1:1 port of codec.csgs.bspline_weights (Cox-de Boor over all K basis functions, IEEE double, same
// operation order -> the same bits as Python; tested bit-for-bit). `bsplineLocal` is the local-support form used by the WGSL
// eval pass (NURBS book A2.2: only the p+1 non-zero weights + the first control point): equal up to rounding, tested to 1e-12.

export const BSPLINE_DEGREE = 3;

export function bsplineWeights(K: number, uIn: number): number[] {
  const p = Math.min(BSPLINE_DEGREE, K - 1);
  if (p <= 0) return [1.0];
  const nInt = K - p - 1;
  const kn: number[] = new Array(p + 1).fill(0.0);
  for (let i = 0; i < nInt; i++) kn.push((i + 1) / (nInt + 1));
  for (let i = 0; i < p + 1; i++) kn.push(1.0);
  const u = Math.min(Math.max(uIn, 0.0), 1.0);
  const m = kn.length - 1;
  let N: number[] = [];
  for (let i = 0; i < m; i++) N.push((kn[i] <= u && u < kn[i + 1]) || (u === 1.0 && kn[i] < kn[i + 1] && kn[i + 1] === 1.0) ? 1.0 : 0.0);
  for (let d = 1; d <= p; d++) {
    const M: number[] = [];
    for (let i = 0; i < m - d; i++) {
      const a = kn[i + d] > kn[i] ? ((u - kn[i]) / (kn[i + d] - kn[i])) * N[i] : 0.0;
      const b = kn[i + d + 1] > kn[i + 1] ? ((kn[i + d + 1] - u) / (kn[i + d + 1] - kn[i + 1])) * N[i + 1] : 0.0;
      M.push(a + b);
    }
    N = M;
  }
  return N.slice(0, K);
}

/** local form: first control point `s0` and the p+1 weights of control points s0 .. s0+p (the others are 0). */
export function bsplineLocal(K: number, uIn: number): { s0: number; w: number[] } {
  const p = Math.min(BSPLINE_DEGREE, K - 1);
  if (p <= 0) return { s0: 0, w: [1.0] };
  const u = Math.min(Math.max(uIn, 0.0), 1.0);
  const kn = (i: number) => Math.min(Math.max((i - p) / (K - p), 0), 1);      // clamped uniform knot vector
  const s = u >= 1.0 ? K - 1 : Math.min(K - 1, p + Math.floor(u * (K - p)));
  const N = new Array(p + 1).fill(0.0), left = new Array(p + 1).fill(0.0), right = new Array(p + 1).fill(0.0);
  N[0] = 1.0;
  for (let j = 1; j <= p; j++) {
    left[j] = u - kn(s + 1 - j);
    right[j] = kn(s + j) - u;
    let saved = 0.0;
    for (let r = 0; r < j; r++) {
      const temp = N[r] / (right[r + 1] + left[j - r]);
      N[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    N[j] = saved;
  }
  return { s0: s - p, w: N };
}
