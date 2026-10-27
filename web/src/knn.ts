// Deterministic spatial helpers of the .csgs v3 `pred` mode — port of codec/knn.py, .
// Positions are integers (knot 0 of group P, quantised). The result of causalKnn is the *exact* k-NN set ordered by
// (squared distance, index), so it does not depend on the grid cell size or the search order — the grid is only speed.
// Arithmetic is done in doubles: exact while the position span stays below 2^25 steps (d2 <= 3*2^50 < 2^53); larger
// spans throw instead of silently diverging from the Python bigints.
export const OCC = 4;
export const SPAN_MAX = 2 ** 25;

/** floor cube root of a non-negative bigint (integer Newton, as codec/knn.py _icbrt). */
export function icbrt(v: bigint): number {
  if (v <= 0n) return 0;
  let x = 1n << BigInt(Math.floor((v.toString(2).length + 2) / 3));
  for (;;) {
    const y = (2n * x + v / (x * x)) / 3n;
    if (y >= x) break;
    x = y;
  }
  while (x * x * x > v) x -= 1n;
  while ((x + 1n) ** 3n <= v) x += 1n;
  return Number(x);
}

/** cell edge: bbox rule cbrt(volume*8/n), halved while the mean occupancy of the non-empty cells exceeds OCC. */
export function gridCell(lo: number[], hi: number[], n: number, pos: Int32Array | null): number {
  const ext = [0, 1, 2].map((c) => Math.max(1, hi[c] - lo[c] + 1));
  const vol = BigInt(ext[0]) * BigInt(ext[1]) * BigInt(ext[2]);
  let cell = Math.max(1, icbrt((vol * 8n) / BigInt(Math.max(1, n))));
  if (pos) {
    while (cell > 1) {
      const ny = Math.floor((hi[1] - lo[1]) / cell) + 1, nz = Math.floor((hi[2] - lo[2]) / cell) + 1;
      const set = new Set<number>();
      for (let i = 0; i < n; i++) {
        const a = Math.floor((pos[i * 3] - lo[0]) / cell), b = Math.floor((pos[i * 3 + 1] - lo[1]) / cell), c = Math.floor((pos[i * 3 + 2] - lo[2]) / cell);
        set.add((a * ny + b) * nz + c);
      }
      if (n <= OCC * set.size) break;
      cell = Math.max(1, Math.floor(cell / 2));
    }
  }
  return cell;
}

/** pos: n*3 ints (knot 0 of group P in section order). Returns n*k indices: for splat i its k nearest among 0..i-1
 *  (ordered by (d2, index)), padded with -1. */
export function causalKnn(pos: Int32Array, n: number, k: number): Int32Array {
  const out = new Int32Array(n * k).fill(-1);
  if (n === 0 || k === 0) return out;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) { const v = pos[i * 3 + c]; if (v < lo[c]) lo[c] = v; if (v > hi[c]) hi[c] = v; }
  const maxExt = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  if (maxExt >= SPAN_MAX) throw new Error('position span too large for exact double arithmetic (knn)');
  let cell = gridCell(lo, hi, n, pos);
  const dims = () => [0, 1, 2].map((c) => Math.floor((hi[c] - lo[c]) / cell) + 1);
  while (dims().reduce((a, b) => a * b, 1) >= 2 ** 52) cell *= 2;   // the cell size only affects speed, never the result
  const [nx, ny, nz] = dims();
  // Static CSR grid (points of a cell in ascending index; a query for point i stops at the first index >= i, which is
  // exactly the causal "already inserted" set). Direct table for up to 2^23 cells, a Map beyond that.
  const total = nx * ny * nz;
  const keyOf = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    keyOf[i] = ((Math.floor((pos[i * 3] - lo[0]) / cell) * ny) + Math.floor((pos[i * 3 + 1] - lo[1]) / cell)) * nz + Math.floor((pos[i * 3 + 2] - lo[2]) / cell);
  }
  const idx = new Int32Array(n);
  let startArr: Int32Array | null = null;
  let cellMap: Map<number, [number, number]> | null = null;
  let nonEmpty = 0;
  if (total <= 2 ** 23) {
    startArr = new Int32Array(total + 1);
    for (let i = 0; i < n; i++) startArr[keyOf[i] + 1]++;
    for (let c = 0; c < total; c++) { if (startArr[c + 1]) nonEmpty++; startArr[c + 1] += startArr[c]; }
    const cur = startArr.slice(0, total);
    for (let i = 0; i < n; i++) idx[cur[keyOf[i]]++] = i;
  } else {
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => keyOf[a] - keyOf[b] || a - b);
    cellMap = new Map();
    for (let t = 0; t < n; t++) {
      idx[t] = order[t];
      const e = cellMap.get(keyOf[order[t]]);
      if (e) e[1] = t + 1; else { cellMap.set(keyOf[order[t]], [t, t + 1]); nonEmpty++; }
    }
  }
  const bestD = new Float64Array(k), bestJ = new Int32Array(k);
  let cnt = 0;
  // insert (d2, j) keeping bestD/bestJ sorted by (d2, j), at most k entries
  const insert = (d2: number, j: number) => {
    if (cnt === k && !(d2 < bestD[k - 1] || (d2 === bestD[k - 1] && j < bestJ[k - 1]))) return;
    let q = cnt < k ? cnt : k - 1;
    while (q > 0 && (d2 < bestD[q - 1] || (d2 === bestD[q - 1] && j < bestJ[q - 1]))) { bestD[q] = bestD[q - 1]; bestJ[q] = bestJ[q - 1]; q--; }
    bestD[q] = d2; bestJ[q] = j;
    if (cnt < k) cnt++;
  };
  for (let i = 1; i < n; i++) {
    const px = pos[i * 3], py = pos[i * 3 + 1], pz = pos[i * 3 + 2];
    const cx = Math.floor((px - lo[0]) / cell), cy = Math.floor((py - lo[1]) / cell), cz = Math.floor((pz - lo[2]) / cell);
    cnt = 0;
    let r = 0;
    const scan = (s: number, e: number) => {
      for (let t = s; t < e; t++) {
        const j = idx[t];
        if (j >= i) break;
        const dx = pos[j * 3] - px, dy = pos[j * 3 + 1] - py, dz = pos[j * 3 + 2] - pz;
        insert(dx * dx + dy * dy + dz * dz, j);
      }
    };
    for (;;) {
      for (let dx = -r; dx <= r; dx++) {
        const ax = cx + dx;
        if (ax < 0 || ax >= nx) continue;
        for (let dy = -r; dy <= r; dy++) {
          const ay = cy + dy;
          if (ay < 0 || ay >= ny) continue;
          const edgeXY = Math.abs(dx) === r || Math.abs(dy) === r;
          const step = edgeXY ? 1 : 2 * r;                 // interior of the xy-square: only the two z faces
          for (let dz = -r; dz <= r; dz += step) {
            const az = cz + dz;
            if (az < 0 || az >= nz) continue;
            const key = (ax * ny + ay) * nz + az;
            if (startArr) scan(startArr[key], startArr[key + 1]);
            else { const e = cellMap!.get(key); if (e) scan(e[0], e[1]); }
          }
        }
      }
      if (cnt === k && bestD[k - 1] <= (r * cell) * (r * cell)) break;
      if (cnt === Math.min(k, i) && r * cell > maxExt) break;
      r++;
      if ((2 * r + 1) ** 3 > 8 * nonEmpty) {              // sparse / outlier case: exact brute force over the earlier points
        cnt = 0;
        for (let j = 0; j < i; j++) {
          const dx = pos[j * 3] - px, dy = pos[j * 3 + 1] - py, dz = pos[j * 3 + 2] - pz;
          insert(dx * dx + dy * dy + dz * dz, j);
        }
        break;
      }
    }
    for (let t = 0; t < cnt; t++) out[i * k + t] = bestJ[t];
  }
  return out;
}
