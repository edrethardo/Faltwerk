// rANS entropy decoder with adaptive models — 1:1 port of codec/rans.py (decoder side), docs/format-csgs.md §3.
// State x stays in [2^23, 2^31) so every operation fits in a JS int32/double exactly.

export const PROB_BITS = 15;
export const PROB_SCALE = 1 << PROB_BITS;
export const RANS_L = 1 << 23;
export const RAW_CHUNK = 15;
export const ESC = 16;
export const NTOK = ESC + 33;
export const INC = 24;
export const IDX_INC = 4;          // SN-VQ codebook-index models (alphabets of 256..16384 symbols), codec/rans.py
const COUNT_LIMIT = 1 << 16;
const REBUILD_FIRST = 16;
const REBUILD_MAX = 1024;
const FIND_LINEAR_MAX = 64;        // larger alphabets are searched by bisection (same symbol, codec/rans.py find)
const LUT_SHIFT = PROB_BITS - 7;   // 128 slot buckets

export class AdaptiveModel {
  n: number;
  counts: Uint32Array;
  total: number;
  cdf: Uint32Array;
  /** first symbol whose interval reaches into slot bucket b (slot >> LUT_SHIFT) */
  lut = new Uint16Array(1 << (PROB_BITS - LUT_SHIFT));
  interval = REBUILD_FIRST;
  until = REBUILD_FIRST;
  inc: number;

  constructor(n: number, inc = INC) {
    if (n < 1 || n > PROB_SCALE / 2) throw new Error('alphabet size out of range');
    this.n = n;
    this.inc = inc;
    this.counts = new Uint32Array(n).fill(1);
    this.total = n;
    this.cdf = new Uint32Array(n + 1);
    this.rebuild();
  }

  rebuild(): void {
    const n = this.n, counts = this.counts, total = this.total, cdf = this.cdf;
    const spare = PROB_SCALE - n;
    let s = 0, best = 0;
    // freq[i] stored temporarily in cdf[i+1]
    for (let i = 0; i < n; i++) {
      const f = 1 + Math.floor((counts[i] * spare) / total);   // < 2^31, exact (see docs: no rounding hazard)
      cdf[i + 1] = f;
      s += f;
      if (counts[i] > counts[best]) best = i;
    }
    cdf[best + 1] += PROB_SCALE - s;
    let acc = 0;
    cdf[0] = 0;
    for (let i = 0; i < n; i++) {
      acc += cdf[i + 1];
      cdf[i + 1] = acc;
    }
    const lut = this.lut;
    let sym = 0;
    for (let b = 0; b < lut.length; b++) {
      const lo = b << LUT_SHIFT;
      while (cdf[sym + 1] <= lo) sym++;
      lut[b] = sym;
    }
  }

  update(s: number): void {
    const counts = this.counts;
    counts[s] += this.inc;
    this.total += this.inc;
    if (this.total > COUNT_LIMIT) {
      let t = 0;
      for (let i = 0; i < this.n; i++) {
        const c = (counts[i] + 1) >>> 1;
        counts[i] = c;
        t += c;
      }
      this.total = t;
    }
    if (--this.until === 0) {
      this.rebuild();
      if (this.interval < REBUILD_MAX) this.interval <<= 1;
      this.until = this.interval;
    }
  }
}

export class RansDecoder {
  buf: Uint8Array;
  pos: number;
  end: number;
  x: number;

  constructor(buf: Uint8Array, pos = 0, end = buf.length) {
    this.buf = buf;
    this.end = end;
    if (end - pos === 0) {
      this.pos = pos;
      this.x = 0;
      return;
    }
    if (end - pos < 4) throw new Error('rANS stream shorter than 4 bytes');
    this.x = (buf[pos] | (buf[pos + 1] << 8) | (buf[pos + 2] << 16) | (buf[pos + 3] << 24)) >>> 0;
    this.pos = pos + 4;
  }

  private renorm(x: number): number {
    const buf = this.buf, end = this.end;
    let pos = this.pos;
    while (x < RANS_L) {
      if (pos >= end) throw new Error('rANS stream overrun');
      x = (x << 8) | buf[pos++];
    }
    this.pos = pos;
    return x;
  }

  get(m: AdaptiveModel): number {
    const x = this.x;
    const slot = x & (PROB_SCALE - 1);
    const cdf = m.cdf;
    // LUT (the contributor, PR #1): start at the first symbol of the slot's bucket; small alphabets then step linearly, large ones
    // (SN-VQ indices) bisect between this bucket's and the next bucket's first symbol (same symbol, codec/rans.py find)
    const b = slot >>> LUT_SHIFT;
    let s = m.lut[b];
    if (m.n <= FIND_LINEAR_MAX) {
      while (cdf[s + 1] <= slot) s++;
    } else {
      let hi = b + 1 < m.lut.length ? m.lut[b + 1] + 1 : m.n;
      while (hi - s > 1) {
        const mid = (s + hi) >>> 1;
        if (cdf[mid] <= slot) s = mid; else hi = mid;
      }
    }
    const start = cdf[s];
    this.x = this.renorm((cdf[s + 1] - start) * (x >>> PROB_BITS) + slot - start);
    m.update(s);
    return s;
  }

  getBits(nbits: number): number {
    let value = 0;
    while (nbits > 0) {
      const nb = nbits > RAW_CHUNK ? RAW_CHUNK : nbits;
      nbits -= nb;
      const x = this.x;
      value = value * (1 << nb) + (x & ((1 << nb) - 1));   // may exceed 2^31: use float arithmetic
      this.x = this.renorm(x >>> nb);
    }
    return value;
  }

  getUint(m: AdaptiveModel): number {
    const t = this.get(m);
    if (t < ESC) return t;
    const n = t - ESC;
    if (n === 0) return ESC;
    if (n <= RAW_CHUNK) {                          // common case: one raw chunk, pure int32 arithmetic
      const x = this.x;
      const bits = x & ((1 << n) - 1);
      this.x = this.renorm(x >>> n);
      return (1 << n) + bits + ESC - 1;
    }
    return 2 ** n + this.getBits(n) + ESC - 1;     // up to 2^32 + ESC: float arithmetic, still exact
  }

  getSint(m: AdaptiveModel): number {
    const u = this.getUint(m);
    if (u < 0x40000000) return u & 1 ? -((u + 1) >> 1) : u >> 1;
    return u % 2 === 0 ? u / 2 : -(u + 1) / 2;
  }

  done(): boolean {
    return this.pos === this.end && this.x === RANS_L;
  }
}
