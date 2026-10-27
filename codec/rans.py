"""rANS entropy coder with adaptive frequency models (decoder core of the .csgs container).

Portable by design: only integers below 2**32 (products below 2**33 where noted), no floats, no numpy, no
dependence on Python's big ints. Every function here maps 1:1 onto TypeScript/WASM (u32 state, byte I/O):

  * byte-wise rANS (ryg_rans "rans_byte" layout): state x in [RANS_L, RANS_L * 256), RANS_L = 2**23, symbols with
    a total of 2**PROB_BITS (models) or 2**n (raw bits, n <= RAW_CHUNK). Encoder runs backwards over the symbols
    and the byte buffer is reversed at the end, so the decoder reads forward.
  * AdaptiveModel: counts per symbol, rebuilt into a CDF with total 2**PROB_BITS on a fixed schedule (after 16, 32,
    ... up to REBUILD_MAX coded symbols); counts are halved when their sum exceeds COUNT_LIMIT. Encoder and decoder
    call the same update() after every symbol, so both sides see identical CDFs.
  * Value tokens: an unsigned value u < ESC is its own token; larger values are token ESC + n followed by n raw bits
    (Exp-Golomb-like: v = u - ESC + 1, n = floor(log2 v), raw = v - 2**n). Signed values are zig-zag mapped first.
See docs/format-csgs.md.
"""

PROB_BITS = 15
PROB_SCALE = 1 << PROB_BITS
RANS_L = 1 << 23
RAW_CHUNK = 15            # raw bits are sent in chunks of at most 15 bits
ESC = 16                  # tokens 0..ESC-1 are literal values
NTOK = ESC + 33           # ESC + n for n = 0..32 -> values up to 2**32 + ESC - 2
INC = 24                  # count increment per coded symbol
IDX_INC = 4               # increment of the SN-VQ codebook-index models (alphabets of 256..16384 symbols)
COUNT_LIMIT = 1 << 16     # halve counts when their sum exceeds this (keeps c * PROB_SCALE < 2**31)
REBUILD_FIRST = 16
REBUILD_MAX = 1024
FIND_LINEAR_MAX = 64      # alphabets up to this size are searched linearly, larger ones by bisection (same result)


class AdaptiveModel:
    __slots__ = ('n', 'counts', 'total', 'cdf', 'interval', 'until', 'inc')

    def __init__(self, n, inc=INC):
        """inc: count increment per coded symbol (INC for the v1 models; the SN-VQ index models with alphabets of
        thousands of symbols use a smaller one, IDX_INC, so the estimate averages over more symbols)."""
        if n < 1 or n > PROB_SCALE // 2:
            raise ValueError('alphabet size out of range')
        self.n = n
        self.inc = inc
        self.counts = [1] * n
        self.total = n
        self.interval = REBUILD_FIRST
        self.until = REBUILD_FIRST
        self.cdf = None
        self.rebuild()

    def rebuild(self):
        """cdf[0..n], cdf[n] = PROB_SCALE, every frequency >= 1; rest goes to the most frequent symbol (lowest index)."""
        n, counts, total = self.n, self.counts, self.total
        spare = PROB_SCALE - n
        freq = [0] * n
        s = 0
        best = 0
        for i in range(n):
            f = 1 + (counts[i] * spare) // total
            freq[i] = f
            s += f
            if counts[i] > counts[best]:
                best = i
        freq[best] += PROB_SCALE - s
        cdf = [0] * (n + 1)
        acc = 0
        for i in range(n):
            cdf[i] = acc
            acc += freq[i]
        cdf[n] = acc
        self.cdf = cdf

    def update(self, s):
        counts = self.counts
        counts[s] += self.inc
        self.total += self.inc
        if self.total > COUNT_LIMIT:
            t = 0
            for i in range(self.n):
                c = (counts[i] + 1) >> 1
                counts[i] = c
                t += c
            self.total = t
        self.until -= 1
        if self.until == 0:
            self.rebuild()
            if self.interval < REBUILD_MAX:
                self.interval <<= 1
            self.until = self.interval

    def find(self, slot):
        """symbol s with cdf[s] <= slot < cdf[s+1]: linear search for small alphabets, binary search for large ones
        (SN-VQ index alphabets reach 4096+); both give the same symbol."""
        cdf = self.cdf
        if self.n <= FIND_LINEAR_MAX:
            s = 0
            while cdf[s + 1] <= slot:
                s += 1
            return s
        lo, hi = 0, self.n
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if cdf[mid] <= slot:
                lo = mid
            else:
                hi = mid
        return lo


class Encoder:
    """Collects symbols in coding order; finish() runs rANS backwards and returns the stream bytes."""
    __slots__ = ('syms',)

    def __init__(self):
        self.syms = []      # flat list: start, freq, scale_bits, start, freq, ...

    def put(self, model, s):
        cdf = model.cdf
        start = cdf[s]
        self.syms += (start, cdf[s + 1] - start, PROB_BITS)
        model.update(s)

    def put_bits(self, value, nbits):
        while nbits > 0:
            nb = RAW_CHUNK if nbits > RAW_CHUNK else nbits
            nbits -= nb
            self.syms += ((value >> nbits) & ((1 << nb) - 1), 1, nb)

    def put_uint(self, model, u):
        if u < ESC:
            self.put(model, u)
            return
        v = u - ESC + 1
        n = v.bit_length() - 1
        self.put(model, ESC + n)
        if n:
            self.put_bits(v - (1 << n), n)

    def put_sint(self, model, v):
        self.put_uint(model, (v << 1) if v >= 0 else ((-v) << 1) - 1)

    def finish(self):
        if not self.syms:
            return b''
        syms = self.syms
        out = bytearray()
        x = RANS_L
        for i in range(len(syms) - 3, -1, -3):
            start, freq, sb = syms[i], syms[i + 1], syms[i + 2]
            x_max = ((RANS_L >> sb) << 8) * freq
            while x >= x_max:
                out.append(x & 255)
                x >>= 8
            x = ((x // freq) << sb) + (x % freq) + start
        out += bytes(((x >> 24) & 255, (x >> 16) & 255, (x >> 8) & 255, x & 255))
        out.reverse()
        return bytes(out)


class Decoder:
    __slots__ = ('buf', 'pos', 'end', 'x')

    def __init__(self, buf, pos=0, end=None):
        self.buf = buf
        self.end = len(buf) if end is None else end
        if self.end - pos == 0:
            self.pos, self.x = pos, 0      # empty stream: any get() is an error
            return
        if self.end - pos < 4:
            raise ValueError('rANS stream shorter than 4 bytes')
        self.x = buf[pos] | (buf[pos + 1] << 8) | (buf[pos + 2] << 16) | (buf[pos + 3] << 24)
        self.pos = pos + 4

    def _renorm(self, x):
        buf, pos, end = self.buf, self.pos, self.end
        while x < RANS_L:
            if pos >= end:
                raise ValueError('rANS stream overrun')
            x = (x << 8) | buf[pos]
            pos += 1
        self.pos = pos
        return x

    def get(self, model):
        x = self.x
        slot = x & (PROB_SCALE - 1)
        s = model.find(slot)
        cdf = model.cdf
        start = cdf[s]
        x = (cdf[s + 1] - start) * (x >> PROB_BITS) + slot - start
        self.x = self._renorm(x)
        model.update(s)
        return s

    def get_bits(self, nbits):
        value = 0
        while nbits > 0:
            nb = RAW_CHUNK if nbits > RAW_CHUNK else nbits
            nbits -= nb
            x = self.x
            value = (value << nb) | (x & ((1 << nb) - 1))
            self.x = self._renorm(x >> nb)
        return value

    def get_uint(self, model):
        t = self.get(model)
        if t < ESC:
            return t
        n = t - ESC
        v = (1 << n) | (self.get_bits(n) if n else 0)
        return v + ESC - 1

    def get_sint(self, model):
        u = self.get_uint(model)
        return (u >> 1) if (u & 1) == 0 else -((u + 1) >> 1)

    def done(self):
        """True when the stream is consumed exactly (final state must be RANS_L, all bytes read)."""
        return self.pos == self.end and self.x == RANS_L
