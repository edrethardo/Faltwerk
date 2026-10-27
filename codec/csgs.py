""".csgs container v1: one self-describing byte block per section (CurveStream-GS bitstream, paper §4.4).

Decoder core: pure Python ints/lists + struct + zlib.crc32, no numpy, written to be ported 1:1 to TypeScript/WASM.
Layout, symbol order and context models: docs/format-csgs.md (keep both in sync).

Data model of a section (what the decoder returns; the encoder takes the same):
  ids      global splat ids of every splat alive in the section, strictly increasing; continuing splats (born before
           g0) first, then newborns with consecutive ids id_base, id_base+1, ...
  n_cont   number of continuing splats
  birth    per newborn: birth frame - g0, in [0, nframes)
  death    per splat: death frame - g0 in [1, nframes] (dies inside / at the end of the section) or nframes+1
           (continues into the next section)
  k        per attribute group: list of K per splat (K must be one of params.ks)
  knots    per attribute group: per splat a flat list of K*dims ints (knot-major: k0d0, k0d1, ..., k1d0, ...)
  sn       per splat: n_sn ints in 0..255 (static higher SH bands; sent at birth and in key sections)

SN-VQ mode (version 2, flag SN_VQ; / S1): the higher SH bands of a splat are an SH degree (0, 1, 2 or 3 =
the first 0 / 9 / 24 / 45 of the n_sn scalars) plus, per degree level > 0, one codebook index per residual stage.
Codebooks live in the KEY section (one per level and stage, entries in the same 0..255 grid; residual stages are
signed steps) and are carried in the decoder state to the delta sections of that key interval. The decoder still
returns `sn` rows (the reconstruction: codebook sum clipped to 0..255, truncated bands at the grid point of 0.0), so
everything downstream of the container is unchanged; `sn_deg`, `sn_idx`, `sn_cb` are returned next to it.
Sections without the flag are byte-identical to version 1.

Follow mode (flag FOLLOW = 16, version 4; / P8b): per splat a bit mask over the knots of group 'P' (bit j = knot j);
masked knots are NOT sent but predicted by the decoder from the mean displacement of the FOLLOW_K nearest explicit splats
(apply_follow). Knot 0 is never masked. `Section.follow` is None / all zero for a plain section. (P8b first used flag 4 /
version 2 for this; renumbered at the integration because S1 owns 4 / 2.) Flags are independent and combinable; the
version is the highest any set flag needs (SN_VQ 2, pred 3, follow 4).
"""
import math
import struct
import zlib

from .rans import IDX_INC, NTOK, AdaptiveModel, Decoder as RansDecoder, Encoder as RansEncoder

MAGIC = b'CSGS'
VERSION = 1
VERSION_SN_VQ = 2                                       # written only for sections with FLAG_SN_VQ (and no FLAG_PRED)
VERSION_PRED = 3                                        # written for sections with FLAG_PRED (S2, ), with or without SN_VQ
FLAG_KEY = 1
FLAG_SHARED_KNOT = 2
FLAG_SN_VQ = 4
FLAG_PRED = 8                                           # motion-prediction mode
FLAG_FOLLOW = 16                                        # follow mode for dormant splats
VERSION_FOLLOW = 4                                      # written for sections with FLAG_FOLLOW (any other flags too)
FLAG_RESCALE = 32                                       # quantisation table differs from the previous section (S7, )
VERSION_RESCALE = 5                                     # written for sections with FLAG_RESCALE (any other flags too)
FLAG_BSPLINE = 64                                       # knots are control points of a clamped B-spline (P16, )
VERSION_BSPLINE = 6                                     # written for sections with FLAG_BSPLINE (any other flags too)
BSPLINE_DEGREE = 3                                      # degree of the spline, lowered to K-1 when K <= 3 (K=2 -> linear)
FOLLOW_K = 4                                            # neighbours per follower
FOLLOW_CELL = 8                                         # grid cell (quantisation steps of P) of the neighbour search (speed only: exact kNN)
PRED_TEMPORAL, PRED_SPATIAL, PRED_VEL0 = 1, 2, 4        # per-group mode bits, see docs/format-csgs.md §8
KNOT_MIN, KNOT_MAX = -(1 << 30), (1 << 30) - 1     # residuals then fit in int32
ST_BIRTH, ST_DEATH, ST_IDS, ST_K = 0, 1, 2, 3          # then one knot stream per group, then SN (+ sn_cb, sn_deg, sn_idx)
SN_CH = 3                  # SN scalars are coefficient-major, SN_CH colour channels per coefficient
MAX_CB = 16384             # codebook entries per stage (AdaptiveModel alphabet limit)


def bspline_weights(K, u):
    """Version 6 / flag BSPLINE: weights of the K control points of a clamped uniform B-spline at u in [0, 1] (u = (t - g0) /
    max(nframes - 1, 1)). Degree p = min(BSPLINE_DEGREE, K - 1); the knot vector has p+1 zeros, K-p-1 uniform interior knots
    (i+1)/(K-p), p+1 ones. Cox-de Boor recursion in IEEE double in exactly this operation order (Python = JS; the TS port
    web/src/csgs.ts compares equal bit for bit). Weights sum to 1, u = 0 -> control point 0, u = 1 -> control point K-1
    (shared / soft boundary knots keep their meaning). K = 1 -> [1.0], K = 2 -> linear = the hat basis."""
    p = min(BSPLINE_DEGREE, K - 1)
    if p <= 0:
        return [1.0]
    n_int = K - p - 1
    kn = [0.0] * (p + 1) + [(i + 1) / (n_int + 1) for i in range(n_int)] + [1.0] * (p + 1)
    u = min(max(float(u), 0.0), 1.0)
    m = len(kn) - 1
    N = [1.0 if (kn[i] <= u < kn[i + 1]) or (u == 1.0 and kn[i] < kn[i + 1] == 1.0) else 0.0 for i in range(m)]
    for d in range(1, p + 1):
        N = [((u - kn[i]) / (kn[i + d] - kn[i]) * N[i] if kn[i + d] > kn[i] else 0.0) +
             ((kn[i + d + 1] - u) / (kn[i + d + 1] - kn[i + 1]) * N[i + 1] if kn[i + d + 1] > kn[i + 1] else 0.0)
             for i in range(m - d)]
    return N[:K]


def sn_dim(deg):
    """number of SN scalars covered by SH degree deg (bands 1..deg, SN_CH channels each)."""
    return SN_CH * ((deg + 1) ** 2 - 1)


class Params:
    """stream-wide parameters, repeated in every section header (sections are self-describing).
    sn_vq: None (v1) or [(deg, [stage sizes...]), ...] with strictly increasing deg in 1..3 (degree 0 is implicit)."""

    def __init__(self, L, ks, groups, n_sn, offsets, scales, shared_knot=True, key_interval=4, sn_vq=None, sn_zero=None,
                 pred=None, knn_k=0, bspline=False):
        """bspline: curve basis of the whole stream (False = hat/linear, P1; True = clamped cubic B-spline, flag 64 / v6, see
        bspline_weights); the integers of the bitstream are control points either way.
        pred: None (v1 knot coding) or one mode per group (bit PRED_TEMPORAL: knot differences j >= 2 as second
        differences; PRED_SPATIAL: differences predicted by the mean of the knn_k nearest already-decoded splats,
        positions = knot 0 of group 'P'; PRED_VEL0: knot 0 of a continuing splat predicted by its previous section's
        end velocity). knn_k: 1..255 neighbours (needed when any group has PRED_SPATIAL). sn_vq and pred are independent and combinable."""
        self.L = int(L)
        self.ks = [int(k) for k in ks]
        self.groups = [(str(t), int(d)) for t, d in groups]
        self.n_sn = int(n_sn)
        f32 = lambda v: struct.unpack('<f', struct.pack('<f', float(v)))[0]     # header stores float32
        self.offsets = [f32(v) for v in offsets]
        self.scales = [f32(v) for v in scales]
        self.shared_knot = bool(shared_knot)
        self.key_interval = int(key_interval)
        self.sn_vq = None if sn_vq is None else [(int(d), [int(s) for s in sizes]) for d, sizes in sn_vq]
        # SN-VQ: grid point of the value 0.0 per SN scalar (used for truncated bands); the encoder derives it from the
        # quantisation table, the decoder reads it from the header (integer-only)
        self.sn_zero = None if self.sn_vq is None else ([0] * self.n_sn if sn_zero is None else [int(v) for v in sn_zero])
        self.pred = None if pred is None else [int(m) for m in pred]
        self.knn_k = int(knn_k)
        self.bspline = bool(bspline)
        nq = sum(d for _, d in self.groups) + self.n_sn
        if len(self.offsets) != nq or len(self.scales) != nq:
            raise ValueError(f'need {nq} quantisation offsets/scales')
        if not (1 <= self.L <= 253) or not self.ks or any(not 1 <= k <= 255 for k in self.ks) or len(set(self.ks)) != len(self.ks):
            raise ValueError('bad L or ks')
        if not 1 <= self.key_interval <= 255 or not 0 <= self.n_sn <= 65535 or len(self.groups) > 255:
            raise ValueError('key_interval 1..255, n_sn <= 65535, <= 255 groups')
        if any(len(t) != 1 or not 0 < ord(t) < 128 or not 1 <= d <= 255 for t, d in self.groups):
            raise ValueError('group tags are single ASCII chars, dims 1..255')
        if self.sn_vq is not None:
            degs = [d for d, _ in self.sn_vq]
            if not degs or len(degs) > 255 or degs != sorted(set(degs)) or degs[0] < 1 or degs[-1] > 3 \
                    or sn_dim(degs[-1]) > self.n_sn:
                raise ValueError('sn_vq levels: strictly increasing degrees in 1..3 with sn_dim(deg) <= n_sn')
            for _, sizes in self.sn_vq:
                if not 1 <= len(sizes) <= 255 or any(not 1 <= s <= MAX_CB for s in sizes):
                    raise ValueError(f'sn_vq stages: 1..255 stages of 1..{MAX_CB} entries')
            if len(self.sn_zero) != self.n_sn or any(not 0 <= v <= 255 for v in self.sn_zero):
                raise ValueError('sn_zero needs n_sn values in 0..255')
        if self.pred is not None:
            if len(self.pred) != len(self.groups) or any(not 0 <= m <= 7 for m in self.pred):
                raise ValueError('pred needs one mode 0..7 per group')
            if any(m & PRED_SPATIAL for m in self.pred):
                if not 1 <= self.knn_k <= 255:
                    raise ValueError('knn_k 1..255 for spatial prediction')
                if ('P', 3) not in self.groups:
                    raise ValueError("spatial prediction needs a group 'P' with 3 dims")
                gp = self.groups.index(('P', 3))
                if any(m & PRED_SPATIAL for m in self.pred[:gp]):
                    raise ValueError("group 'P' must be coded before any spatially predicted group")
            elif self.knn_k:
                raise ValueError('knn_k only with spatial prediction')

    def __eq__(self, o):
        return isinstance(o, Params) and vars(self) == vars(o)


class Section:
    FIELDS = ('index', 'g0', 'nframes', 'ids', 'n_cont', 'birth', 'death', 'k', 'knots', 'sn', 'sn_deg', 'sn_idx', 'sn_cb',
              'follow')

    def __init__(self, index, g0, nframes, ids, n_cont, birth, death, k, knots, sn, key=None,
                 sn_deg=None, sn_idx=None, sn_cb=None, follow=None):
        """SN-VQ sections carry sn_deg (per splat: SH degree, one of 0 and the params.sn_vq degrees), sn_idx (per
        splat: one codebook index per stage of its level, [] for degree 0) and, in KEY sections, sn_cb (per level:
        per stage: list of entries; stage 0 entries are 0..255 rows of sn_dim(deg) values, later stages signed
        steps). `sn` must then equal vq_rows(...) of these (the decoder returns exactly that)."""
        self.index, self.g0, self.nframes = index, g0, nframes
        self.ids, self.n_cont, self.birth, self.death = ids, n_cont, birth, death
        self.k, self.knots, self.sn, self.key = k, knots, sn, key
        self.sn_deg, self.sn_idx, self.sn_cb = sn_deg, sn_idx, sn_cb
        self.follow = follow if follow is not None and any(follow) else None

    def __eq__(self, o):
        return isinstance(o, Section) and all(getattr(self, f) == getattr(o, f) for f in self.FIELDS)

    def first_difference(self, o):
        for f in self.FIELDS:
            if getattr(self, f) != getattr(o, f):
                return f
        return None


def vq_row(level, idxs, cb, zero, deg_of_level):
    """SN row (n_sn ints 0..255) of one splat: level 0 -> the zero row; else the sum of its codebook entries over the
    stages, clipped to 0..255, for the first sn_dim(deg) scalars, zero row beyond. Integer-only (decoder core)."""
    row = list(zero)
    if level == 0:
        return row
    stages = cb[level - 1]
    d = sn_dim(deg_of_level[level - 1])
    base = stages[0][idxs[0]]
    for c in range(d):
        v = base[c]
        for s in range(1, len(stages)):
            v += stages[s][idxs[s]][c]
        row[c] = 0 if v < 0 else 255 if v > 255 else v
    return row


class State:
    """what a decoder keeps between sections: the splats that continue, in order (+ the key section's SN codebooks)."""

    def __init__(self):
        self.ids, self.last, self.sn, self.k0, self.next_index = [], [], [], [], None
        self.sn_deg, self.sn_idx, self.sn_cb = [], [], None
        self.vel = []              # per group, per continuing splat: end velocity in knot units per frame (PRED_VEL0)
        self.params = None         # quantisation table the integers above live on (S7: sections may change it)

    @staticmethod
    def after(sec, params, prev=None):
        """prev: the state before `sec` (carries the key section's codebooks through the delta sections)."""
        st = State()
        G = len(params.groups)
        cont = sec.nframes + 1
        dims = [d for _, d in params.groups]
        kidx = {k: i for i, k in enumerate(params.ks)}
        st.last = [[] for _ in range(G)]
        vq = params.sn_vq is not None
        st.vel = [[] for _ in range(G)]
        for i, dc in enumerate(sec.death):
            if dc == cont:
                st.ids.append(sec.ids[i])
                st.sn.append(sec.sn[i])
                st.k0.append(kidx[sec.k[0][i]])
                for g in range(G):
                    row = sec.knots[g][i]
                    st.last[g].append(row[-dims[g]:])
                    st.vel[g].append(end_velocity(row, sec.k[g][i], dims[g], sec.nframes))
                if vq:
                    st.sn_deg.append(sec.sn_deg[i])
                    st.sn_idx.append(sec.sn_idx[i])
        if vq:
            if sec.sn_cb is not None:
                st.sn_cb = sec.sn_cb
            elif prev is not None:
                st.sn_cb = prev.sn_cb
        st.next_index = sec.index + 1
        st.params = params
        return st


# ---------------------------------------------------------------- quantisation change between sections (S7, flag RESCALE)

def _nq_knots(params):
    return sum(d for _, d in params.groups)


def needs_rescale(p_old, p_new):
    """True when the knot quantisation table (offset/scale of P R S O D) of a section differs from the one the previous
    section's integers live on."""
    n = _nq_knots(p_new)
    return p_old.offsets[:n] != p_new.offsets[:n] or p_old.scales[:n] != p_new.scales[:n]


def rescale_value(q, off_o, sc_o, off_n, sc_n):
    """integer q on the old grid -> nearest integer on the new grid. IEEE double arithmetic in exactly this order
    (the same in Python and JS): floor((off_o + q * sc_o - off_n) / sc_n + 0.5); offsets/scales are the float32 header values."""
    return int(math.floor((off_o + q * sc_o - off_n) / sc_n + 0.5))


def rescale_state(st, p_old, p_new):
    """copy of the decoder state with `last` (and the PRED_VEL0 end velocities) converted to the new quantisation table;
    ids/SN/K contexts are unchanged. Velocities are differences: scaled by sc_o/sc_n, offsets cancel."""
    out = State()
    out.ids, out.sn, out.k0, out.next_index = st.ids, st.sn, st.k0, st.next_index
    out.sn_deg, out.sn_idx, out.sn_cb, out.params = st.sn_deg, st.sn_idx, st.sn_cb, p_new
    q0 = 0
    out.last, out.vel = [], []
    for g, (_, d) in enumerate(p_new.groups):
        o_o, s_o = p_old.offsets[q0:q0 + d], p_old.scales[q0:q0 + d]
        o_n, s_n = p_new.offsets[q0:q0 + d], p_new.scales[q0:q0 + d]
        q0 += d
        out.last.append([[rescale_value(r[c], o_o[c], s_o[c], o_n[c], s_n[c]) for c in range(d)] for r in st.last[g]])
        out.vel.append([[int(math.floor(r[c] * s_o[c] / s_n[c] + 0.5)) for c in range(d)] for r in st.vel[g]])
    return out


def end_velocity(row, K, d, nframes):
    """knot step of the last knot interval scaled to ONE frame: round(diff * (K-1) / (nframes-1)); 0 for K = 1 or
    a 1-frame section. Integer-only (floor division, round half up)."""
    if K < 2 or nframes < 2:
        return [0] * d
    den = nframes - 1
    return [_div_round(2 * (row[(K - 1) * d + c] - row[(K - 2) * d + c]) * (K - 1) + den, 2 * den) for c in range(d)]


def _div_round(num, den):
    return num // den


def resample_knot(row, Kn, d, j, Ki):
    """value of a Kn-knot hat curve (flat row) at the frame of knot j of a Ki-knot curve on the same section:
    u = j (Kn-1) / (Ki-1) in knot units, linear interpolation with round half up. Integer-only."""
    if Kn == 1:
        return row[:d]
    if Ki == 1:
        return row[:d]
    num = j * (Kn - 1)
    q, rem = divmod(num, Ki - 1)
    if rem == 0 or q >= Kn - 1:
        q = min(q, Kn - 1)
        return row[q * d:(q + 1) * d]
    a, b = row[q * d:(q + 1) * d], row[(q + 1) * d:(q + 2) * d]
    return [a[c] + (2 * (b[c] - a[c]) * rem + (Ki - 1)) // (2 * (Ki - 1)) for c in range(d)]


def predict_diffs(rows_n, ks_n, d, Ki, k):
    """mean over k neighbour curves of their knot differences resampled to a Ki-knot grid: [(Ki-1)*d] ints (diff-major).
    Fewer than one neighbour -> zeros."""
    out = [0] * ((Ki - 1) * d)
    if not rows_n:
        return out
    n = len(rows_n)
    for row, Kn in zip(rows_n, ks_n):
        prev = resample_knot(row, Kn, d, 0, Ki)
        for j in range(1, Ki):
            cur = resample_knot(row, Kn, d, j, Ki)
            base = (j - 1) * d
            for c in range(d):
                out[base + c] += cur[c] - prev[c]
            prev = cur
    for t in range(len(out)):
        out[t] = (2 * out[t] + n) // (2 * n)
    return out


# ---------------------------------------------------------------- context models (identical on both sides)

class Models:
    def __init__(self, params, nframes):
        G, nk, L = len(params.groups), len(params.ks), params.L
        self.birth = [AdaptiveModel(nframes) for _ in range(nframes)]   # context: previous newborn's birth
        self.death = [AdaptiveModel(L + 2) for _ in range(2)]            # 0 continuing, 1 newborn
        self.ids = AdaptiveModel(NTOK)
        self.k = [[AdaptiveModel(nk) for _ in range(nk + 1)] for _ in range(G)]
        # first knot: [g][mode][dim], mode 0 = residual vs. previous last knot, 1 = DPCM vs. previous absolute splat
        self.first = [[[AdaptiveModel(NTOK) for _ in range(d)] for _ in range(2)] for _, d in params.groups]
        # knot differences: [g][kidx][dim][bucket], bucket 0 = first difference, 1..3 = min(|previous diff|, 2) + 1
        self.diff = [[[[AdaptiveModel(NTOK) for _ in range(4)] for _ in range(d)] for _ in range(nk)]
                     for _, d in params.groups]
        # SN: [scalar c][bucket6 of |residual of scalar c - SN_CH| (same colour channel, previous coefficient)]
        self.sn = [[AdaptiveModel(NTOK) for _ in range(6)] for _ in range(params.n_sn)]
        if params.sn_vq is not None:
            nl = len(params.sn_vq) + 1
            # residual-stage codebook entries: [scalar c][bucket6 of the previous coefficient's entry, same channel]
            self.cb_res = [[AdaptiveModel(NTOK) for _ in range(6)] for _ in range(params.n_sn)]
            self.deg = [AdaptiveModel(nl) for _ in range(nl)]            # context: previous splat's level
            self.idx = [[AdaptiveModel(s, IDX_INC) for s in sizes] for _, sizes in params.sn_vq]   # [level-1][stage]
        # follow mask bits: [kidx of group P][bit of the previous knot]
        self.follow = [[AdaptiveModel(2) for _ in range(2)] for _ in range(nk)]


def _bucket6(v):
    if v < 0:
        v = -v
    return v if v <= 2 else 3 if v <= 4 else 4 if v <= 8 else 5


def _bucket(v):
    if v < 0:
        v = -v
    return 3 if v >= 2 else v + 1


# ---------------------------------------------------------------- follow mode (version 4, flag FOLLOW)

def _follow_den(params):
    """common knot grid: D = max(ks) - 1, every K-1 must divide it (ks 1,2,3,5,9 -> D = 8)."""
    D = max(params.ks) - 1
    if D < 1 or any(k > 1 and D % (k - 1) for k in params.ks):
        raise ValueError('follow mode needs every K-1 to divide max(ks)-1')
    return D


def _at(row, K, u, D, c):
    """D x (value of dim c of a 3-dim knot row with K knots at grid position u in 0..D), exact integer."""
    if K == 1:
        return D * row[c]
    s = D // (K - 1)
    seg, rem = divmod(u, s)
    v = row[seg * 3 + c] * (s - rem)
    if rem:
        v += row[(seg + 1) * 3 + c] * rem
    return v * (D // s)


def _neighbours(cands, grid, q, kmax):
    """kmax nearest candidates (by squared distance of knot-0 positions, ties: lower splat index) of point q."""
    cq = [v // FOLLOW_CELL for v in q]
    found = []
    r = 0
    max_r = grid['span']
    while r <= max_r:
        if (2 * r + 1) ** 3 - (2 * r - 1) ** 3 > len(grid['cells']):     # sparse: exact brute force is cheaper
            found = sorted(((pp[0] - q[0]) ** 2 + (pp[1] - q[1]) ** 2 + (pp[2] - q[2]) ** 2, i) for i, pp in cands.items())
            return [i for _, i in found[:kmax]]
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if max(abs(dx), abs(dy), abs(dz)) != r:
                        continue
                    for i in grid['cells'].get((cq[0] + dx, cq[1] + dy, cq[2] + dz), ()):
                        pp = cands[i]
                        found.append(((pp[0] - q[0]) ** 2 + (pp[1] - q[1]) ** 2 + (pp[2] - q[2]) ** 2, i))
        if len(found) >= kmax:
            found.sort()
            if found[kmax - 1][0] <= (r * FOLLOW_CELL) ** 2:
                return [i for _, i in found[:kmax]]
        r += 1
    found.sort()
    return [i for _, i in found[:kmax]]


def apply_follow(params, sec):
    """returns sec with every masked knot of group 'P' replaced by its prediction (decoder rule, exact integers):
    neighbours = the FOLLOW_K splats with follow mask 0 whose knot-0 position is nearest to the follower's knot 0;
    for masked knot j with anchor a (= nearest unmasked knot below j), knot grid positions u = j*D/(K-1):
      pred_c = x_a,c + floor((S_c + D*n/2) / (D*n)),  S_c = sum over neighbours of (V(u_j) - V(u_a)),
    V = D x the neighbour's piecewise-linear curve at grid position u (_at), n = number of neighbours (no neighbour:
    the follower holds its anchor). Followers never serve as neighbours, so the result does not depend on order."""
    if not sec.follow:
        return sec
    g = [t for t, _ in params.groups].index('P')
    D = _follow_den(params)
    rows = [list(r) for r in sec.knots[g]]
    K = sec.k[g]
    cand_idx = [i for i, m in enumerate(sec.follow) if m == 0]
    cands = {i: rows[i][:3] for i in cand_idx}
    cells = {}
    for i in cand_idx:
        cells.setdefault(tuple(v // FOLLOW_CELL for v in cands[i]), []).append(i)
    if cells:
        lo = [min(c[d] for c in cells) for d in range(3)]; hi = [max(c[d] for c in cells) for d in range(3)]
    grid = {'cells': cells, 'span': 0}
    for i, m in enumerate(sec.follow):
        if not m:
            continue
        Ki = K[i]
        if m & 1 or Ki < 2 or m >> Ki:
            raise ValueError('follow mask: knot 0 or a knot beyond K masked')
        if cells:
            q = [v // FOLLOW_CELL for v in rows[i][:3]]
            grid['span'] = max(max(abs(q[d] - lo[d]), abs(q[d] - hi[d])) for d in range(3))
        nb = _neighbours(cands, grid, rows[i][:3], FOLLOW_K) if cells else []
        st = D // (Ki - 1)
        a = 0
        for j in range(1, Ki):
            if not m >> j & 1:
                a = j
                continue
            for c in range(3):
                if nb:
                    S = sum(_at(rows[n], K[n], j * st, D, c) - _at(rows[n], K[n], a * st, D, c) for n in nb)
                    q = D * len(nb)
                    rows[i][j * 3 + c] = rows[i][a * 3 + c] + (S + q // 2) // q
                else:
                    rows[i][j * 3 + c] = rows[i][a * 3 + c]
    knots = list(sec.knots)
    knots[g] = rows
    return Section(index=sec.index, g0=sec.g0, nframes=sec.nframes, ids=sec.ids, n_cont=sec.n_cont, birth=sec.birth,
                   death=sec.death, k=sec.k, knots=knots, sn=sec.sn, key=sec.key, sn_deg=sec.sn_deg, sn_idx=sec.sn_idx,
                   sn_cb=sec.sn_cb, follow=sec.follow)


def _held(rows, K, follow):
    """what the knot stream carries for followers: masked knots = their anchor value (zero differences)."""
    out = []
    for i, r in enumerate(rows):
        m = follow[i]
        if not m:
            out.append(r); continue
        r = list(r)
        for j in range(1, K[i]):
            if m >> j & 1:
                r[j * 3:j * 3 + 3] = r[(j - 1) * 3:(j - 1) * 3 + 3]
        out.append(r)
    return out


def _with_held(p, sec):
    """sec as the knot streams carry it: group 'P' with held (anchor) values for masked knots; sec itself if no follow."""
    if not sec.follow:
        return sec
    g = [t for t, _ in p.groups].index('P')
    knots = list(sec.knots)
    knots[g] = _held(sec.knots[g], sec.k[g], sec.follow)
    return Section(index=sec.index, g0=sec.g0, nframes=sec.nframes, ids=sec.ids, n_cont=sec.n_cont, birth=sec.birth,
                   death=sec.death, k=sec.k, knots=knots, sn=sec.sn, key=sec.key, sn_deg=sec.sn_deg, sn_idx=sec.sn_idx,
                   sn_cb=sec.sn_cb, follow=sec.follow)

# ---------------------------------------------------------------- header

def _version(follow, pred, vq, rescale=False, bspline=False):
    return VERSION_BSPLINE if bspline else VERSION_RESCALE if rescale else VERSION_FOLLOW if follow else VERSION_PRED if pred else VERSION_SN_VQ if vq else VERSION


def _pack_header(params, sec, key, id_base, sn_pred, stream_lens, rescale=False):
    h = bytearray()
    vq = params.sn_vq is not None
    pred = params.pred is not None
    flags = (FLAG_KEY if key else 0) | (FLAG_SHARED_KNOT if params.shared_knot else 0) | (FLAG_SN_VQ if vq else 0) \
        | (FLAG_PRED if pred else 0) | (FLAG_FOLLOW if sec.follow else 0) | (FLAG_RESCALE if rescale else 0) \
        | (FLAG_BSPLINE if params.bspline else 0)
    ver = _version(bool(sec.follow), pred, vq, rescale, params.bspline)    # highest needed
    h += struct.pack('<4sBBIIHHIII', MAGIC, ver, flags, sec.index, sec.g0, params.L,
                     sec.nframes, sec.n_cont, len(sec.ids) - sec.n_cont, id_base)
    h += struct.pack('<BB', params.key_interval, len(params.groups))
    for t, d in params.groups:
        h += struct.pack('<BB', ord(t), d)
    h += struct.pack('<B', len(params.ks)) + bytes(params.ks)
    if pred:
        h += struct.pack('<B', params.knn_k) + bytes(params.pred)
    h += struct.pack('<H', params.n_sn) + bytes(sn_pred)
    if vq:                                                  # v2 block: zero grid point, levels (degree + stage sizes)
        h += bytes(params.sn_zero) + struct.pack('<B', len(params.sn_vq))
        for deg, sizes in params.sn_vq:
            h += struct.pack('<BB', deg, len(sizes)) + struct.pack(f'<{len(sizes)}H', *sizes)
    nq = len(params.offsets)
    h += struct.pack(f'<{nq}f', *params.offsets) + struct.pack(f'<{nq}f', *params.scales)
    h += struct.pack('<B', len(stream_lens)) + struct.pack(f'<{len(stream_lens)}I', *stream_lens)
    return bytes(h)


def _unpack_header(buf):
    if len(buf) < 4 or buf[:4] != MAGIC:
        raise ValueError('not a .csgs section')
    o = 0
    (_, ver, flags, index, g0, L, nframes, n_cont, n_new, id_base) = struct.unpack_from('<4sBBIIHHIII', buf, o)
    if ver not in (VERSION, VERSION_SN_VQ, VERSION_PRED, VERSION_FOLLOW, VERSION_RESCALE, VERSION_BSPLINE):
        raise ValueError(f'unsupported .csgs version {ver}')
    vq = bool(flags & FLAG_SN_VQ)
    pred = bool(flags & FLAG_PRED)
    follow = bool(flags & FLAG_FOLLOW)
    rescale = bool(flags & FLAG_RESCALE)
    bspline = bool(flags & FLAG_BSPLINE)
    if ver != _version(follow, pred, vq, rescale, bspline):
        raise ValueError('version and flags (SN_VQ / pred / follow / rescale / bspline) disagree')
    o += struct.calcsize('<4sBBIIHHIII')
    key_interval, G = struct.unpack_from('<BB', buf, o); o += 2
    groups = []
    for _ in range(G):
        t, d = struct.unpack_from('<BB', buf, o); o += 2
        groups.append((chr(t), d))
    nk = buf[o]; o += 1
    ks = list(buf[o:o + nk]); o += nk
    knn_k, modes = 0, None
    if pred:
        knn_k = buf[o]; o += 1
        modes = list(buf[o:o + G]); o += G
    (n_sn,) = struct.unpack_from('<H', buf, o); o += 2
    sn_pred = list(buf[o:o + n_sn]); o += n_sn
    sn_vq = sn_zero = None
    if vq:
        sn_zero = list(buf[o:o + n_sn]); o += n_sn
        nl = buf[o]; o += 1
        sn_vq = []
        for _ in range(nl):
            deg, ns = struct.unpack_from('<BB', buf, o); o += 2
            sizes = list(struct.unpack_from(f'<{ns}H', buf, o)); o += 2 * ns
            sn_vq.append((deg, sizes))
    nq = sum(d for _, d in groups) + n_sn
    offsets = list(struct.unpack_from(f'<{nq}f', buf, o)); o += 4 * nq
    scales = list(struct.unpack_from(f'<{nq}f', buf, o)); o += 4 * nq
    ns = buf[o]; o += 1
    lens = list(struct.unpack_from(f'<{ns}I', buf, o)); o += 4 * ns
    params = Params(L, ks, groups, n_sn, offsets, scales, bool(flags & FLAG_SHARED_KNOT), key_interval, sn_vq, sn_zero,
                    modes, knn_k, bspline)
    return params, dict(key=bool(flags & FLAG_KEY), follow=follow, rescale=rescale, index=index, g0=g0, nframes=nframes, n_cont=n_cont,
                        n_new=n_new, id_base=id_base, sn_pred=sn_pred), lens, o


# ---------------------------------------------------------------- encoder

class Encoder:
    """sequential section encoder; section i is a key section when i % key_interval == 0 (no reference)."""

    def __init__(self, params):
        self.params = params
        self.state = None

    def encode(self, sec, key=None, collect=None):
        p = self.params
        if key is None:
            key = sec.index % p.key_interval == 0 or self.state is None
        st = None if key else self.state
        if not key and (st is None or st.next_index != sec.index):
            raise ValueError('non-key section needs the previous section')
        out = encode_section(p, sec, key, st, collect)
        self.state = State.after(sec, p, self.state)
        return out


def _check(p, sec, key, st):
    n, nc, G = len(sec.ids), sec.n_cont, len(p.groups)
    nn = n - nc
    if not (0 <= nc <= n) or len(sec.birth) != nn or len(sec.death) != n or len(sec.sn) != n:
        raise ValueError('inconsistent section lengths')
    if not (1 <= sec.nframes <= p.L):
        raise ValueError('nframes out of range')
    if len(sec.k) != G or len(sec.knots) != G or any(len(sec.k[g]) != n or len(sec.knots[g]) != n for g in range(G)):
        raise ValueError('k/knots need one entry per group and splat')
    for i in range(1, n):
        if sec.ids[i] <= sec.ids[i - 1]:
            raise ValueError('ids must be strictly increasing')
    for i in range(nn):
        if sec.ids[nc + i] != sec.ids[nc] + i:
            raise ValueError('newborn ids must be consecutive')
        if not (0 <= sec.birth[i] < sec.nframes) or not (sec.birth[i] < sec.death[nc + i]):
            raise ValueError('bad birth')
    for dc in sec.death:
        if not 1 <= dc <= sec.nframes + 1:
            raise ValueError('bad death code')
    ks = set(p.ks)
    for g, (_, d) in enumerate(p.groups):
        for i in range(n):
            if sec.k[g][i] not in ks or len(sec.knots[g][i]) != sec.k[g][i] * d:
                raise ValueError('K not in params.ks or wrong knot count')
            for v in sec.knots[g][i]:
                if not KNOT_MIN <= v <= KNOT_MAX:
                    raise ValueError('knot value outside [-2**30, 2**30)')
    for row in sec.sn:
        if len(row) != p.n_sn or any(not 0 <= v <= 255 for v in row):
            raise ValueError('sn must be n_sn values in 0..255')
    if sec.follow:
        if len(sec.follow) != n:
            raise ValueError('follow needs one mask per splat')
        if apply_follow(p, sec).knots != sec.knots:
            raise ValueError('follow knots must equal the neighbour prediction (call apply_follow first)')
    if p.sn_vq is not None:
        _check_vq(p, sec, key, st)
    elif sec.sn_deg is not None or sec.sn_idx is not None or sec.sn_cb is not None:
        raise ValueError('sn_deg/sn_idx/sn_cb need params.sn_vq')
    if st is not None:
        if nc != len(st.ids) or sec.ids[:nc] != st.ids:
            raise ValueError('continuing splats differ from the previous section')
        for i in range(nc):
            if sec.sn[i] != st.sn[i]:
                raise ValueError('SN of a continuing splat changed')
        if p.sn_vq is not None:
            for i in range(nc):
                if sec.sn_deg[i] != st.sn_deg[i] or sec.sn_idx[i] != st.sn_idx[i]:
                    raise ValueError('SN degree/index of a continuing splat changed')
        if p.shared_knot:
            for g, (_, d) in enumerate(p.groups):
                for i in range(nc):
                    if sec.knots[g][i][:d] != st.last[g][i]:
                        raise ValueError('shared knot differs from the previous last knot')


def _check_vq(p, sec, key, st):
    n = len(sec.ids)
    degs = [d for d, _ in p.sn_vq]
    level_of = {d: i + 1 for i, d in enumerate(degs)}
    if sec.sn_deg is None or sec.sn_idx is None or len(sec.sn_deg) != n or len(sec.sn_idx) != n:
        raise ValueError('SN-VQ sections need sn_deg and sn_idx per splat')
    if key:
        cb = sec.sn_cb
        if cb is None or len(cb) != len(p.sn_vq):
            raise ValueError('key section needs sn_cb with one entry list per level')
        for li, (deg, sizes) in enumerate(p.sn_vq):
            if len(cb[li]) != len(sizes):
                raise ValueError('sn_cb stages differ from params.sn_vq')
            for s, (size, entries) in enumerate(zip(sizes, cb[li])):
                if len(entries) != size:
                    raise ValueError('codebook size differs from params.sn_vq')
                lo, hi = (0, 255) if s == 0 else (-255, 255)
                for e in entries:
                    if len(e) != sn_dim(deg) or any(not lo <= v <= hi for v in e):
                        raise ValueError('codebook entry: sn_dim(deg) values, 0..255 (stage 0) or -255..255 (residual)')
    else:
        if sec.sn_cb is not None:
            raise ValueError('codebooks are sent in key sections only')
        cb = st.sn_cb
        if cb is None:
            raise ValueError('delta section without codebooks in the state')
    for i in range(n):
        d = sec.sn_deg[i]
        if d != 0 and d not in level_of:
            raise ValueError('sn_deg must be 0 or one of the params.sn_vq degrees')
        lv = level_of.get(d, 0)
        idxs = sec.sn_idx[i]
        if lv == 0:
            if idxs:
                raise ValueError('degree 0 has no indices')
        else:
            sizes = p.sn_vq[lv - 1][1]
            if len(idxs) != len(sizes) or any(not 0 <= x < s for x, s in zip(idxs, sizes)):
                raise ValueError('sn_idx: one index per stage, below the codebook size')
        if sec.sn[i] != vq_row(lv, idxs, cb, p.sn_zero, degs):
            raise ValueError('sn row differs from its VQ reconstruction')


def encode_section(p, sec, key, st, collect=None):
    """bytes of one section. collect (optional dict) receives the coded integer symbols per stream name."""
    rescale = False
    if st is not None and not key and st.params is not None:
        n_k = _nq_knots(p)
        if st.params.offsets[n_k:] != p.offsets[n_k:] or st.params.scales[n_k:] != p.scales[n_k:]:
            raise ValueError('the SN quantisation table must not change inside a key interval')
        if needs_rescale(st.params, p):
            rescale = True
            st = rescale_state(st, st.params, p)
    _check(p, sec, key, st)
    n, nc, G = len(sec.ids), sec.n_cont, len(p.groups)
    nn = n - nc
    m = Models(p, sec.nframes)
    kidx = {k: i for i, k in enumerate(p.ks)}
    nk = len(p.ks)
    names = stream_names(p, bool(sec.follow))
    sym = {nm: [] for nm in names} if collect is not None else None
    ksec = _with_held(p, sec)                              # follow mode: the knot streams carry held anchor values
    e = RansEncoder()                                      # birth
    pb = 0
    for b in sec.birth:
        e.put(m.birth[pb], b)
        pb = b
    streams = [e.finish()]
    if sym is not None:
        sym['birth'] += sec.birth

    e = RansEncoder()                                      # death
    for i, dc in enumerate(sec.death):
        e.put(m.death[0 if i < nc else 1], dc)
    streams.append(e.finish())
    if sym is not None:
        sym['death'] += sec.death

    e = RansEncoder()                                      # ids of continuing splats (key sections only)
    if key:
        prev = 0
        for i in range(nc):
            e.put_uint(m.ids, sec.ids[i] - prev)
            if sym is not None:
                sym['ids'].append(sec.ids[i] - prev)
            prev = sec.ids[i]
    streams.append(e.finish())

    e = RansEncoder()                                      # K choice
    ki = [[kidx[k] for k in sec.k[g]] for g in range(G)]
    for i in range(n):
        c0 = st.k0[i] if (not key and i < nc) else nk
        e.put(m.k[0][c0], ki[0][i])
        for g in range(1, G):
            e.put(m.k[g][ki[0][i]], ki[g][i])
    streams.append(e.finish())
    if sym is not None:
        sym['k'] += [ki[g][i] for i in range(n) for g in range(G)]

    knn = [None]                                           # causal neighbour lists (spatial prediction), built once
    for g, (tag, d) in enumerate(p.groups):                # knots per attribute group
        e = RansEncoder()
        s = sym[f'knots_{tag}'] if sym is not None else None
        if p.pred is not None:
            streams.append(_encode_knots_pred(p, ksec, key, st, m, g, ki, knn, s))
            continue
        first_res, first_abs, mdiff = m.first[g][0], m.first[g][1], m.diff[g]
        prev_abs = [0] * d
        for i in range(n):
            row = ksec.knots[g][i]
            K = sec.k[g][i]
            if not key and i < nc:
                if not p.shared_knot:
                    ref = st.last[g][i]
                    for c in range(d):
                        e.put_sint(first_res[c], row[c] - ref[c])
                        if s is not None:
                            s.append(row[c] - ref[c])
            else:
                for c in range(d):
                    e.put_sint(first_abs[c], row[c] - prev_abs[c])
                    if s is not None:
                        s.append(row[c] - prev_abs[c])
                prev_abs = row[:d]
            mk = mdiff[ki[g][i]]
            for c in range(d):
                mc = mk[c]
                pd = None
                for j in range(1, K):
                    v = row[j * d + c] - row[(j - 1) * d + c]
                    e.put_sint(mc[0 if pd is None else _bucket(pd)], v)
                    if s is not None:
                        s.append(v)
                    pd = v
        streams.append(e.finish())

    e = RansEncoder()                                      # SN (newborns; every splat in key sections)
    s = sym['sn'] if sym is not None else None
    msn = m.sn
    coded = range(0 if key else nc, n)
    vq = p.sn_vq is not None
    if vq:
        # v2: the sn stream stays empty; sn_pred predicts the stage-0 codebook entries (median per scalar)
        base_rows = [e_ for lvl in sec.sn_cb for e_ in lvl[0]] if key else []
        sn_pred = [sorted(r[c] for r in base_rows if c < len(r))[sum(1 for r in base_rows if c < len(r)) // 2]
                   if any(c < len(r) for r in base_rows) else 128 for c in range(p.n_sn)]
    else:
        sn_pred = [sorted(sec.sn[i][c] for i in coded)[len(coded) // 2] if len(coded) else 128 for c in range(p.n_sn)]
        for i in coded:
            row = sec.sn[i]
            res = [0] * p.n_sn
            for c in range(p.n_sn):
                v = row[c] - sn_pred[c]
                res[c] = v
                e.put_sint(msn[c][_bucket6(res[c - SN_CH]) if c >= SN_CH else 0], v)
                if s is not None:
                    s.append(v)
    streams.append(e.finish())

    if vq:
        degs = [d for d, _ in p.sn_vq]
        level_of = {d: i + 1 for i, d in enumerate(degs)}
        e = RansEncoder()                                  # sn_cb: codebooks (key sections only)
        s = sym['sn_cb'] if sym is not None else None
        if key:
            for lvl in sec.sn_cb:
                for si, entries in enumerate(lvl):
                    for row in entries:
                        res = [0] * len(row)
                        for c in range(len(row)):
                            v = row[c] - sn_pred[c] if si == 0 else row[c]
                            res[c] = v
                            mm = (msn if si == 0 else m.cb_res)[c][_bucket6(res[c - SN_CH]) if c >= SN_CH else 0]
                            e.put_sint(mm, v)
                            if s is not None:
                                s.append(v)
        streams.append(e.finish())
        e = RansEncoder()                                  # sn_deg: level per coded splat, context previous level
        s = sym['sn_deg'] if sym is not None else None
        prev_lv = 0
        levels = [level_of.get(sec.sn_deg[i], 0) for i in range(n)]
        for i in coded:
            e.put(m.deg[prev_lv], levels[i])
            if s is not None:
                s.append(levels[i])
            prev_lv = levels[i]
        streams.append(e.finish())
        e = RansEncoder()                                  # sn_idx: one index per stage, model per (level, stage)
        s = sym['sn_idx'] if sym is not None else None
        for i in coded:
            lv = levels[i]
            if lv:
                for si, x in enumerate(sec.sn_idx[i]):
                    e.put(m.idx[lv - 1][si], x)
                    if s is not None:
                        s.append(x)
        streams.append(e.finish())

    if sec.follow:                                         # follow mask bits of group P (K >= 2 only), last stream
        gP = [t for t, _ in p.groups].index('P')
        e = RansEncoder()
        for i in range(n):
            pb = 0
            for j in range(1, sec.k[gP][i]):
                b = sec.follow[i] >> j & 1
                e.put(m.follow[ki[gP][i]][pb], b)
                pb = b
        streams.append(e.finish())
        if sym is not None:
            sym['follow'] = list(sec.follow)

    if collect is not None:
        for k, v in sym.items():
            collect.setdefault(k, []).extend(v)
    id_base = sec.ids[nc] if nn else 0
    head = _pack_header(p, sec, key, id_base, sn_pred, [len(x) for x in streams], rescale)
    body = head + b''.join(streams)
    return body + struct.pack('<I', zlib.crc32(body) & 0xffffffff)


# ---------------------------------------------------------------- motion-prediction mode (v3, FLAG_PRED)

def _knn_of(p, rows_P, knn):
    """causal k nearest neighbours by knot 0 of group 'P' (built once per section, shared by all groups)."""
    if knn[0] is None:
        from .knn import causal_knn
        knn[0] = causal_knn([r[:3] for r in rows_P], p.knn_k)
    return knn[0]


def _first_ref(p, st, g, i, d, mode):
    ref = st.last[g][i]
    if mode & PRED_VEL0:
        v = st.vel[g][i]
        return [ref[c] + v[c] for c in range(d)]
    return ref


def _residual(v, prev_v, pred, j, c, d, mode):
    """coded residual of knot difference v (knot j, dim c) under the group's mode; prev_v = actual previous difference."""
    r = v
    if pred is not None:
        if (mode & PRED_TEMPORAL) and j >= 2:
            r = (v - prev_v) - (pred[(j - 1) * d + c] - pred[(j - 2) * d + c])
        else:
            r = v - pred[(j - 1) * d + c]
    elif (mode & PRED_TEMPORAL) and j >= 2:
        r = v - prev_v
    return r


def _encode_knots_pred(p, sec, key, st, m, g, ki, knn, s):
    tag, d = p.groups[g]
    mode = p.pred[g]
    n, nc = len(sec.ids), sec.n_cont
    e = RansEncoder()
    first_res, first_abs, mdiff = m.first[g][0], m.first[g][1], m.diff[g]
    rows, ks = sec.knots[g], sec.k[g]
    prev_abs = [0] * d
    for i in range(n):                                     # phase 1: knot 0 of every splat
        row = rows[i]
        if not key and i < nc:
            if not p.shared_knot:
                ref = _first_ref(p, st, g, i, d, mode)
                for c in range(d):
                    e.put_sint(first_res[c], row[c] - ref[c])
                    if s is not None:
                        s.append(row[c] - ref[c])
        else:
            for c in range(d):
                e.put_sint(first_abs[c], row[c] - prev_abs[c])
                if s is not None:
                    s.append(row[c] - prev_abs[c])
            prev_abs = row[:d]
    nb = _knn_of(p, sec.knots[p.groups.index(('P', 3))], knn) if mode & PRED_SPATIAL else None
    for i in range(n):                                     # phase 2: knot differences
        row, K = rows[i], ks[i]
        pred = None
        if nb is not None and K > 1:
            js = [j for j in nb[i] if j >= 0]
            pred = predict_diffs([rows[j] for j in js], [ks[j] for j in js], d, K, len(js)) if js else None
        mk = mdiff[ki[g][i]]
        for c in range(d):
            mc = mk[c]
            pr = None
            prev_v = 0
            for j in range(1, K):
                v = row[j * d + c] - row[(j - 1) * d + c]
                r = _residual(v, prev_v, pred, j, c, d, mode)
                e.put_sint(mc[0 if pr is None else _bucket(pr)], r)
                if s is not None:
                    s.append(r)
                pr = r
                prev_v = v
    return e.finish()


def _decode_knots_pred(p, dec, key, st, m, g, ki, k, nc, n, knots, knn):
    tag, dm = p.groups[g]
    mode = p.pred[g]
    first_res, first_abs, mdiff = m.first[g][0], m.first[g][1], m.diff[g]
    prev_abs = [0] * dm
    rows = []
    for i in range(n):                                     # phase 1
        K = k[g][i]
        row = [0] * (K * dm)
        if not key and i < nc:
            ref = _first_ref(p, st, g, i, dm, mode) if not p.shared_knot else st.last[g][i]
            if p.shared_knot:
                for c in range(dm):
                    row[c] = ref[c]
            else:
                for c in range(dm):
                    row[c] = ref[c] + dec.get_sint(first_res[c])
        else:
            for c in range(dm):
                row[c] = prev_abs[c] + dec.get_sint(first_abs[c])
            prev_abs = row[:dm]
        rows.append(row)
    if mode & PRED_SPATIAL:
        gp = p.groups.index(('P', 3))
        nb = _knn_of(p, rows if gp == g else knots[gp], knn)
    else:
        nb = None
    for i in range(n):                                     # phase 2
        row, K = rows[i], k[g][i]
        pred = None
        if nb is not None and K > 1:
            js = [j for j in nb[i] if j >= 0]
            pred = predict_diffs([rows[j] for j in js], [k[g][j] for j in js], dm, K, len(js)) if js else None
        mk = mdiff[ki[g][i]]
        for c in range(dm):
            mc = mk[c]
            pr = None
            prev_v = 0
            for j in range(1, K):
                r = dec.get_sint(mc[0 if pr is None else _bucket(pr)])
                v = r
                if pred is not None:
                    if (mode & PRED_TEMPORAL) and j >= 2:
                        v = r + prev_v + (pred[(j - 1) * dm + c] - pred[(j - 2) * dm + c])
                    else:
                        v = r + pred[(j - 1) * dm + c]
                elif (mode & PRED_TEMPORAL) and j >= 2:
                    v = r + prev_v
                row[j * dm + c] = row[(j - 1) * dm + c] + v
                pr = r
                prev_v = v
    return rows


def stream_names(p, follow=False):
    names = ['birth', 'death', 'ids', 'k'] + [f'knots_{t}' for t, _ in p.groups] + ['sn']
    if p.sn_vq is not None:
        names += ['sn_cb', 'sn_deg', 'sn_idx']
    if follow:
        names.append('follow')                             # always the last stream
    return names


def stream_sizes(buf):
    """{stream name: bytes} plus 'header' (incl. CRC) of an encoded section."""
    p, h, lens, o = _unpack_header(buf)
    out = dict(zip(stream_names(p, h['follow']), lens))
    out['header'] = o + 4
    return out


# ---------------------------------------------------------------- decoder

class Decoder:
    """sequential decoder; may start at any key section (random access)."""

    def __init__(self):
        self.state = None
        self.params = None

    def decode(self, buf):
        p, h, _, _ = _unpack_header(buf)
        st = None
        if not h['key']:
            if self.state is None or self.state.next_index != h['index']:
                raise ValueError('non-key section without its previous section (seek to a key section)')
            st = self.state
        sec = decode_section(buf, st)
        self.params = p
        self.state = State.after(sec, p, self.state)
        return sec


def decode_section(buf, st):
    if len(buf) < 8 or (zlib.crc32(buf[:-4]) & 0xffffffff) != struct.unpack_from('<I', buf, len(buf) - 4)[0]:
        raise ValueError('CRC mismatch')
    p, h, lens, o = _unpack_header(buf)
    key, nc, nn, nf = h['key'], h['n_cont'], h['n_new'], h['nframes']
    vq = p.sn_vq is not None
    if len(lens) != len(stream_names(p, h['follow'])) or o + sum(lens) + 4 != len(buf):
        raise ValueError('stream table does not match the section size')
    if not key and nc != len(st.ids):
        raise ValueError('n_cont differs from the decoder state')
    if key and h['rescale']:
        raise ValueError('RESCALE flag on a key section')
    if not key and st.params is not None:                  # quantisation change (S7): the state moves to this section's table
        n_k = _nq_knots(p)
        if st.params.offsets[n_k:] != p.offsets[n_k:] or st.params.scales[n_k:] != p.scales[n_k:]:
            raise ValueError('the SN quantisation table changed inside a key interval')
        if needs_rescale(st.params, p) != h['rescale']:
            raise ValueError('quantisation table change and the RESCALE flag disagree')
        if h['rescale']:
            st = rescale_state(st, st.params, p)
    n, G, nk = nc + nn, len(p.groups), len(p.ks)
    m = Models(p, nf)
    starts = []
    for ln in lens:
        starts.append((o, o + ln)); o += ln

    def dec(i):
        return RansDecoder(buf, starts[i][0], starts[i][1])

    def end(d):
        if not (d.done() or (d.x == 0 and d.pos == d.end)):
            raise ValueError('stream not consumed exactly')

    d = dec(ST_BIRTH)
    birth, pb = [], 0
    for _ in range(nn):
        pb = d.get(m.birth[pb])
        birth.append(pb)
    end(d)
    d = dec(ST_DEATH)
    death = [d.get(m.death[0 if i < nc else 1]) for i in range(n)]
    end(d)
    d = dec(ST_IDS)
    if key:
        ids, prev = [], 0
        for _ in range(nc):
            prev += d.get_uint(m.ids)
            ids.append(prev)
    else:
        ids = list(st.ids)
    end(d)
    ids += [h['id_base'] + i for i in range(nn)]

    d = dec(ST_K)
    ki = [[0] * n for _ in range(G)]
    for i in range(n):
        c0 = st.k0[i] if (not key and i < nc) else nk
        k0 = d.get(m.k[0][c0])
        ki[0][i] = k0
        for g in range(1, G):
            ki[g][i] = d.get(m.k[g][k0])
    end(d)
    k = [[p.ks[x] for x in ki[g]] for g in range(G)]

    knots = []
    knn = [None]
    for g, (_, dm) in enumerate(p.groups):
        d = dec(ST_K + 1 + g)
        if p.pred is not None:
            knots.append(_decode_knots_pred(p, d, key, st, m, g, ki, k, nc, n, knots, knn))
            end(d)
            continue
        first_res, first_abs, mdiff = m.first[g][0], m.first[g][1], m.diff[g]
        prev_abs = [0] * dm
        rows = []
        for i in range(n):
            K = k[g][i]
            row = [0] * (K * dm)
            if not key and i < nc:
                ref = st.last[g][i]
                if p.shared_knot:
                    for c in range(dm):
                        row[c] = ref[c]
                else:
                    for c in range(dm):
                        row[c] = ref[c] + d.get_sint(first_res[c])
            else:
                for c in range(dm):
                    row[c] = prev_abs[c] + d.get_sint(first_abs[c])
                prev_abs = row[:dm]
            mk = mdiff[ki[g][i]]
            for c in range(dm):
                mc = mk[c]
                pd = None
                for j in range(1, K):
                    v = d.get_sint(mc[0 if pd is None else _bucket(pd)])
                    row[j * dm + c] = row[(j - 1) * dm + c] + v
                    pd = v
            rows.append(row)
        end(d)
        knots.append(rows)

    d = dec(ST_K + 1 + G)
    sn = [] if key else [list(x) for x in st.sn]
    msn, sn_pred = m.sn, h['sn_pred']
    sn_deg = sn_idx = sn_cb = None
    if not vq:
        for _ in range(n if key else nn):
            row = [0] * p.n_sn
            res = [0] * p.n_sn
            for c in range(p.n_sn):
                v = d.get_sint(msn[c][_bucket6(res[c - SN_CH]) if c >= SN_CH else 0])
                res[c] = v
                row[c] = sn_pred[c] + v
                if not 0 <= row[c] <= 255:
                    raise ValueError('SN value out of range')
            sn.append(row)
        end(d)
    else:
        end(d)                                             # sn stream is empty in SN-VQ sections
        degs = [dg for dg, _ in p.sn_vq]
        d = dec(ST_K + 2 + G)                              # sn_cb
        if key:
            sn_cb = []
            for deg, sizes in p.sn_vq:
                dim, lvl = sn_dim(deg), []
                for si, size in enumerate(sizes):
                    entries = []
                    for _ in range(size):
                        row = [0] * dim
                        res = [0] * dim
                        for c in range(dim):
                            mm = (msn if si == 0 else m.cb_res)[c][_bucket6(res[c - SN_CH]) if c >= SN_CH else 0]
                            v = d.get_sint(mm)
                            res[c] = v
                            row[c] = sn_pred[c] + v if si == 0 else v
                            if (si == 0 and not 0 <= row[c] <= 255) or (si > 0 and not -255 <= v <= 255):
                                raise ValueError('codebook entry out of range')
                        entries.append(row)
                    lvl.append(entries)
                sn_cb.append(lvl)
            cb = sn_cb
        else:
            cb = st.sn_cb
            if cb is None:
                raise ValueError('delta section without codebooks in the decoder state')
        end(d)
        d = dec(ST_K + 3 + G)                              # sn_deg
        levels = [] if key else [degs.index(x) + 1 if x else 0 for x in st.sn_deg]
        prev_lv = 0
        for _ in range(n if key else nn):
            lv = d.get(m.deg[prev_lv])
            levels.append(lv)
            prev_lv = lv
        end(d)
        d = dec(ST_K + 4 + G)                              # sn_idx
        sn_deg = [degs[lv - 1] if lv else 0 for lv in levels]
        sn_idx = [] if key else [list(x) for x in st.sn_idx]
        for i in range(0 if key else nc, n):
            lv = levels[i]
            idxs = [d.get(m.idx[lv - 1][si]) for si in range(len(p.sn_vq[lv - 1][1]))] if lv else []
            sn_idx.append(idxs)
            sn.append(vq_row(lv, idxs, cb, p.sn_zero, degs))
        end(d)
    sec = Section(index=h['index'], g0=h['g0'], nframes=nf, ids=ids, n_cont=nc, birth=birth, death=death,
                  k=k, knots=knots, sn=sn, key=key, sn_deg=sn_deg, sn_idx=sn_idx, sn_cb=sn_cb)
    if h['follow']:
        gP = [t for t, _ in p.groups].index('P')
        d = dec(len(lens) - 1)                             # follow masks: the last stream
        follow = []
        for i in range(n):
            mask, pb = 0, 0
            for j in range(1, k[gP][i]):
                pb = d.get(m.follow[ki[gP][i]][pb])
                mask |= pb << j
            follow.append(mask)
        end(d)
        sec.follow = follow if any(follow) else None
        sec = apply_follow(p, sec)
    return sec
