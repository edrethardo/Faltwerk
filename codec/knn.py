"""Deterministic spatial helpers of the .csgs motion-prediction mode (S2, ). Decoder core: pure Python ints,
no numpy, no floats, so the TypeScript port reproduces the neighbour sets bit-exactly.

  causal_knn(pos, k)   pos: list of [x, y, z] ints (knot 0 of group P of the section, in section order).
                       Returns per splat i the indices of the k nearest splats among 0..i-1 (ordered by squared
                       Euclidean distance, ties by smaller index), padded with -1 when fewer exist. Uniform grid hash,
                       incremental insertion (point i is inserted after its own query), ring search with the exact
                       stopping rule "k-th best squared distance <= (r * cell)^2 after ring r".
  morton_order(pos)    permutation (list of indices) sorting the points along a 3-D Morton (Z-order) curve, ties by index.
  grid_cell(...)       cell size rule shared by both sides: cell = max(1, round(cbrt(volume * 8 / n))) via integer cube root.
"""


def _icbrt(v):
    """floor cube root of a non-negative int (integer Newton)."""
    if v <= 0:
        return 0
    x = 1 << ((v.bit_length() + 2) // 3)
    while True:
        y = (2 * x + v // (x * x)) // 3
        if y >= x:
            break
        x = y
    while x * x * x > v:
        x -= 1
    while (x + 1) ** 3 <= v:
        x += 1
    return x


def grid_cell(lo, hi, n, pos=None):
    """cell edge: start from the bounding-box rule cbrt(volume * 8 / n) and, when `pos` is given, halve it while the
    mean occupancy of the NON-EMPTY cells exceeds OCC (splats live on thin surfaces, so the bbox volume overestimates).
    Both sides compute this from the same integer positions, so it is deterministic."""
    ext = [max(1, hi[c] - lo[c] + 1) for c in range(3)]
    vol = ext[0] * ext[1] * ext[2]
    cell = max(1, _icbrt(vol * 8 // max(1, n)))
    if pos is not None:
        while cell > 1:
            occ = len({((p[0] - lo[0]) // cell, (p[1] - lo[1]) // cell, (p[2] - lo[2]) // cell) for p in pos})
            if n <= OCC * occ:
                break
            cell = max(1, cell // 2)
    return cell


OCC = 4


def causal_knn(pos, k):
    n = len(pos)
    out = [[-1] * k for _ in range(n)]
    if n == 0 or k == 0:
        return out
    lo = [min(p[c] for p in pos) for c in range(3)]
    hi = [max(p[c] for p in pos) for c in range(3)]
    cell = grid_cell(lo, hi, n, pos)
    cells = {}
    for i in range(n):
        p = pos[i]
        cx, cy, cz = (p[0] - lo[0]) // cell, (p[1] - lo[1]) // cell, (p[2] - lo[2]) // cell
        if i > 0:
            best = []                                           # list of (d2, j), kept sorted, len <= k
            r = 0
            while True:
                # cells of Chebyshev ring r around (cx, cy, cz)
                for dx in range(-r, r + 1):
                    ax = cx + dx
                    if ax < 0:
                        continue
                    for dy in range(-r, r + 1):
                        ay = cy + dy
                        if ay < 0:
                            continue
                        edge_xy = abs(dx) == r or abs(dy) == r
                        for dz in (range(-r, r + 1) if edge_xy else ((-r, r) if r else (0,))):
                            az = cz + dz
                            if az < 0:
                                continue
                            lst = cells.get((ax, ay, az))
                            if not lst:
                                continue
                            for j in lst:
                                q = pos[j]
                                d2 = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2
                                if len(best) < k or (d2, j) < best[-1]:
                                    # insert keeping order
                                    pos_ = len(best)
                                    while pos_ > 0 and (d2, j) < best[pos_ - 1]:
                                        pos_ -= 1
                                    best.insert(pos_, (d2, j))
                                    if len(best) > k:
                                        best.pop()
                if len(best) == k and best[-1][0] <= (r * cell) ** 2:
                    break
                if len(best) == min(k, i):                     # all decoded points seen? only when ring covers the grid
                    if r * cell > max(hi[c] - lo[c] for c in range(3)):
                        break
                r += 1
                if (2 * r + 1) ** 3 > 8 * len(cells):          # sparse / outlier case: scanning cells costs more than
                    best = []                                  # scanning all inserted points -> exact brute force
                    for lst in cells.values():
                        for j in lst:
                            q = pos[j]
                            d2 = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2
                            if len(best) < k or (d2, j) < best[-1]:
                                pos_ = len(best)
                                while pos_ > 0 and (d2, j) < best[pos_ - 1]:
                                    pos_ -= 1
                                best.insert(pos_, (d2, j))
                                if len(best) > k:
                                    best.pop()
                    break
            for t, (_, j) in enumerate(best):
                out[i][t] = j
        cells.setdefault((cx, cy, cz), []).append(i)
    return out


def _spread(v):
    """spread the low 21 bits of v so that there are two zero bits between each (Morton)."""
    v &= (1 << 21) - 1
    v = (v | (v << 32)) & 0x1f00000000ffff
    v = (v | (v << 16)) & 0x1f0000ff0000ff
    v = (v | (v << 8)) & 0x100f00f00f00f00f
    v = (v | (v << 4)) & 0x10c30c30c30c30c3
    v = (v | (v << 2)) & 0x1249249249249249
    return v


def morton_key(p, lo, shift):
    return (_spread((p[0] - lo[0]) >> shift) | (_spread((p[1] - lo[1]) >> shift) << 1)
            | (_spread((p[2] - lo[2]) >> shift) << 2))


def morton_order(pos):
    n = len(pos)
    if n == 0:
        return []
    lo = [min(int(p[c]) for p in pos) for c in range(3)]
    hi = [max(int(p[c]) for p in pos) for c in range(3)]
    span = max(hi[c] - lo[c] for c in range(3))
    shift = max(0, span.bit_length() - 21)
    keys = [(morton_key([int(v) for v in p], lo, shift), i) for i, p in enumerate(pos)]
    keys.sort()
    return [i for _, i in keys]
