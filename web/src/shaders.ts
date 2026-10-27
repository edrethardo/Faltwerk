// WGSL for the CurveStream-GS web decoder (paper §4.5): curve evaluation -> projection + SH -> bitonic depth sort ->
// instanced alpha-blended quads (front to back) -> blit. Projection/SH/alpha follow gsplat 1.x conventions
// (pinhole, eps2d 0.3, 3-sigma radius, alpha = min(0.999, o*exp(-sigma)), skip alpha < 1/255, SH degree 3, +0.5).

export const EVAL_WGSL = /* wgsl */ `
struct EvalU { t: f32, g0: f32, span: f32, n: u32 };
@group(0) @binding(0) var<uniform> U: EvalU;
@group(0) @binding(1) var<storage, read> hdr: array<u32>;
@group(0) @binding(2) var<storage, read> knots: array<f32>;
@group(0) @binding(3) var<storage, read_write> world: array<vec4f>;

var<private> BSP: bool = false;                  // v6 (hdr word 2 bit 31): clamped cubic B-spline instead of the hat basis
struct BW { s0: u32, p: u32, w: vec4f };
fn knv(i: i32, p: i32, K: i32) -> f32 { return clamp(f32(i - p) / f32(K - p), 0.0, 1.0); }
fn bspl(K: u32) -> BW {                         // local support (NURBS book A2.2), = web/src/bspline.ts bsplineLocal
  let p = i32(min(3u, K - 1u));
  let Ki = i32(K);
  let u = clamp((U.t - U.g0) / U.span, 0.0, 1.0);
  var s = Ki - 1;
  if (u < 1.0) { s = min(Ki - 1, p + i32(floor(u * f32(Ki - p)))); }
  var N = array<f32, 4>(1.0, 0.0, 0.0, 0.0);
  var le = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  var ri = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  for (var j = 1; j <= p; j++) {
    le[j] = u - knv(s + 1 - j, p, Ki);
    ri[j] = knv(s + j, p, Ki) - u;
    var saved = 0.0;
    for (var r = 0; r < j; r++) {
      let temp = N[r] / (ri[r + 1] + le[j - r]);
      N[r] = saved + ri[r + 1] * temp;
      saved = le[j - r] * temp;
    }
    N[j] = saved;
  }
  return BW(u32(s - p), u32(p), vec4f(N[0], N[1], N[2], N[3]));
}
fn seg(K: u32) -> vec2f {                       // (segment j, weight of knot j+1)
  if (K <= 1u) { return vec2f(0.0, 0.0); }
  let km1 = f32(K - 1u);
  let tau = clamp((U.t - U.g0) * km1 / U.span, 0.0, km1);
  let j = min(floor(tau), km1 - 1.0);
  return vec2f(j, tau - j);
}
fn ev1(off: u32, K: u32) -> f32 {
  if (BSP) {
    let b = bspl(K); var acc = 0.0;
    for (var r = 0u; r <= b.p; r++) { acc += b.w[r] * knots[off + b.s0 + r]; }
    return acc;
  }
  let s = seg(K); let a = off + u32(s.x);
  if (K <= 1u) { return knots[a]; }
  return mix(knots[a], knots[a + 1u], s.y);
}
fn ev3(off: u32, K: u32) -> vec3f {
  if (BSP) {
    let b = bspl(K); var acc = vec3f(0.0);
    for (var r = 0u; r <= b.p; r++) { let a = off + (b.s0 + r) * 3u; acc += b.w[r] * vec3f(knots[a], knots[a + 1u], knots[a + 2u]); }
    return acc;
  }
  let s = seg(K); let a = off + u32(s.x) * 3u;
  let A = vec3f(knots[a], knots[a + 1u], knots[a + 2u]);
  if (K <= 1u) { return A; }
  return mix(A, vec3f(knots[a + 3u], knots[a + 4u], knots[a + 5u]), s.y);
}
fn ev4(off: u32, K: u32) -> vec4f {
  if (BSP) {
    let b = bspl(K); var acc = vec4f(0.0);
    for (var r = 0u; r <= b.p; r++) { let a = off + (b.s0 + r) * 4u; acc += b.w[r] * vec4f(knots[a], knots[a + 1u], knots[a + 2u], knots[a + 3u]); }
    return acc;
  }
  let s = seg(K); let a = off + u32(s.x) * 4u;
  let A = vec4f(knots[a], knots[a + 1u], knots[a + 2u], knots[a + 3u]);
  if (K <= 1u) { return A; }
  return mix(A, vec4f(knots[a + 4u], knots[a + 5u], knots[a + 6u], knots[a + 7u]), s.y);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.n) { return; }
  let h = i * 8u;
  let fi = floor(U.t);
  let alive = f32(hdr[h]) <= fi && fi < f32(hdr[h + 1u]);
  if (!alive) { world[i * 4u] = vec4f(0.0); return; }
  let kp = hdr[h + 2u];
  BSP = (kp >> 31u) != 0u;
  let p = ev3(hdr[h + 3u], kp & 15u);
  var q = ev4(hdr[h + 4u], (kp >> 4u) & 15u);
  q = q / max(length(q), 1e-12);                // nlerp
  let s = ev3(hdr[h + 5u], (kp >> 8u) & 15u);
  let o = ev1(hdr[h + 6u], (kp >> 12u) & 15u);
  let d = ev3(hdr[h + 7u], (kp >> 16u) & 15u);
  world[i * 4u] = vec4f(p, 1.0);
  world[i * 4u + 1u] = q;
  world[i * 4u + 2u] = vec4f(s, o);
  world[i * 4u + 3u] = vec4f(d, 0.0);
}
`;

// .splq frame (src/splq.ts): the 256-byte header is the uniform, 4 u32 per splat -> same world layout as EVAL_WGSL.
export const EVALQ_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> H: array<vec4f, 16>;
@group(0) @binding(1) var<storage, read> qd: array<u32>;
@group(0) @binding(2) var<storage, read_write> world: array<vec4f>;
@group(0) @binding(3) var<storage, read> run: array<u32>;      // 16-bit positions (flag 4), else a dummy

fn hf(k: u32) -> f32 { return H[k >> 2u][k & 3u]; }
fn dq(k: u32, q: f32) -> f32 { return hf(4u + k) + q * hf(18u + k); }
fn u8at(w: u32, b: u32) -> f32 { return f32((w >> (8u * b)) & 255u); }
fn i8at(w: u32, b: u32) -> f32 { return f32(extractBits(bitcast<i32>(w), 8u * b, 8u)); }
fn qmul(a: vec4f, b: vec4f) -> vec4f {          // Hamilton product, wxyz
  return vec4f(a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w, a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
               a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y, a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = bitcast<u32>(H[0].z);
  if (i >= n) { return; }
  let w0 = qd[i * 4u]; let w1 = qd[i * 4u + 1u]; let w2 = qd[i * 4u + 2u]; let w3 = qd[i * 4u + 3u];
  var p = vec3f(dq(0u, f32(w0 & 1023u)), dq(1u, f32((w0 >> 10u) & 1023u)), dq(2u, f32((w0 >> 20u) & 1023u)));
  if ((bitcast<u32>(H[0].w) & 4u) != 0u) {
    let a = run[i * 2u]; let b = run[i * 2u + 1u];
    p = vec3f(dq(0u, f32(a & 65535u)), dq(1u, f32(a >> 16u)), dq(2u, f32(b & 65535u)));
  }
  var q = vec4f(dq(3u, i8at(w1, 0u)), dq(4u, i8at(w1, 1u)), dq(5u, i8at(w1, 2u)), dq(6u, i8at(w1, 3u)));
  q = q / max(length(q), 1e-12);
  var s = vec3f(dq(7u, u8at(w2, 0u)), dq(8u, u8at(w2, 1u)), dq(9u, u8at(w2, 2u)));
  let o = dq(10u, u8at(w2, 3u));
  let d = vec3f(dq(11u, u8at(w3, 0u)), dq(12u, u8at(w3, 1u)), dq(13u, u8at(w3, 2u)));
  if ((bitcast<u32>(H[0].w) & 1u) != 0u) {
    p = vec3f(dot(H[8], vec4f(p, 1.0)), dot(H[9], vec4f(p, 1.0)), dot(H[10], vec4f(p, 1.0)));
    q = qmul(H[11], q);
    s = s + vec3f(H[12].x);
  }
  world[i * 4u] = vec4f(p, 1.0);
  world[i * 4u + 1u] = q;
  world[i * 4u + 2u] = vec4f(s, o);
  world[i * 4u + 3u] = vec4f(d, 0.0);
}
`;

export const PROJECT_WGSL = /* wgsl */ `
struct Cam { view: mat4x4f, campos: vec4f, intr: vec4f, size: vec4f, n: u32, n2: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> C: Cam;
@group(0) @binding(1) var<storage, read> world: array<vec4f>;
@group(0) @binding(2) var<storage, read> sn: array<u32>;
@group(0) @binding(3) var<storage, read> snq: array<f32>;      // 45 offsets, 45 scales
@group(0) @binding(4) var<storage, read_write> splat2d: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> vals: array<u32>;
@group(0) @binding(7) var<storage, read_write> counter: array<atomic<u32>>;   // drawIndirect args

fn shn(i: u32, k: u32) -> f32 {                 // dequantised SN scalar k of splat i
  let w = sn[i * 12u + (k >> 2u)];
  let b = f32((w >> ((k & 3u) * 8u)) & 255u);
  return snq[k] + b * snq[45u + k];
}
fn shc(i: u32, c: u32) -> vec3f {               // coefficient c (1..15) of splat i
  let k = (c - 1u) * 3u;
  return vec3f(shn(i, k), shn(i, k + 1u), shn(i, k + 2u));
}

fn sh_color(i: u32, dc: vec3f, dir: vec3f) -> vec3f {
  let x = dir.x; let y = dir.y; let z = dir.z;
  var r = 0.28209479177387814 * dc;
  if (C.p0 != 0u) { return max(r + 0.5, vec3f(0.0)); }    // SH0-only source (.splq): no higher bands
  r += 0.4886025119029199 * (-y * shc(i, 1u) + z * shc(i, 2u) - x * shc(i, 3u));
  let xx = x * x; let yy = y * y; let zz = z * z; let xy = x * y; let yz = y * z; let xz = x * z;
  r += 1.0925484305920792 * xy * shc(i, 4u) + -1.0925484305920792 * yz * shc(i, 5u)
     + 0.31539156525252005 * (2.0 * zz - xx - yy) * shc(i, 6u) + -1.0925484305920792 * xz * shc(i, 7u)
     + 0.5462742152960396 * (xx - yy) * shc(i, 8u);
  r += -0.5900435899266435 * y * (3.0 * xx - yy) * shc(i, 9u) + 2.890611442640554 * xy * z * shc(i, 10u)
     + -0.4570457994644658 * y * (4.0 * zz - xx - yy) * shc(i, 11u)
     + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shc(i, 12u)
     + -0.4570457994644658 * x * (4.0 * zz - xx - yy) * shc(i, 13u) + 1.445305721320277 * z * (xx - yy) * shc(i, 14u)
     + -0.5900435899266435 * x * (xx - 3.0 * yy) * shc(i, 15u);
  return max(r + 0.5, vec3f(0.0));
}

// C.p1: 0 own sort; 1 sort shared with later eyes (no screen cull); 2 splat2d only, reuse the shared order
fn hide(i: u32) { splat2d[i * 3u] = vec4f(0.0); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= C.n2) { return; }
  let own = C.p1 != 2u;
  if (own) {
    vals[i] = i;
    keys[i] = 0xffffffffu;
  }
  if (i >= C.n) { return; }
  let w0 = world[i * 4u];
  if (w0.w == 0.0) { return; }                   // not alive at t
  let p = w0.xyz;
  let mc = (C.view * vec4f(p, 1.0)).xyz;
  let z = mc.z;
  if (z < C.size.z || z > 1e10) { if (!own) { hide(i); } return; }   // C.size.z = near plane per scene (P13)
  let q = world[i * 4u + 1u];
  let so = world[i * 4u + 2u];
  let s = exp(so.xyz);
  let qw = q.x; let qx = q.y; let qy = q.z; let qz = q.w;
  // rotation columns (gsplat normalized_quat_to_rotmat, wxyz)
  let r0 = vec3f(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy));
  let r1 = vec3f(2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx));
  let r2 = vec3f(2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy));
  let sig = s.x * s.x * mat3x3f(r0 * r0.x, r0 * r0.y, r0 * r0.z)
          + s.y * s.y * mat3x3f(r1 * r1.x, r1 * r1.y, r1 * r1.z)
          + s.z * s.z * mat3x3f(r2 * r2.x, r2 * r2.y, r2 * r2.z);
  let W = mat3x3f(C.view[0].xyz, C.view[1].xyz, C.view[2].xyz);
  let cc = W * sig * transpose(W);
  let fx = C.intr.x; let fy = C.intr.y; let cx = C.intr.z; let cy = C.intr.w;
  let Wd = C.size.x; let Hd = C.size.y;
  let tfx = 0.5 * Wd / fx; let tfy = 0.5 * Hd / fy;
  let lxp = (Wd - cx) / fx + 0.3 * tfx; let lxn = cx / fx + 0.3 * tfx;
  let lyp = (Hd - cy) / fy + 0.3 * tfy; let lyn = cy / fy + 0.3 * tfy;
  let tx = z * min(lxp, max(-lxn, mc.x / z));
  let ty = z * min(lyp, max(-lyn, mc.y / z));
  let j0 = vec3f(fx / z, 0.0, -fx * tx / (z * z));
  let j1 = vec3f(0.0, fy / z, -fy * ty / (z * z));
  let a = dot(j0, cc * j0) + 0.3;
  let b = dot(j0, cc * j1);
  let c = dot(j1, cc * j1) + 0.3;
  let det = a * c - b * b;
  let op = 1.0 / (1.0 + exp(-so.w));
  if (det <= 0.0 || op < 1.0 / 255.0) { if (!own) { hide(i); } return; }
  let conic = vec3f(c / det, -b / det, a / det);
  let mean2 = vec2f(fx * mc.x / z + cx, fy * mc.y / z + cy);
  let bm = 0.5 * (a + c);
  let v1 = bm + sqrt(max(0.01, bm * bm - det));
  // beyond sqrt(2 ln(255 op)) sigma the fragment's alpha is under 1/255 and discarded
  let rad = ceil(min(3.0, sqrt(2.0 * log(255.0 * op))) * sqrt(v1));
  let off = rad <= 0.0 || mean2.x + rad <= 0.0 || mean2.x - rad >= Wd || mean2.y + rad <= 0.0 || mean2.y - rad >= Hd;
  if (off && C.p1 != 1u) { if (!own) { hide(i); } return; }
  let col = sh_color(i, world[i * 4u + 3u].xyz, normalize(p - C.campos.xyz));
  splat2d[i * 3u] = vec4f(mean2, z, rad);
  splat2d[i * 3u + 1u] = vec4f(conic, op);
  splat2d[i * 3u + 2u] = vec4f(col, 0.0);
  if (!own) { return; }
  keys[i] = bitcast<u32>(z);
  atomicAdd(&counter[1], 1u);
}
`;

export const SORT_WGSL = /* wgsl */ `
struct SortU { k: u32, j: u32, mode: u32, p: u32 };
@group(0) @binding(0) var<uniform> S: SortU;
@group(0) @binding(1) var<storage, read_write> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> vals: array<u32>;

@compute @workgroup_size(256)
fn global_step(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  let j = S.j;
  let i = 2u * j * (t / j) + (t % j);
  let l = i + j;
  if (l >= arrayLength(&keys)) { return; }
  let asc = (i & S.k) == 0u;
  let ki = keys[i]; let kl = keys[l];
  if ((ki > kl) == asc && ki != kl) {
    keys[i] = kl; keys[l] = ki;
    let vi = vals[i]; vals[i] = vals[l]; vals[l] = vi;
  }
}

var<workgroup> sk: array<u32, 512>;
var<workgroup> sv: array<u32, 512>;

fn lstep(t: u32, base: u32, k: u32, j: u32) {
  let i = 2u * j * (t / j) + (t % j);
  let l = i + j;
  let asc = ((base + i) & k) == 0u;
  let ki = sk[i]; let kl = sk[l];
  if ((ki > kl) == asc && ki != kl) {
    sk[i] = kl; sk[l] = ki;
    let vi = sv[i]; sv[i] = sv[l]; sv[l] = vi;
  }
}

// mode 0: full bitonic sort of every 512-block (k = 2..512); mode 1: finish merge stage S.k for j = 256..1
@compute @workgroup_size(256)
fn local_sort(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let t = lid.x;
  let base = wid.x * 512u;
  sk[t] = keys[base + t]; sk[t + 256u] = keys[base + t + 256u];
  sv[t] = vals[base + t]; sv[t + 256u] = vals[base + t + 256u];
  workgroupBarrier();
  if (S.mode == 0u) {
    for (var k = 2u; k <= 512u; k = k * 2u) {
      for (var j = k / 2u; j > 0u; j = j / 2u) {
        lstep(t, base, k, j);
        workgroupBarrier();
      }
    }
  } else {
    for (var j = 256u; j > 0u; j = j / 2u) {
      lstep(t, base, S.k, j);
      workgroupBarrier();
    }
  }
  keys[base + t] = sk[t]; keys[base + t + 256u] = sk[t + 256u];
  vals[base + t] = sv[t]; vals[base + t + 256u] = sv[t + 256u];
}
`;

export const RASTER_WGSL = /* wgsl */ `
struct Cam { view: mat4x4f, campos: vec4f, intr: vec4f, size: vec4f, n: u32, n2: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> C: Cam;
@group(0) @binding(1) var<storage, read> splat2d: array<vec4f>;
@group(0) @binding(2) var<storage, read> vals: array<u32>;

// Real-world depth (WebXR depth-sensing, matchDepthView false); see Occlusion in xr.ts.
// params: rawValueToMeters, kind (0 off, 1 float32, 2 uint16), depth size
// extra: near, margin (metres), far, w 1 = texels are hardware depth 0..1 between near and far (Quest), 0 = metres from near
// dbg.x 1: occluded fragments are tinted red instead of dropped
struct Occ { F: mat4x4f, zrow: vec4f, params: vec4f, extra: vec4f, dbg: vec4f };
@group(1) @binding(0) var<uniform> O: Occ;
@group(1) @binding(1) var depth_f32: texture_2d<f32>;
@group(1) @binding(2) var depth_u16: texture_2d<u32>;

fn real_occludes(px: vec2f, z: f32) -> bool {
  let kind = O.params.y;
  if (kind < 0.5) { return false; }
  let w = vec4f(px * z, z, 1.0);
  let h = O.F * w;
  if (h.w <= 0.0) { return false; }
  let uv = h.xy / h.w;
  if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) { return false; }
  let t = vec2i(uv * O.params.zw);
  var raw: f32;
  if (kind > 1.5) { raw = f32(textureLoad(depth_u16, t, 0).r); } else { raw = textureLoad(depth_f32, t, 0).r; }
  let dist = -dot(O.zrow, w);
  if (O.extra.w > 0.5) {
    let n = O.extra.x; let f = O.extra.z;
    let d = select(raw, raw / 65535.0, kind > 1.5);
    if (d <= 0.0 || d >= 1.0) { return false; }
    return n * f / (f - d * (f - n)) + O.extra.y < dist;
  }
  let meters = raw * O.params.x;
  return meters > 0.0 && meters + O.extra.y < dist - O.extra.x;
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) d: vec2f,
  @location(1) conic: vec3f,
  @location(2) col: vec4f,
  @location(3) @interpolate(flat) z: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let idx = vals[ii];
  let a = splat2d[idx * 3u];
  let b = splat2d[idx * 3u + 1u];
  let c = splat2d[idx * 3u + 2u];
  let corner = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
  let px = a.xy + corner * a.w;
  var o: VOut;
  o.pos = vec4f(px.x / C.size.x * 2.0 - 1.0, 1.0 - px.y / C.size.y * 2.0, 0.0, 1.0);
  o.d = corner * a.w;
  o.conic = b.xyz;
  o.col = vec4f(c.xyz, b.w);
  o.z = a.z;
  return o;
}

@fragment
fn fs(v: VOut) -> @location(0) vec4f {
  let d = v.d;
  let sigma = 0.5 * (v.conic.x * d.x * d.x + v.conic.z * d.y * d.y) + v.conic.y * d.x * d.y;
  if (sigma < 0.0) { discard; }
  let alpha = min(0.999, v.col.w * exp(-sigma));
  if (alpha < 1.0 / 255.0) { discard; }
  if (real_occludes(v.pos.xy, v.z)) {
    if (O.dbg.x < 0.5) { discard; }
    return vec4f(alpha, 0.0, 0.0, alpha);
  }
  return vec4f(v.col.rgb * alpha, alpha);
}
`;

export const BLIT_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> bg: vec4f;    // background behind the premultiplied splats
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let c = textureLoad(src, vec2i(p.xy), 0);
  return vec4f(clamp(c.rgb + (1.0 - c.a) * bg.rgb, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

// Eye blit into an XR projection layer. The triangle covers the pass viewport; `origin` is that
// viewport's top-left so a sub-rect still samples the eye-sized float target. Alpha stays (passthrough)
// or is forced to 1. Unorm layer formats want sRGB bytes; `-srgb` formats encode on write.
export const BLIT_XR_WGSL = /* wgsl */ `
// gain: linear rgb gain, w contrast; grade: x saturation, y on (recolor.ts)
struct BlitU { origin: vec2f, flags: vec2f, gain: vec4f, grade: vec4f }; // flags.x keepAlpha, flags.y encodeSrgb
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> U: BlitU;

fn lin2srgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let dims = vec2i(textureDimensions(src));
  let texel = clamp(vec2i(p.xy) - vec2i(U.origin), vec2i(0), dims - vec2i(1));
  let c = textureLoad(src, texel, 0);
  var rgb = clamp(c.rgb, vec3f(0.0), vec3f(1.0));
  if (U.grade.y > 0.5 && c.a > 1e-4) {
    var x = clamp(c.rgb / c.a, vec3f(0.0), vec3f(1.0));
    x = pow(min(pow(x, vec3f(2.2)) * U.gain.rgb, vec3f(1.0)), vec3f(1.0 / 2.2));
    x = (x - 0.5) * U.gain.w + 0.5;
    x = mix(vec3f(dot(x, vec3f(0.2126, 0.7152, 0.0722))), x, U.grade.x);
    rgb = clamp(x, vec3f(0.0), vec3f(1.0)) * clamp(c.a, 0.0, 1.0);
  }
  if (U.flags.y > 0.5) { rgb = lin2srgb(rgb); }
  let a = select(1.0, clamp(c.a, 0.0, 1.0), U.flags.x > 0.5);
  return vec4f(rgb, a);
}
`;

// |render − reference| per pixel; the reference (GT/original still) is fitted with the same contain rect as the
// GT pane (camera.ts fitRect), so both are pixel-aligned. gain scales the difference (×1 = plain |ours − Original|).
export const DIFF_WGSL = /* wgsl */ `
struct DiffU { ox: f32, oy: f32, dw: f32, dh: f32, refW: f32, refH: f32, gain: f32, pad: f32 };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var refTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> U: DiffU;
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let c = clamp(textureLoad(src, vec2i(p.xy), 0).rgb, vec3f(0.0), vec3f(1.0));
  let u = (p.xy - vec2f(U.ox, U.oy)) / vec2f(U.dw, U.dh);
  var g = vec3f(0.0);
  if (u.x >= 0.0 && u.y >= 0.0 && u.x < 1.0 && u.y < 1.0) {
    g = textureLoad(refTex, vec2i(u * vec2f(U.refW, U.refH)), 0).rgb;
  }
  return vec4f(clamp(abs(c - g) * U.gain, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;
