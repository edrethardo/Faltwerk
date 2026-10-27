// WebXR + WebGPU binding for the splat player. Session startup follows the goo MR path
// (requiredFeatures: ['webgpu'], projection layer, local-floor when the runtime grants it).
import type { Camera } from './renderer.ts';

declare global {
  class XRGPUBinding {
    constructor(session: XRSession, device: GPUDevice);
    createProjectionLayer(init: { colorFormat: GPUTextureFormat; depthStencilFormat?: GPUTextureFormat; scaleFactor?: number }): XRProjectionLayer;
    getViewSubImage(layer: XRProjectionLayer, view: XRView): XRGPUSubImage;
    getPreferredColorFormat(): GPUTextureFormat;
    getDepthInformation?(view: XRView): XRGPUDepthInformation | null;
  }
  interface XRGPUDepthInformation {
    readonly width: number;
    readonly height: number;
    readonly texture: GPUTexture;
    readonly normDepthBufferFromNormView: XRRigidTransform;
    readonly rawValueToMeters: number;
    readonly projectionMatrix: Float32Array;
    readonly transform: XRRigidTransform;
    /** Quest: the texels are hardware depth (0..1) between these planes, not metres */
    readonly depthNear?: number;
    readonly depthFar?: number;
    getViewDescriptor?: () => GPUTextureViewDescriptor;
  }
  interface XRGPUSubImage {
    viewport: XRViewport;
    colorTexture: GPUTexture;
    depthStencilTexture?: GPUTexture | null;
    getViewDescriptor?: () => GPUTextureViewDescriptor;
  }
}

export interface Placement {
  /** scene-to-room rotation, column-major 3×3 */
  R: Float32Array;
  /** metres per scene unit */
  scale: number;
  /** entry camera position in scene space */
  cam: [number, number, number];
  /** viewer position in room space at anchor */
  anchor: [number, number, number];
  /** scale at anchor; a pinch stays inside 0.05×–20× this */
  scale0: number;
  /** false while the runtime is still reporting an emulated pose */
  locked: boolean;
}

/** One eye, ready for Renderer.frameEyes. */
export interface XREye {
  cam: Camera;
  color: GPUTextureView;
  format: GPUTextureFormat;
  attachmentWidth: number;
  attachmentHeight: number;
  viewport: { x: number; y: number; width: number; height: number };
  /** false when an earlier eye in this frame already cleared the same attachment */
  clear: boolean;
  /** passthrough: keep the splats' alpha. opaque blend: write 1 */
  keepAlpha: boolean;
  /** layer format is unorm, not `-srgb`, so the blit encodes */
  encodeSrgb: boolean;
  /** real-world depth for this eye (depth-sensing), null when the frame has none */
  occ: Occlusion | null;
}

/**
 * Depth-sensing occlusion for one eye, laid out as the raster shader's `Occ` uniform.
 * A fragment at pixel (u, v) of a splat at camera depth z (scene units) is w = (u z, v z, z, 1):
 * `F · w` is the depth-buffer uv times clip w, and `zrow · w` the point's z in the depth camera's view (metres, −forward).
 */
export interface Occlusion {
  /**
   * F (16), zrow (4), params: rawValueToMeters, kind (1 float32, 2 uint16), width, height;
   * extra: near, margin, far, 1 when texels are hardware depth (Quest) else 0; debug: x 1 tints occluded red
   */
  data: Float32Array<ArrayBuffer>;
  texture: GPUTexture;
  kind: 'float' | 'uint';
  view?: GPUTextureViewDescriptor;
}

const OCC_MARGIN = 0.02;

function mul4(a: ArrayLike<number>, b: ArrayLike<number>, out = new Float32Array(16)): Float32Array {
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    out[c * 4 + r] = s;
  }
  return out;
}

/** WebGPU clip (z in 0..1): near = P[14] / P[10]. */
function projNear(p: ArrayLike<number>): number {
  if (Math.abs(p[10]) < 1e-8) return 0;
  const near = p[14] / p[10];
  return near > 0 && near < 10 ? near : 0;
}

/** NDC xy → normalised view uv (top-left, +y down), z dropped. */
const NDC_TO_UV = new Float32Array([0.5, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 1]);

export function occlusionData(
  cam: Camera, scale: number, eyePose: ArrayLike<number>, depthPoseInv: ArrayLike<number>,
  depthProj: ArrayLike<number>, uvFromView: ArrayLike<number>,
  rawValueToMeters: number, kind: 'float' | 'uint', width: number, height: number,
  windowDepth: { near: number; far: number } | null = null, out = new Float32Array(32),
): Float32Array {
  const s = scale;
  const E = new Float32Array([s / cam.fx, 0, 0, 0, 0, -s / cam.fy, 0, 0, -s * cam.cx / cam.fx, s * cam.cy / cam.fy, -s, 0, 0, 0, 0, 1]);
  const A = mul4(mul4(depthPoseInv, eyePose), E);
  const F = mul4(mul4(uvFromView, NDC_TO_UV), mul4(depthProj, A));
  out.set(F, 0);
  out[16] = A[2]; out[17] = A[6]; out[18] = A[10]; out[19] = A[14];
  out[20] = rawValueToMeters; out[21] = kind === 'float' ? 1 : 2; out[22] = width; out[23] = height;
  out[24] = windowDepth ? windowDepth.near : projNear(depthProj); out[25] = OCC_MARGIN;
  out[26] = windowDepth ? windowDepth.far : 0; out[27] = windowDepth ? 1 : 0;
  return out;
}

type V3 = [number, number, number];

const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];

function norm(a: V3): V3 | null {
  const l = Math.hypot(a[0], a[1], a[2]);
  if (l < 1e-8) return null;
  return [a[0] / l, a[1] / l, a[2] / l];
}

function horizontal(v: V3, up: V3): V3 | null {
  return norm(sub(v, mul(up, dot(v, up))));
}

/** Columns: right, up, forward. Right = forward × up (OpenCV image-right when up is scene up). */
function frameOf(forward: V3, upIn: V3): [V3, V3, V3] | null {
  const up = norm(upIn);
  if (!up) return null;
  let fwd = horizontal(forward, up);
  if (!fwd) {
    const hint: V3 = Math.abs(up[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    fwd = norm(cross(hint, up));
    if (!fwd) return null;
  }
  const right = norm(cross(fwd, up));
  if (!right) return null;
  return [right, up, fwd];
}

function m3(cols: [V3, V3, V3]): Float32Array {
  const m = new Float32Array(9);
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) m[c * 3 + r] = cols[c][r];
  return m;
}

function mulM3(a: Float32Array, b: Float32Array): Float32Array {
  const m = new Float32Array(9);
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) {
    m[c * 3 + r] = a[r] * b[c * 3] + a[3 + r] * b[c * 3 + 1] + a[6 + r] * b[c * 3 + 2];
  }
  return m;
}

function transposeM3(a: Float32Array): Float32Array {
  const m = new Float32Array(9);
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) m[c * 3 + r] = a[r * 3 + c];
  return m;
}

function mulM3V(a: Float32Array, v: V3): V3 {
  return [
    a[0] * v[0] + a[3] * v[1] + a[6] * v[2],
    a[1] * v[0] + a[4] * v[1] + a[7] * v[2],
    a[2] * v[0] + a[5] * v[1] + a[8] * v[2],
  ];
}

/**
 * Anchor the scene so `center` sits `meters` from the head along the horizontal look, scene up
 * aligned with world up. Headset motion is converted back into scene units; the camera matrix
 * itself stays a pure rotation so the splat covariance (view 3×3) stays valid.
 */
export function place(cam: V3, center: V3, up: V3, head: V3, headForward: V3, meters: number): Placement | null {
  const scene = frameOf(sub(center, cam), up);
  const room = frameOf(headForward, [0, 1, 0]);
  if (!scene || !room) return null;
  const dist = Math.hypot(center[0] - cam[0], center[1] - cam[1], center[2] - cam[2]);
  const scale = dist > 1e-8 && meters > 0 ? meters / dist : 1;
  return {
    R: mulM3(m3(room), transposeM3(m3(scene))),
    scale,
    scale0: scale,
    cam: [cam[0], cam[1], cam[2]],
    anchor: [head[0], head[1], head[2]],
    locked: false,
  };
}

/** fx/fy/cx/cy from a WebXR projection matrix, in the viewport's pixel space (y down). */
export function intrinsics(p: ArrayLike<number>, width: number, height: number) {
  return {
    fx: p[0] * 0.5 * width,
    fy: p[5] * 0.5 * height,
    cx: width * (1 - p[8]) * 0.5,
    cy: height * (1 + p[9]) * 0.5,
  };
}

/**
 * OpenCV camera for one eye. `pose` is the eye's camera-to-room matrix (column-major, GL: x right,
 * y up, −z forward). `proj` is the WebXR projection matrix.
 */
export function eyeCamera(p: Placement, proj: ArrayLike<number>, pose: ArrayLike<number>, width: number, height: number): Camera {
  const Rt = transposeM3(p.R);
  const c0 = mulM3V(Rt, [pose[0], pose[1], pose[2]]);
  const c1 = mulM3V(Rt, [pose[4], pose[5], pose[6]]);
  const c2 = mulM3V(Rt, [pose[8], pose[9], pose[10]]);
  const r0: V3 = c0;
  const r1 = mul(c1, -1);
  const r2 = mul(c2, -1);
  const eye: V3 = [pose[12], pose[13], pose[14]];
  const campos = add(p.cam, mul(mulM3V(Rt, sub(eye, p.anchor)), 1 / p.scale));
  const view = new Float32Array(16);
  const rows = [r0, r1, r2];
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) view[c * 4 + r] = rows[r][c];
  view[12] = -(r0[0] * campos[0] + r0[1] * campos[1] + r0[2] * campos[2]);
  view[13] = -(r1[0] * campos[0] + r1[1] * campos[1] + r1[2] * campos[2]);
  view[14] = -(r2[0] * campos[0] + r2[1] * campos[1] + r2[2] * campos[2]);
  view[15] = 1;
  const k = intrinsics(proj, width, height);
  return { view, campos, ...k, width, height };
}

/** Both hands stay pinched, apart, before the content follows them. */
const PLACE_ARM_MS = 400;
const PLACE_ARM_SPAN = 0.12;
const PLACE_YAW_SPAN = 0.1;
/** A hand or a twist that jumps this far in one frame is a let-go glitch, not a move. */
const PLACE_JUMP = 0.05;
const PLACE_YAW_JUMP = 0.18;
const SCALE_STEP = 0.08;
const PINCH_ON = 0.016;
const PINCH_OFF = 0.04;
const PINCH_ON_BONE = 0.032;
const PINCH_OFF_BONE = 0.055;

export interface GrabState {
  armAt: number | null;
  placing: boolean;
  last: [V3, V3] | null;
}

export function grabState(): GrabState {
  return { armAt: null, placing: false, last: null };
}

/** Thumb–index gap, in metres. `radii` is the sum of the joint radii; 0 when the runtime omits them. */
export function pinchClosed(was: boolean, gap: number, radii: number): boolean {
  const on = radii > 0 ? PINCH_ON : PINCH_ON_BONE;
  const off = radii > 0 ? PINCH_OFF : PINCH_OFF_BONE;
  return was ? gap < off : gap < on;
}

function copy3(v: V3): V3 {
  return [v[0], v[1], v[2]];
}

function yawAboutY(radians: number): Float32Array {
  const c = Math.cos(radians);
  const s = Math.sin(radians);
  const m = new Float32Array(9);
  m[0] = c; m[2] = -s; m[4] = 1; m[6] = s; m[8] = c;
  return m;
}

/**
 * One frame of a two-hand grab. The content translates with the pinch midpoint, yaws about world
 * up, and scales about the previous midpoint so it stays between the hands. The view 3×3 stays a
 * rotation. A hand that teleports more than 5 cm is dropped; returns false in that case.
 */
export function applyGrab(p: Placement, from: [V3, V3], to: [V3, V3]): boolean {
  const [la, lb] = from;
  const [a, b] = to;
  if (Math.hypot(a[0] - la[0], a[1] - la[1], a[2] - la[2]) > PLACE_JUMP) return false;
  if (Math.hypot(b[0] - lb[0], b[1] - lb[1], b[2] - lb[2]) > PLACE_JUMP) return false;
  const lastMid = mul(add(la, lb), 0.5);
  const mid = mul(add(a, b), 0.5);
  let k = 1;
  const lastDist = Math.hypot(lb[0] - la[0], lb[1] - la[1], lb[2] - la[2]);
  const dist = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  if (dist >= 0.06 && lastDist >= 0.06) {
    const next = dist / lastDist;
    if (Math.abs(next - 1) <= SCALE_STEP) k = next;
  }
  const lo = p.scale0 * 0.05;
  const hi = p.scale0 * 20;
  if (p.scale > 1e-8) {
    if (p.scale * k < lo) k = lo / p.scale;
    else if (p.scale * k > hi) k = hi / p.scale;
  }
  let yaw = 0;
  const spanH = Math.hypot(b[0] - a[0], b[2] - a[2]);
  const lastH = Math.hypot(lb[0] - la[0], lb[2] - la[2]);
  if (spanH >= PLACE_YAW_SPAN && lastH >= PLACE_YAW_SPAN) {
    const ang = Math.atan2(b[0] - a[0], b[2] - a[2]);
    const prev = Math.atan2(lb[0] - la[0], lb[2] - la[2]);
    const d = Math.atan2(Math.sin(ang - prev), Math.cos(ang - prev));
    if (Math.abs(d) <= PLACE_YAW_JUMP) yaw = d;
  }
  const spun = mulM3V(yawAboutY(yaw), sub(p.anchor, lastMid));
  p.anchor = add(mid, mul(spun, k));
  p.scale *= k;
  if (yaw !== 0) p.R = mulM3(yawAboutY(yaw), p.R);
  return true;
}

/** Arm after both pinches have stayed at least 12 cm apart for 400 ms, then follow. */
export function stepGrab(g: GrabState, a: V3, b: V3, now: number, placement: Placement): void {
  const span = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const spanning = span >= (g.armAt !== null && !g.placing ? PLACE_ARM_SPAN * 0.65 : PLACE_ARM_SPAN);
  if (g.placing || spanning) {
    if (!g.placing) {
      if (g.armAt === null) g.armAt = now;
      if (now - g.armAt >= PLACE_ARM_MS) g.placing = true;
    }
    if (g.placing) {
      if (g.last) applyGrab(placement, g.last, [a, b]);
      g.last = [copy3(a), copy3(b)];
    }
  } else {
    g.armAt = null;
    g.placing = false;
    g.last = null;
  }
}

function jointAt(frame: XRFrame, space: XRSpace | undefined, ref: XRReferenceSpace): { p: V3; r: number } | null {
  if (!space) return null;
  const withJoint = frame as XRFrame & { getJointPose?: (joint: XRSpace, base: XRSpace) => (XRPose & { radius?: number }) | undefined };
  const pose: (XRPose & { radius?: number }) | undefined = withJoint.getJointPose?.(space, ref) ?? frame.getPose(space, ref);
  if (!pose) return null;
  const q = pose.transform.position;
  return { p: [q.x, q.y, q.z], r: pose.radius ?? 0 };
}

function readTips(frame: XRFrame, hand: XRHand, ref: XRReferenceSpace): { gap: number; radii: number; point: V3 } | null {
  const thumb = jointAt(frame, hand.get('thumb-tip'), ref);
  const index = jointAt(frame, hand.get('index-finger-tip'), ref);
  if (!thumb || !index) return null;
  return {
    gap: Math.hypot(thumb.p[0] - index.p[0], thumb.p[1] - index.p[1], thumb.p[2] - index.p[2]) - thumb.r - index.r,
    radii: thumb.r + index.r,
    point: [(thumb.p[0] + index.p[0]) * 0.5, (thumb.p[1] + index.p[1]) * 0.5, (thumb.p[2] + index.p[2]) * 0.5],
  };
}

/** Pinch points of the left and right hands. Both must be pinched before the content moves. */
class Hands {
  private readonly pinch = new Map<'left' | 'right', boolean>();
  private readonly gate: GrabState = grabState();

  update(frame: XRFrame, session: XRSession, ref: XRReferenceSpace, placement: Placement, now: number): void {
    const pts = new Map<'left' | 'right', V3>();
    const seen = new Set<'left' | 'right'>();
    for (let i = 0; i < session.inputSources.length; i++) {
      const source = session.inputSources[i];
      const hand = source.hand;
      const side = source.handedness;
      if (!hand || (side !== 'left' && side !== 'right')) continue;
      seen.add(side);
      let tips: { gap: number; radii: number; point: V3 } | null = null;
      try {
        tips = readTips(frame, hand, ref);
      } catch {
        tips = null;
      }
      const was = this.pinch.get(side) ?? false;
      const held = !!tips && pinchClosed(was, tips.gap, tips.radii);
      this.pinch.set(side, held);
      if (held && tips) pts.set(side, tips.point);
    }
    for (const side of ['left', 'right'] as const) if (!seen.has(side)) this.pinch.delete(side);
    const a = pts.get('left');
    const b = pts.get('right');
    if (!a || !b) {
      this.gate.armAt = null;
      this.gate.placing = false;
      this.gate.last = null;
      return;
    }
    stepGrab(this.gate, a, b, now, placement);
  }
}

/** Immersive-ar session bound to a WebGPU device. null from diagnose means it should work. */
export class ImmersiveAR {
  readonly session: XRSession;
  readonly binding: XRGPUBinding;
  readonly layer: XRProjectionLayer;
  readonly refSpace: XRReferenceSpace;
  readonly colorFormat: GPUTextureFormat;
  /** The real world shows through wherever alpha is 0. */
  readonly passthrough: boolean;
  /** GPU depth (float32 or unsigned-short) was granted with matchDepthView false */
  readonly depthSensing: boolean;
  /** turns the real-world depth test off without ending the session */
  occlude = true;
  /** tint occluded splat fragments red instead of dropping them */
  occDebug = false;
  private occLogged = false;
  private readonly hands = new Hands();
  private readonly occData: Float32Array<ArrayBuffer>[] = [];

  private constructor(
    session: XRSession,
    binding: XRGPUBinding,
    layer: XRProjectionLayer,
    refSpace: XRReferenceSpace,
    colorFormat: GPUTextureFormat,
    passthrough: boolean,
    depthSensing: boolean,
 ) {
    this.session = session;
    this.binding = binding;
    this.layer = layer;
    this.refSpace = refSpace;
    this.colorFormat = colorFormat;
    this.passthrough = passthrough;
    this.depthSensing = depthSensing;
  }

  static async diagnose(mode: XRSessionMode = 'immersive-ar'): Promise<string | null> {
    if (!navigator.xr) return 'no WebXR (navigator.xr)';
    const ok = await navigator.xr.isSessionSupported(mode).catch(() => false);
    if (!ok) return `${mode} not supported`;
    if (!('XRGPUBinding' in globalThis)) return 'no WebXR/WebGPU binding (XRGPUBinding)';
    return null;
  }

  static async start(device: GPUDevice, scaleFactor = 1): Promise<ImmersiveAR> {
    const init = {
      requiredFeatures: ['webgpu'],
      optionalFeatures: ['local-floor', 'hand-tracking', 'depth-sensing'],
      depthSensing: { usagePreference: ['gpu-optimized'], dataFormatPreference: ['float32', 'unsigned-short'], matchDepthView: false },
    } as XRSessionInit;
    const session = await requestWithTimeout('immersive-ar', init, 6000);
    try {
      const binding = new XRGPUBinding(session, device);
      const colorFormat = binding.getPreferredColorFormat?.() ?? navigator.gpu.getPreferredCanvasFormat();
      const layer = binding.createProjectionLayer({ colorFormat, scaleFactor });
      session.updateRenderState({ layers: [layer], depthNear: 0.01, depthFar: 50 } as XRRenderStateInit);
      const features = session.enabledFeatures ?? [];
      const floor = features.includes('local-floor');
      const refSpace = await session.requestReferenceSpace(floor ? 'local-floor' : 'local');
      const blend = session.environmentBlendMode;
      const passthrough = blend === 'alpha-blend' || blend === 'additive';
      const depth = ImmersiveAR.depthFormat(session, binding);
      console.info(`MR: ${colorFormat} blend=${blend} floor=${floor} hands=${features.includes('hand-tracking')} depth=${depth ?? 'off'}`);
      return new ImmersiveAR(session, binding, layer, refSpace, colorFormat, passthrough, depth !== null);
    } catch (error) {
      void session.end().catch(() => {});
      throw error;
    }
  }

  private static depthFormat(session: XRSession, binding: XRGPUBinding): 'float32' | 'unsigned-short' | null {
    if (!session.enabledFeatures?.includes('depth-sensing') || !binding.getDepthInformation) return null;
    try {
      const fmt = (session as XRSession & { depthDataFormat?: string }).depthDataFormat;
      if (fmt === 'float32' || fmt === 'unsigned-short') return fmt;
      if (fmt) console.warn(`MR: depth-sensing format ${fmt} has no WebGPU texture; no occlusion`);
    } catch { /* granted but not usable */ }
    return null;
  }

  private occlusionFor(view: XRView, cam: Camera, scale: number, slot: number): Occlusion | null {
    if (!this.depthSensing || !this.occlude) return null;
    let info: XRGPUDepthInformation | null = null;
    try {
      info = this.binding.getDepthInformation!(view);
    } catch {
      return null;
    }
    if (!info?.texture) return null;
    const fmt = info.texture.format;
    const kind = fmt === 'r32float' ? 'float' : fmt === 'r16uint' ? 'uint' : null;
    if (!kind) return null;
    const data = this.occData[slot] ?? (this.occData[slot] = new Float32Array(32));
    const n = info.depthNear, f = info.depthFar;
    const windowDepth = typeof n === 'number' && typeof f === 'number' && n > 0 ? { near: n, far: f > n ? f : 1e6 } : null;
    occlusionData(cam, scale, view.transform.matrix, info.transform.inverse.matrix, info.projectionMatrix,
      info.normDepthBufferFromNormView.matrix, info.rawValueToMeters, kind, info.width, info.height, windowDepth, data);
    data[28] = this.occDebug ? 1 : 0;
    const layers = info.texture.depthOrArrayLayers;
    const src = info.getViewDescriptor?.();
    const desc: GPUTextureViewDescriptor = src ? { ...src } : layers > 1 ? { baseArrayLayer: slot } : {};
    delete desc.format;
    if (layers > 1) { desc.dimension = '2d'; desc.arrayLayerCount = 1; }
    if (!this.occLogged) {
      this.occLogged = true;
      console.info(`MR depth: ${fmt} ${info.width}x${info.height}x${layers}, rawValueToMeters ${info.rawValueToMeters}, ` +
        `near ${n} far ${f} (${windowDepth ? 'hardware depth' : 'linear'}), view descriptor ${src ? JSON.stringify(src) : 'none'}, ` +
        `proj[10] ${info.projectionMatrix[10].toFixed(4)} proj[14] ${info.projectionMatrix[14].toFixed(4)}`);
    }
    return { data, texture: info.texture, kind, view: desc };
  }

  /** Move, yaw and scale the placement when both hands are pinched. */
  pinch(frame: XRFrame, placement: Placement, now: number): void {
    this.hands.update(frame, this.session, this.refSpace, placement, now);
  }

  /** Per-eye cameras and projection-layer views for this pose. */
  eyes(pose: XRViewerPose, placement: Placement): XREye[] {
    const encodeSrgb = !this.colorFormat.endsWith('-srgb');
    const keepAlpha = this.passthrough;
    const drawn: { tex: GPUTexture; layer: number }[] = [];
    const out: XREye[] = [];
    for (const view of pose.views) {
      const sub = this.binding.getViewSubImage(this.layer, view);
      const vp = sub.viewport;
      if (vp.width < 1 || vp.height < 1) continue;
      const tex = sub.colorTexture;
      const srcDesc = sub.getViewDescriptor?.();
      const layerIndex = srcDesc?.baseArrayLayer ?? 0;
      const mip = srcDesc?.baseMipLevel ?? 0;
      const desc: GPUTextureViewDescriptor = { ...(srcDesc ?? {}) };
      delete desc.format;
      const clear = !drawn.some((d) => d.tex === tex && d.layer === layerIndex);
      drawn.push({ tex, layer: layerIndex });
      const cam = eyeCamera(placement, view.projectionMatrix, view.transform.matrix, vp.width, vp.height);
      out.push({
        cam,
        occ: this.occlusionFor(view, cam, placement.scale, out.length),
        color: tex.createView(desc),
        format: this.colorFormat,
        attachmentWidth: Math.max(1, tex.width >> mip),
        attachmentHeight: Math.max(1, tex.height >> mip),
        viewport: { x: vp.x, y: vp.y, width: vp.width, height: vp.height },
        clear,
        keepAlpha,
        encodeSrgb,
      });
    }
    return out;
  }
}

function requestWithTimeout(mode: XRSessionMode, init: XRSessionInit, ms: number): Promise<XRSession> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const timer = window.setTimeout(() => {
      if (settled) return;
      settled = true;
      reject(new Error(`requestSession timed out after ${ms}ms`));
    }, ms);
    navigator.xr!.requestSession(mode, init).then(
      (session) => {
        if (settled) {
          void session.end();
          return;
        }
        settled = true;
        window.clearTimeout(timer);
        resolve(session);
      },
      (error: unknown) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        reject(error instanceof Error ? error : new Error(String(error)));
      },
 );
  });
}
