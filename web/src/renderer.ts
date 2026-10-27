// WebGPU splat renderer for packed .csgs sections (see shaders.ts). One Renderer per device.
import { BLIT_WGSL, BLIT_XR_WGSL, DIFF_WGSL, EVAL_WGSL, EVALQ_WGSL, PROJECT_WGSL, RASTER_WGSL, SORT_WGSL } from './shaders.ts';
import type { PackedSection } from './pack.ts';
import { SPLQ_FLAG_P16, SPLQ_FLAG_SN, SPLQ_HDR, splqSnWords, type SplqFrame } from './splq.ts';
import type { XREye } from './xr.ts';
import { gradeGain, type Grade } from './recolor.ts';

// Near plane (camera space z): splats closer than this are culled, like gsplat's near_plane. Scene-dependent, default as in gsplat.
export const renderCfg = { near: 0.01 };

export interface Camera {
  view: Float32Array;     // 4x4 world->camera, column-major, OpenCV axes (x right, y down, z forward)
  campos: [number, number, number];
  fx: number; fy: number; cx: number; cy: number;
  width: number; height: number;
}

interface PendingWrite { buf: GPUBuffer; data: ArrayBufferView; done: number }

export interface GpuSection {
  ps: PackedSection; hdr: GPUBuffer; knots: GPUBuffer; sn: GPUBuffer; snq: GPUBuffer; bytes: number;
  /** writes still to issue; null once the section is fully on the GPU */
  pending: PendingWrite[] | null;
  bgs?: { ver: number; eval: GPUBindGroup; proj: GPUBindGroup; raster: GPUBindGroup };
}

/** one .splq frame slot on the GPU (re-filled in place when the frame changes; capacity grows only). */
export interface GpuQFrame {
  q: true; n: number; cap: number; hdr: GPUBuffer; data: GPUBuffer; bytes: number;
  run: GPUBuffer | null; sn: GPUBuffer | null; snq: GPUBuffer | null; flags: number;   // optional blocks
  bgs?: { ver: number; eval: GPUBindGroup; proj: GPUBindGroup; raster: GPUBindGroup };
}
export type GpuSource = GpuSection | GpuQFrame;
const isQ = (g: GpuSource): g is GpuQFrame => (g as GpuQFrame).q === true;

/** difference view: |render − ref| instead of the render; ref fitted into the pane with rect (camera.ts fitRect). */
export interface DiffRef { tex: GPUTexture; ox: number; oy: number; dw: number; dh: number; gain: number }

const ST = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
export const PASSES = ['eval', 'project', 'sort', 'raster', 'blit'] as const;

function nextPow2(n: number) { let p = 512; while (p < n) p *= 2; return p; }

export class Renderer {
  device: GPUDevice;
  format: GPUTextureFormat;
  bytes = new Map<string, number>();     // live GPU allocations by name
  peakBytes = 0;
  private evalPipe: GPUComputePipeline; private evalqPipe: GPUComputePipeline; private projPipe: GPUComputePipeline;
  private dummySn: GPUBuffer; private dummySnq: GPUBuffer;
  private sortGlobal: GPUComputePipeline; private sortLocal: GPUComputePipeline;
  private rasterPipe: GPURenderPipeline; private blitPipe: GPURenderPipeline; private diffPipe: GPURenderPipeline | null = null;
  private evalU: GPUBuffer; private camU: GPUBuffer; private indirect: GPUBuffer; private bgU: GPUBuffer; private diffU: GPUBuffer | null = null;
  private sortU: GPUBuffer | null = null; private sortSteps: { k: number; j: number; mode: number }[] = [];
  private cap = 0; private n2 = 0;
  private world!: GPUBuffer; private splat2d!: GPUBuffer; private keys!: GPUBuffer; private vals!: GPUBuffer;
  private target: GPUTexture | null = null; private targetSize = [0, 0];
  private xrTargets: GPUTexture[] = [];
  private xrBlitPipe: GPURenderPipeline | null = null;
  private xrBlitFormat: GPUTextureFormat | null = null;
  private xrBlitU: GPUBuffer | null = null;
  private xrBlitData = new Float32Array(12);
  private evalScratch = new ArrayBuffer(16);
  private evalF = new Float32Array(this.evalScratch); private evalI = new Uint32Array(this.evalScratch);
  private camScratch = new ArrayBuffer(128);
  private camF = new Float32Array(this.camScratch); private camI = new Uint32Array(this.camScratch);
  private indirectReset = new Uint32Array([4, 0, 0, 0]);
  private views = new WeakMap<GPUTexture, GPUTextureView>();
  private blitBGs = new WeakMap<GPUTexture, { pipe: GPURenderPipeline; bg: GPUBindGroup }>();
  private occLayout: GPUBindGroupLayout;
  private occFloat: GPUTexture; private occUint: GPUTexture;
  private occOff: GPUBindGroup;
  private occSlots: { buf: GPUBuffer; bg: GPUBindGroup; tex: GPUTexture | null; layer: number }[] = [];
  private ts: { set: GPUQuerySet; resolve: GPUBuffer; read: GPUBuffer[]; busy: boolean[]; next: number } | null = null;
  lastTimings: Record<string, number> | null = null;
  timingsLog: Record<string, number>[] = [];
  visibleBuf: GPUBuffer;

  constructor(device: GPUDevice, format: GPUTextureFormat, timestamps: boolean) {
    this.device = device; this.format = format;
    const mod = (code: string) => device.createShaderModule({ code });
    const cp = (code: string, entryPoint = 'main') => device.createComputePipeline({ layout: 'auto', compute: { module: mod(code), entryPoint } });
    this.evalPipe = cp(EVAL_WGSL);
    this.evalqPipe = cp(EVALQ_WGSL);
    this.projPipe = cp(PROJECT_WGSL);
    const sm = mod(SORT_WGSL);
    this.sortGlobal = device.createComputePipeline({ layout: 'auto', compute: { module: sm, entryPoint: 'global_step' } });
    this.sortLocal = device.createComputePipeline({ layout: 'auto', compute: { module: sm, entryPoint: 'local_sort' } });
    const rm = mod(RASTER_WGSL);
    const blend: GPUBlendComponent = { srcFactor: 'one-minus-dst-alpha', dstFactor: 'one', operation: 'add' };
    const V = GPUShaderStage.VERTEX, F = GPUShaderStage.FRAGMENT;
    const rasterBGL = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: V, buffer: { type: 'uniform' } },
      { binding: 1, visibility: V, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: V, buffer: { type: 'read-only-storage' } }] });
    this.occLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: F, buffer: { type: 'uniform' } },
      { binding: 1, visibility: F, texture: { sampleType: 'unfilterable-float' } },
      { binding: 2, visibility: F, texture: { sampleType: 'uint' } }] });
    const tex1 = (format: GPUTextureFormat) => device.createTexture({ size: [1, 1], format, usage: GPUTextureUsage.TEXTURE_BINDING });
    this.occFloat = tex1('r32float'); this.occUint = tex1('r16uint');
    this.occOff = this.occSlot(-1).bg;
    this.rasterPipe = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [rasterBGL, this.occLayout] }), vertex: { module: rm, entryPoint: 'vs' },
      fragment: { module: rm, entryPoint: 'fs', targets: [{ format: 'rgba16float', blend: { color: blend, alpha: blend } }] },
      primitive: { topology: 'triangle-strip' },
    });
    const bm = mod(BLIT_WGSL);
    this.blitPipe = device.createRenderPipeline({
      layout: 'auto', vertex: { module: bm, entryPoint: 'vs' },
      fragment: { module: bm, entryPoint: 'fs', targets: [{ format }] }, primitive: { topology: 'triangle-list' },
    });
    this.bgU = this.buf('bgU', 16, GPUBufferUsage.UNIFORM | CD);
    this.evalU = this.buf('evalU', 16, GPUBufferUsage.UNIFORM | CD);
    this.camU = this.buf('camU', 128, GPUBufferUsage.UNIFORM | CD);
    this.indirect = this.buf('indirect', 16, GPUBufferUsage.INDIRECT | ST | CD | CS);
    this.visibleBuf = this.buf('visibleRead', 16, GPUBufferUsage.MAP_READ | CD);
    this.dummySn = this.buf('dummySn', 48, ST);
    this.dummySnq = this.buf('dummySnq', 360, ST);
    if (timestamps) {
      const n = PASSES.length * 2;
      this.ts = {
        set: device.createQuerySet({ type: 'timestamp', count: n }),
        resolve: this.buf('tsResolve', n * 8, GPUBufferUsage.QUERY_RESOLVE | CS),
        read: [0, 1, 2].map((i) => this.buf(`tsRead${i}`, n * 8, GPUBufferUsage.MAP_READ | CD)),
        busy: [false, false, false], next: 0,
      };
    }
  }

  /** desktop canvas background, linear 0..1 (the splats are composited over it) */
  setBackground(r: number, g: number, b: number): void {
    this.device.queue.writeBuffer(this.bgU, 0, new Float32Array([r, g, b, 1]));
  }

  /** grade applied by the XR eye blit; null turns it off */
  setGrade(g: Grade | null): void {
    const d = this.xrBlitData;
    if (!g) { d.fill(0, 4); return; }
    d.set(gradeGain(g), 4);
    d[7] = g.contrast; d[8] = g.saturation; d[9] = 1;
  }

  private buf(name: string, size: number, usage: number): GPUBuffer {
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage, label: name });
    this.bytes.set(name, (this.bytes.get(name) ?? 0) + b.size);
    this.peakBytes = Math.max(this.peakBytes, this.gpuBytes());
    return b;
  }
  private free(name: string, b: GPUBuffer) {
    this.bytes.set(name, (this.bytes.get(name) ?? 0) - b.size);
    b.destroy();
  }
  gpuBytes(): number {
    let s = 0;
    for (const v of this.bytes.values()) s += v;
    if (this.target) s += this.targetSize[0] * this.targetSize[1] * 8;
    for (const t of this.xrTargets) s += t.width * t.height * 8;
    return s;
  }

  /** Allocates the section's buffers; the data goes up through {@link pump}. */
  upload(ps: PackedSection): GpuSection {
    const pending: PendingWrite[] = [];
    const mk = (nm: string, data: ArrayBufferView) => {
      const b = this.buf(`sec_${nm}`, data.byteLength, ST | CD);
      pending.push({ buf: b, data, done: 0 });
      return b;
    };
    const snq = new Float32Array(90); snq.set(ps.snOffset, 0); snq.set(ps.snScale, 45);
    const g: GpuSection = { ps, hdr: mk('hdr', ps.hdr), knots: mk('knots', ps.knots), sn: mk('sn', ps.sn), snq: mk('snq', snq), bytes: 0, pending };
    g.bytes = g.hdr.size + g.knots.size + g.sn.size + g.snq.size;
    this.ensureCapacity(ps.n);
    return g;
  }

  /** Issue up to `budget` bytes of a section's pending writes (Infinity finishes it). */
  pump(g: GpuSection, budget: number): void {
    const q = this.device.queue;
    while (g.pending?.length && budget > 0) {
      const w = g.pending[0];
      const len = Math.min(w.data.byteLength - w.done, budget >= 2 ** 30 ? Infinity : Math.max(4, budget & ~3));
      q.writeBuffer(w.buf, w.done, w.data.buffer, w.data.byteOffset + w.done, len);
      w.done += len;
      budget -= len;
      if (w.done >= w.data.byteLength) g.pending.shift();
    }
    if (g.pending && !g.pending.length) g.pending = null;
  }

  release(g: GpuSource) {
    if (isQ(g)) {
      this.free('q_hdr', g.hdr); this.free('q_data', g.data);
      if (g.run) this.free('q_p16', g.run);
      if (g.sn) this.free('q_sn', g.sn);
      if (g.snq) this.free('q_snq', g.snq);
      return;
    }
    this.free('sec_hdr', g.hdr); this.free('sec_knots', g.knots); this.free('sec_sn', g.sn); this.free('sec_snq', g.snq);
  }

  /** upload a .splq frame into `slot` (reused when it is large enough and has the same optional blocks, else replaced). */
  uploadQ(fr: SplqFrame, slot: GpuQFrame | null = null): GpuQFrame {
    const opt = fr.flags & (SPLQ_FLAG_SN | SPLQ_FLAG_P16);
    if (!slot || slot.cap < fr.n || slot.flags !== opt) {
      if (slot) this.release(slot);
      const cap = Math.ceil(fr.n * 1.05) + 256;
      slot = { q: true, n: fr.n, cap, hdr: this.buf('q_hdr', SPLQ_HDR, GPUBufferUsage.UNIFORM | CD), data: this.buf('q_data', cap * 16, ST | CD),
        run: fr.run ? this.buf('q_p16', cap * 8, ST | CD) : null, sn: fr.sn ? this.buf('q_sn', cap * 48, ST | CD) : null,
        snq: fr.sn ? this.buf('q_snq', 360, ST | CD) : null, flags: opt, bytes: 0 };
      slot.bytes = slot.hdr.size + slot.data.size + (slot.run?.size ?? 0) + (slot.sn?.size ?? 0) + (slot.snq?.size ?? 0);
    }
    slot.n = fr.n;
    this.device.queue.writeBuffer(slot.hdr, 0, fr.header.buffer, fr.header.byteOffset, SPLQ_HDR);
    this.device.queue.writeBuffer(slot.data, 0, fr.data.buffer, fr.data.byteOffset, fr.n * 16);
    if (fr.run && slot.run) this.device.queue.writeBuffer(slot.run, 0, fr.run.buffer, fr.run.byteOffset, fr.n * 8);
    const snw = splqSnWords(fr);
    if (snw && slot.sn && slot.snq) {
      this.device.queue.writeBuffer(slot.sn, 0, snw.buffer, 0, fr.n * 48);
      this.device.queue.writeBuffer(slot.snq, 0, fr.snq!.buffer, fr.snq!.byteOffset, 360);
    }
    this.ensureCapacity(fr.n);
    return slot;
  }

  private ensureCapacity(n: number) {
    if (n <= this.cap) return;
    if (this.cap) { this.free('world', this.world); this.free('splat2d', this.splat2d); this.free('keys', this.keys); this.free('vals', this.vals); }
    this.cap = n;
    this.n2 = nextPow2(n);
    this.world = this.buf('world', n * 64, ST | CS);
    this.splat2d = this.buf('splat2d', n * 48, ST);
    this.keys = this.buf('keys', this.n2 * 4, ST | CS);
    this.vals = this.buf('vals', this.n2 * 4, ST | CS);
    // sort schedule for n2
    const steps: { k: number; j: number; mode: number }[] = [{ k: 512, j: 0, mode: 0 }];
    for (let k = 1024; k <= this.n2; k *= 2) {
      for (let j = k / 2; j >= 512; j /= 2) steps.push({ k, j, mode: 2 });
      steps.push({ k, j: 0, mode: 1 });
    }
    this.sortSteps = steps;
    if (this.sortU) this.free('sortU', this.sortU);
    this.sortU = this.buf('sortU', steps.length * 256, GPUBufferUsage.UNIFORM | CD);
    const u = new Uint32Array(steps.length * 64);
    steps.forEach((s, i) => { u[i * 64] = s.k; u[i * 64 + 1] = s.j; u[i * 64 + 2] = s.mode === 2 ? 0 : s.mode; });
    this.device.queue.writeBuffer(this.sortU, 0, u);
    this.sortBGs = steps.map((s, idx) => {
      const pipe = s.mode === 2 ? this.sortGlobal : this.sortLocal;
      return this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.sortU!, offset: idx * 256, size: 16 } },
        { binding: 1, resource: { buffer: this.keys, size: this.n2 * 4 } }, { binding: 2, resource: { buffer: this.vals, size: this.n2 * 4 } }] });
    });
    this.capVersion++;
  }
  private sortBGs: GPUBindGroup[] = [];
  private capVersion = 0;

  private bindGroups(g: GpuSource) {
    if (g.bgs && g.bgs.ver === this.capVersion) return g.bgs;
    const dev = this.device;
    const q = isQ(g);
    g.bgs = {
      ver: this.capVersion,
      eval: q
        ? dev.createBindGroup({ layout: this.evalqPipe.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: g.hdr } }, { binding: 1, resource: { buffer: g.data } },
          { binding: 2, resource: { buffer: this.world } }, { binding: 3, resource: { buffer: g.run ?? this.dummySn } }] })
        : dev.createBindGroup({ layout: this.evalPipe.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: this.evalU } }, { binding: 1, resource: { buffer: g.hdr } },
          { binding: 2, resource: { buffer: g.knots } }, { binding: 3, resource: { buffer: this.world } }] }),
      proj: dev.createBindGroup({ layout: this.projPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.camU } }, { binding: 1, resource: { buffer: this.world } },
        { binding: 2, resource: { buffer: q ? (g.sn ?? this.dummySn) : g.sn } }, { binding: 3, resource: { buffer: q ? (g.snq ?? this.dummySnq) : g.snq } },
        { binding: 4, resource: { buffer: this.splat2d } }, { binding: 5, resource: { buffer: this.keys } },
        { binding: 6, resource: { buffer: this.vals } }, { binding: 7, resource: { buffer: this.indirect } }] }),
      raster: dev.createBindGroup({ layout: this.rasterPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.camU } }, { binding: 1, resource: { buffer: this.splat2d } },
        { binding: 2, resource: { buffer: this.vals, size: this.n2 * 4 } }] }),
    };
    return g.bgs;
  }

  sortDispatches() { return this.sortSteps.length; }

  private ensureTarget(w: number, h: number) {
    if (this.target && this.targetSize[0] === w && this.targetSize[1] === h) return;
    this.target?.destroy();
    this.target = this.device.createTexture({ size: [w, h], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.targetSize = [w, h];
    this.peakBytes = Math.max(this.peakBytes, this.gpuBytes());
  }

  /** encode one frame at global frame time t (fractional) into `out` (canvas texture) and submit.
   *  opts.diff: show |render − ref| (difference view) instead of the render. */
  frame(g: GpuSource, t: number, cam: Camera, out: GPUTexture | null, opts: { timing?: boolean; readVisible?: boolean; diff?: DiffRef | null } = {}): void {
    const dev = this.device;
    const ts = opts.timing && this.ts && !this.ts.busy[this.ts.next] ? this.ts : null;
    const enc = dev.createCommandEncoder();
    this.record(enc, g, t, cam, out ? { kind: 'canvas', texture: out, diff: opts.diff ?? null } : null, true, ts, null);
    let slot = -1;
    if (ts) {
      slot = ts.next; ts.next = (ts.next + 1) % ts.read.length; ts.busy[slot] = true;
      const cnt = out ? PASSES.length * 2 : (PASSES.length - 1) * 2;
      enc.resolveQuerySet(ts.set, 0, cnt, ts.resolve, 0);
      enc.copyBufferToBuffer(ts.resolve, 0, ts.read[slot], 0, cnt * 8);
    }
    const readVis = opts.readVisible && this.visibleBuf.mapState === 'unmapped';
    if (readVis) enc.copyBufferToBuffer(this.indirect, 0, this.visibleBuf, 0, 16);
    dev.queue.submit([enc.finish()]);
    if (ts && slot >= 0) {
      const rb = ts.read[slot], hasBlit = !!out;
      rb.mapAsync(GPUMapMode.READ).then(() => {
        const v = new BigUint64Array(rb.getMappedRange().slice(0));
        rb.unmap(); ts.busy[slot] = false;
        const r: Record<string, number> = {};
        PASSES.forEach((nm, p) => { if (p < 4 || hasBlit) r[nm] = Number(v[2 * p + 1] - v[2 * p]) / 1e6; });
        r.gpu_total = Number(v[(hasBlit ? 9 : 7)] - v[0]) / 1e6;
        this.lastTimings = r; this.timingsLog.push(r);
      }).catch(() => { ts.busy[slot] = false; });
    }
    if (readVis) {
      this.visibleBuf.mapAsync(GPUMapMode.READ).then(() => {
        this.lastVisible = new Uint32Array(this.visibleBuf.getMappedRange().slice(0))[1];
        this.visibleBuf.unmap();
      }).catch(() => {});
    }
  }

  /** Both eyes of one immersive frame. Eval runs on the first eye; each eye is its own submit so its camera uniform is visible to that submit only. */
  frameEyes(g: GpuSource, t: number, eyes: XREye[]): void {
    const shared = eyes.length > 1;
    eyes.forEach((eye, i) => {
      const target = this.xrTarget(i, eye.cam.width, eye.cam.height);
      const enc = this.device.createCommandEncoder();
      this.record(enc, g, t, eye.cam, { kind: 'xr', eye }, i === 0, null, target, shared ? (i === 0 ? 1 : 2) : 0, this.occFor(i, eye.occ));
      this.device.queue.submit([enc.finish()]);
    });
  }

  private xrTarget(eye: number, w: number, h: number): GPUTexture {
    const cur = this.xrTargets[eye];
    if (cur && cur.width === w && cur.height === h) return cur;
    cur?.destroy();
    const t = this.device.createTexture({
      size: [Math.max(1, w), Math.max(1, h)],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      label: `xr-eye-${eye}`,
    });
    this.xrTargets[eye] = t;
    this.peakBytes = Math.max(this.peakBytes, this.gpuBytes());
    return t;
  }

  private occGroup(buf: GPUBuffer, tex: GPUTexture | null, kind: 'float' | 'uint', desc?: GPUTextureViewDescriptor): GPUBindGroup {
    return this.device.createBindGroup({ layout: this.occLayout, entries: [
      { binding: 0, resource: { buffer: buf } },
      { binding: 1, resource: (tex && kind === 'float' ? tex : this.occFloat).createView(tex && kind === 'float' ? desc : undefined) },
      { binding: 2, resource: (tex && kind === 'uint' ? tex : this.occUint).createView(tex && kind === 'uint' ? desc : undefined) }] });
  }

  /** slot -1: the always-off group (zero uniform) */
  private occSlot(i: number) {
    if (i >= 0 && this.occSlots[i]) return this.occSlots[i];
    const buf = this.buf('occU', 128, GPUBufferUsage.UNIFORM | CD);
    const s = { buf, bg: this.occGroup(buf, null, 'float'), tex: null as GPUTexture | null, layer: -1 };
    if (i >= 0) this.occSlots[i] = s;
    return s;
  }

  /** Real-world depth test for eye `i` (one uniform per eye, so both survive their submits). */
  private occFor(i: number, occ: XREye['occ']): GPUBindGroup {
    if (!occ) return this.occOff;
    const s = this.occSlot(i);
    this.device.queue.writeBuffer(s.buf, 0, occ.data);
    const layer = occ.view?.baseArrayLayer ?? 0;
    if (s.tex !== occ.texture || s.layer !== layer) {
      s.tex = occ.texture; s.layer = layer;
      s.bg = this.occGroup(s.buf, occ.texture, occ.kind, occ.view);
    }
    return s.bg;
  }

  private xrBlitPipeline(format: GPUTextureFormat): { pipe: GPURenderPipeline; uni: GPUBuffer } {
    if (!this.xrBlitU) this.xrBlitU = this.buf('xrBlitU', 48, GPUBufferUsage.UNIFORM | CD);
    if (!this.xrBlitPipe || this.xrBlitFormat !== format) {
      const bm = this.device.createShaderModule({ code: BLIT_XR_WGSL });
      this.xrBlitPipe = this.device.createRenderPipeline({
        label: 'xr-blit',
        layout: 'auto',
        vertex: { module: bm, entryPoint: 'vs' },
        fragment: { module: bm, entryPoint: 'fs', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      });
      this.xrBlitFormat = format;
    }
    return { pipe: this.xrBlitPipe, uni: this.xrBlitU };
  }

  private record(
    enc: GPUCommandEncoder,
    g: GpuSource,
    t: number,
    cam: Camera,
    dest: { kind: 'canvas'; texture: GPUTexture; diff?: DiffRef | null } | { kind: 'xr'; eye: XREye } | null,
    doEval: boolean,
    ts: { set: GPUQuerySet } | null,
    raster: GPUTexture | null,
    /** 0 own sort; 1 sort for this and the following eyes; 2 reuse that order (splat2d only) */
    order: 0 | 1 | 2 = 0,
    occ: GPUBindGroup = this.occOff,
 ): void {
    const dev = this.device, q = isQ(g), n = q ? g.n : g.ps.n;
    this.ensureCapacity(n);
    if (!raster) this.ensureTarget(cam.width, cam.height);
    const target = raster ?? this.target!;
    if (doEval && !q) {
      const ps = g.ps, ef = this.evalF;
      ef[0] = t; ef[1] = ps.g0; ef[2] = Math.max(ps.nframes - 1, 1); this.evalI[3] = n;
      dev.queue.writeBuffer(this.evalU, 0, this.evalScratch);
    }
    const cf = this.camF, ci = this.camI;
    cf.set(cam.view, 0);
    cf[16] = cam.campos[0]; cf[17] = cam.campos[1]; cf[18] = cam.campos[2];
    cf[20] = cam.fx; cf[21] = cam.fy; cf[22] = cam.cx; cf[23] = cam.cy;
    cf[24] = cam.width; cf[25] = cam.height; cf[26] = renderCfg.near; cf[27] = 0;   // size.z: near plane per scene
    ci[28] = n; ci[29] = this.n2; ci[30] = q && !g.sn ? 1 : 0; ci[31] = order;   // 1 = SH0 only (.splq without SN bands, P23)
    dev.queue.writeBuffer(this.camU, 0, this.camScratch);
    if (order !== 2) dev.queue.writeBuffer(this.indirect, 0, this.indirectReset);

    const tw = (p: number) => (ts ? { querySet: ts.set, beginningOfPassWriteIndex: 2 * p, endOfPassWriteIndex: 2 * p + 1 } : undefined);
    const bgs = this.bindGroups(g);
    if (doEval) {
      const pass = enc.beginComputePass({ timestampWrites: tw(0) });
      pass.setPipeline(q ? this.evalqPipe : this.evalPipe);
      pass.setBindGroup(0, bgs.eval);
      pass.dispatchWorkgroups(Math.ceil(n / 256));
      pass.end();
    }

    let pass = enc.beginComputePass({ timestampWrites: tw(1) });
    pass.setPipeline(this.projPipe);
    pass.setBindGroup(0, bgs.proj);
    pass.dispatchWorkgroups(Math.ceil(this.n2 / 256));
    pass.end();

    if (order !== 2) {
      pass = enc.beginComputePass({ timestampWrites: tw(2) });
      this.sortSteps.forEach((s, idx) => {
        pass.setPipeline(s.mode === 2 ? this.sortGlobal : this.sortLocal);
        pass.setBindGroup(0, this.sortBGs[idx]);
        pass.dispatchWorkgroups(s.mode === 2 ? Math.ceil(this.n2 / 2 / 256) : this.n2 / 512);
      });
      pass.end();
    }

    const targetView = this.viewOf(target);
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: targetView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }],
      timestampWrites: tw(3),
    });
    rp.setPipeline(this.rasterPipe);
    rp.setBindGroup(0, bgs.raster);
    rp.setBindGroup(1, occ);
    rp.drawIndirect(this.indirect, 0);
    rp.end();

    if (dest?.kind === 'canvas') {
      const bp = enc.beginRenderPass({
        colorAttachments: [{ view: dest.texture.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
        timestampWrites: tw(4),
      });
      const df = dest.diff;
      if (df) {                                            // |render − ref| instead of the render
        if (!this.diffPipe) {
          const dm = dev.createShaderModule({ code: DIFF_WGSL });
          this.diffPipe = dev.createRenderPipeline({
            layout: 'auto', vertex: { module: dm, entryPoint: 'vs' },
            fragment: { module: dm, entryPoint: 'fs', targets: [{ format: this.format }] }, primitive: { topology: 'triangle-list' },
          });
          this.diffU = this.buf('diffU', 32, GPUBufferUsage.UNIFORM | CD);
        }
        dev.queue.writeBuffer(this.diffU!, 0, new Float32Array([df.ox, df.oy, df.dw, df.dh, df.tex.width, df.tex.height, df.gain, 0]));
        bp.setPipeline(this.diffPipe);
        bp.setBindGroup(0, dev.createBindGroup({ layout: this.diffPipe.getBindGroupLayout(0), entries: [
          { binding: 0, resource: this.viewOf(target) }, { binding: 1, resource: df.tex.createView() }, { binding: 2, resource: { buffer: this.diffU! } }] }));
      } else {
        bp.setPipeline(this.blitPipe);
        bp.setBindGroup(0, this.blitBindGroup(this.blitPipe, target, this.bgU));
      }
      bp.draw(3);
      bp.end();
    } else if (dest?.kind === 'xr') {
      const eye = dest.eye;
      const { pipe, uni } = this.xrBlitPipeline(eye.format);
      this.xrBlitData[0] = eye.viewport.x; this.xrBlitData[1] = eye.viewport.y;
      this.xrBlitData[2] = eye.keepAlpha ? 1 : 0; this.xrBlitData[3] = eye.encodeSrgb ? 1 : 0;
      dev.queue.writeBuffer(uni, 0, this.xrBlitData);
      const bp = enc.beginRenderPass({
        colorAttachments: [{
          view: eye.color,
          clearValue: { r: 0, g: 0, b: 0, a: eye.keepAlpha ? 0 : 1 },
          loadOp: eye.clear ? 'clear' : 'load',
          storeOp: 'store',
        }],
      });
      bp.setPipeline(pipe);
      bp.setBindGroup(0, this.blitBindGroup(pipe, target, uni));
      const full = eye.viewport.x === 0 && eye.viewport.y === 0
        && eye.viewport.width === eye.attachmentWidth && eye.viewport.height === eye.attachmentHeight;
      // Full-layer views (Quest texture array) must not set a viewport: visionOS treats it as a compositor hint.
      if (!full) bp.setViewport(eye.viewport.x, eye.viewport.y, eye.viewport.width, eye.viewport.height, 0, 1);
      bp.draw(3);
      bp.end();
    }
  }
  lastVisible = 0;

  private viewOf(tex: GPUTexture): GPUTextureView {
    let v = this.views.get(tex);
    if (!v) { v = tex.createView(); this.views.set(tex, v); }
    return v;
  }

  private blitBindGroup(pipe: GPURenderPipeline, src: GPUTexture, uni: GPUBuffer | null): GPUBindGroup {
    const hit = this.blitBGs.get(src);
    if (hit && hit.pipe === pipe) return hit.bg;
    const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: this.viewOf(src) }];
    if (uni) entries.push({ binding: 1, resource: { buffer: uni } });
    const bg = this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    this.blitBGs.set(src, { pipe, bg });
    return bg;
  }

  /** float render target -> 8-bit RGB (same clamp as the blit), for PSNR checks. */
  async readTarget(): Promise<{ w: number; h: number; rgb: Uint8Array }> {
    const { w, h, bpr, src } = await this.readTargetHalf();
    const rgb = new Uint8Array(w * h * 3);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
      const v = half(src[(y * bpr) / 2 + x * 4 + c]);
      rgb[(y * w + x) * 3 + c] = Math.round(Math.min(1, Math.max(0, v)) * 255);
    }
    return { w, h, rgb };
  }

  /** every `step`-th pixel of the last desktop frame, premultiplied RGBA; null before the first frame. */
  async readTargetRGBA(step = 1): Promise<Float32Array | null> {
    if (!this.target) return null;
    const { w, h, bpr, src } = await this.readTargetHalf();
    const out = new Float32Array(Math.ceil((w * h) / step) * 4);
    let o = 0;
    for (let i = 0; i < w * h; i += step) {
      const base = Math.floor(i / w) * (bpr / 2) + (i % w) * 4;
      for (let c = 0; c < 4; c++) out[o++] = half(src[base + c]);
    }
    return out;
  }

  private async readTargetHalf(): Promise<{ w: number; h: number; bpr: number; src: Uint16Array }> {
    const [w, h] = this.targetSize;
    const bpr = Math.ceil((w * 8) / 256) * 256;
    const rb = this.device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.MAP_READ | CD });
    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.target! }, { buffer: rb, bytesPerRow: bpr }, [w, h]);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const src = new Uint16Array(rb.getMappedRange().slice(0));
    rb.destroy();
    return { w, h, bpr, src };
  }

  /** evaluated world attributes of splats `idx` (GPU eval pass output of the last frame). */
  async readWorld(idx: number[]): Promise<Float32Array[]> {
    const rb = this.device.createBuffer({ size: this.world.size, usage: GPUBufferUsage.MAP_READ | CD });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.world, 0, rb, 0, this.world.size);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(rb.getMappedRange().slice(0));
    rb.destroy();
    return idx.map((i) => all.slice(i * 16, i * 16 + 16));
  }
}

function half(h: number): number {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}
