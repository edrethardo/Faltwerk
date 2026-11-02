// comparison viewer: 2-4 panes side by side, ONE shared camera and ONE timeline.
// Panes: ours (.falt sections — legacy .csgs too —, curve player), PLY baselines as .splq frame sequences (SH0, same renderer), GT stills
// in snap-to-test-camera mode. Data and labels come from manifest.json (make_viewer_data.py manifest).
// pane quality label (metrics from the run JSONs), GT pane title per scene (e.g. "Original"),
// difference view |ours − Original| in snap mode (renderer diff blit against the GT still, pixel-aligned via fitRect).
import { Renderer, renderCfg, type Camera, type DiffRef, type GpuQFrame, type GpuSection, type GpuSource } from './renderer.ts';
import { evalSplat, type PackedSection } from './pack.ts';
import { evalSplq, parseSplq, type SplqFrame } from './splq.ts';
import { camKey, fitRect, orbitCam, snapCam, type Orbit, type TestCam, type View } from './camera.ts';
import { localizeDom, tr } from './i18n.ts';

// ---------------------------------------------------------------- manifest
// pane kind 'falt' = Faltwerk sections; 'csgs' = the same, as written by manifests before (alias)
interface PaneSpec {
  id: string; title: string; kind: 'falt' | 'csgs' | 'splq'; rate_label: string; rates: { label: string; mbit_s: number }[];
  metric_label?: string;
  sections?: { url: string; g0: number; nframes: number; bytes: number }[]; frames?: string[]; bytes?: number;
  note?: string; default_off?: boolean; source?: string;
}
interface SceneSpec {
  id: string; title: string; fps: number; frames: number; note: string; orbit: Orbit; testcams: TestCam[];
  gt: { dir: string; ext: string; cams: number[]; title?: string } | null; panes: PaneSpec[]; default_snap?: string; near?: number;
}

const Q = new URLSearchParams(location.search);
const OFFSCREEN = Q.get('offscreen') === '1';
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const marks: Record<string, number> = { script_start_ms: performance.now() };
const P9: any = { ready: false, error: null, marks };
(window as any).__p9 = P9;
localizeDom();

function fail(msg: string): never {
  $('status').textContent = tr('Fehler: ', 'Error: ') + msg;
  P9.error = msg;
  throw new Error(msg);
}
const fmtMB = (b: number) => (b / 2 ** 20).toFixed(b < 10 * 2 ** 20 ? 1 : 0).replace('.', ',') + ' MB';

/** fetch with byte progress */
async function fetchBuf(url: string, onBytes: (n: number) => void): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  if (!r.body) { const b = await r.arrayBuffer(); onBytes(b.byteLength); return b; }
  const len = Number(r.headers.get('content-length') ?? 0);
  const rd = r.body.getReader();
  const parts: Uint8Array[] = []; let got = 0;
  for (;;) {
    const { done, value } = await rd.read();
    if (done) break;
    parts.push(value); got += value.length; onBytes(value.length);
  }
  const out = new Uint8Array(len === got ? len : got);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out.buffer;
}

/** one fetch queue PER PANE: all panes stream at the same time (own slots), each in timeline order, so a light pane is
 *  never starved behind a heavy one (e.g. 30 heavy per-frame dumps before our 2nd section). Cancelled requests are skipped. */
const FETCH_SLOTS_PER_PANE = Math.max(1, Number(Q.get('slots') ?? 3));
class FetchQueue {
  private q: { prio: number; seq: number; run: () => Promise<void>; cancelled: () => boolean }[] = [];
  private active = 0; private seq = 0; private slots: number;
  constructor(slots: number) { this.slots = slots; }
  fetch(prio: number, url: string, onBytes: (n: number) => void, cancelled: () => boolean): Promise<ArrayBuffer> {
    return new Promise((res, rej) => {
      this.q.push({ prio, seq: this.seq++, run: () => fetchBuf(url, onBytes).then(res, rej), cancelled });
      this.q.sort((a, b) => a.prio - b.prio || a.seq - b.seq);
      this.pump();
    });
  }
  private pump() {
    while (this.active < this.slots && this.q.length) {
      const job = this.q.shift()!;
      if (job.cancelled()) continue;
      this.active++;
      job.run().finally(() => { this.active--; this.pump(); });
    }
  }
}

// ---------------------------------------------------------------- pane players
abstract class Pane {
  el: HTMLDivElement; canvas: HTMLCanvasElement; label: HTMLDivElement; bar: HTMLDivElement;
  spec: { id: string; title: string; rate_label: string; metric_label?: string };
  bytesTotal = 0; bytesLoaded = 0;
  last: { f: number; t: number; cam: string } | null = null;
  firstFrameMs = 0; loadedMs = 0; renders = 0; status = '';
  visible = true; dead = false;
  fq = new FetchQueue(FETCH_SLOTS_PER_PANE);
  /** pf = frame this pane shows. started = takes part in the current loop; buffering = ran out of data in this loop and
   *  holds its last frame until the loop restarts (then it rejoins if it has frame 0). */
  pf = 0; started = false; buffering = false; stalls = 0; t0 = performance.now();
  constructor(spec: { id: string; title: string; rate_label: string; metric_label?: string }, grid: HTMLElement) {
    this.spec = spec;
    this.el = document.createElement('div'); this.el.className = 'pane'; this.el.dataset.id = spec.id;
    this.canvas = document.createElement('canvas');
    this.label = document.createElement('div'); this.label.className = 'label';
    const pr = document.createElement('div'); pr.className = 'progress';
    this.bar = document.createElement('div'); pr.appendChild(this.bar);
    this.el.append(this.canvas, this.label, pr);
    grid.appendChild(this.el);
  }
  abstract ready(f: number): boolean;
  abstract loadedAll(): boolean;
  progress() { return this.bytesTotal ? this.bytesLoaded / this.bytesTotal : 1; }
  /** fully loaded, or (like <video> canplaythrough) the current frame is there and the measured download rate finishes
   *  the rest before one loop at `fps` has played */
  canPlayThrough(fps: number, frames: number) {
    if (this.loadedAll()) return true;
    if (!this.ready(this.pf)) return false;
    const el = (performance.now() - this.t0) / 1000;
    if (el < 0.5 || !this.bytesLoaded || !this.bytesTotal) return false;
    const eta = (this.bytesTotal - this.bytesLoaded) / (this.bytesLoaded / el);
    return eta < (frames / Math.max(fps, 1e-3)) * 0.9;
  }
  updateLabel(extra = '') {
    const p = this.progress();
    this.bar.style.width = `${(p * 100).toFixed(1)}%`;
    this.bar.parentElement!.style.opacity = p >= 1 && this.loadedAll() ? '0' : '1';
    const ld = p < 1 ? ` · ${tr('lädt', 'loading')} ${(p * 100).toFixed(0)} % (${fmtMB(this.bytesLoaded)} / ${fmtMB(this.bytesTotal)})` : '';
    this.label.innerHTML = `<b>${this.spec.title}</b> <span class="rate">${this.spec.rate_label}</span>` +
      (this.spec.metric_label ? ` <span class="qual">${this.spec.metric_label}</span>` : '') +
      `<span class="st">${ld}${this.status ? ' · ' + this.status : ''}${extra}</span>`;
  }
  destroy() { this.dead = true; this.el.remove(); }
}

abstract class SplatPane extends Pane {
  ctx: GPUCanvasContext; renderer: Renderer; device: GPUDevice; format: GPUTextureFormat;
  constructor(spec: PaneSpec, grid: HTMLElement, device: GPUDevice, format: GPUTextureFormat, ts: boolean) {
    super(spec, grid);
    this.device = device; this.format = format;
    this.ctx = this.canvas.getContext('webgpu') as GPUCanvasContext;
    if (!OFFSCREEN) this.ctx.configure({ device, format, alphaMode: 'opaque' });
    this.renderer = new Renderer(device, format, ts);
  }
  abstract source(f: number, t: number): { g: GpuSource; t: number } | null;
  private off: GPUTexture | null = null;
  /** canvas texture, or (offscreen=1: SwiftShader headless has no usable swap chain) a private texture */
  outTexture(): GPUTexture {
    if (!OFFSCREEN) return this.ctx.getCurrentTexture();
    if (!this.off || this.off.width !== this.canvas.width || this.off.height !== this.canvas.height) {
      this.off?.destroy();
      this.off = this.device.createTexture({ size: [this.canvas.width, this.canvas.height], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
    return this.off;
  }
  abstract splats(): number;
  render(f: number, t: number, cam: Camera, opts: { timing?: boolean; diff?: DiffRef | null } = {}): boolean {
    const s = this.source(f, t);
    if (!s) return false;
    this.renderer.frame(s.g, s.t, cam, this.outTexture(), { timing: opts.timing ?? true, readVisible: this.renders % 10 === 0, diff: opts.diff ?? null });
    this.renders++;
    this.last = { f, t: s.t, cam: camKey(cam) };
    if (!this.firstFrameMs) {
      const dev = this.device;
      dev.queue.onSubmittedWorkDone().then(() => { this.firstFrameMs = performance.now(); marks[`first_frame_${this.spec.id}_ms`] = this.firstFrameMs; });
    }
    return true;
  }
}

const faltKind = (k: string) => k === 'falt' || k === 'csgs';

class FaltPane extends SplatPane {
  sections: NonNullable<PaneSpec['sections']>;
  decoded = new Map<number, PackedSection>();
  gpu = new Map<number, GpuSection>();
  timings: any[] = [];
  worker: Worker; interp: () => boolean;
  constructor(spec: PaneSpec, grid: HTMLElement, device: GPUDevice, format: GPUTextureFormat, ts: boolean, base: URL, interp: () => boolean) {
    super(spec, grid, device, format, ts);
    this.interp = interp;
    this.sections = spec.sections!;
    this.bytesTotal = this.sections.reduce((a, s) => a + s.bytes, 0);
    this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e) => {
      const { seq, ps, timing, error } = e.data;
      if (error) { this.status = tr('Fehler: ', 'Error: ') + error; P9.error = `${spec.id}: ${error}`; return; }
      timing.recv_ms = performance.now();
      this.timings.push(timing);
      this.decoded.set(seq, ps);
      if (this.decoded.size === this.sections.length) { this.loadedMs = performance.now(); marks[`loaded_${spec.id}_ms`] = this.loadedMs; }
    };
    // fetch all sections (parallel, progress), decode strictly in order in the worker (sequential decoder state)
    const bufs = this.sections.map((s) => this.fq.fetch(s.g0, new URL(s.url, base).href, (n) => { this.bytesLoaded += n; }, () => this.dead));
    const worker = this.worker, sections = this.sections;
    (async () => {
      for (let i = 0; i < bufs.length; i++) {             // post in section order, whatever order the fetches finish in
        const buf = await bufs[i];
        worker.postMessage({ seq: i, url: sections[i].url, targetN: 0, buf }, [buf]);
      }
    })().catch((e) => { this.status = String(e); P9.error = `${spec.id}: ${e}`; });
  }
  frames() { return this.sections.reduce((a, s) => a + s.nframes, 0); }
  loadedAll() { return this.decoded.size === this.sections.length; }
  secAt(t: number) {
    let i = this.sections.findIndex((s) => t >= s.g0 && t < s.g0 + s.nframes);
    if (i < 0) i = t < this.sections[0].g0 ? 0 : this.sections.length - 1;
    return i;
  }
  ready(f: number) { return this.decoded.has(this.secAt(f)); }
  splats() { const g = this.gpu.values().next().value; return g ? g.ps.nCoded : 0; }
  source(f: number, t: number) {
    const tt = this.interp() ? t : f;
    const i = this.secAt(tt);
    if (!this.decoded.has(i)) return null;
    let g = this.gpu.get(i);
    if (!g) {
      g = this.renderer.upload(this.decoded.get(i)!);
      this.renderer.pump(g, Infinity);
      this.gpu.set(i, g);
    }
    // GPU: current + next section only
    for (const [k, v] of [...this.gpu]) if (k !== i && k !== i + 1) { this.renderer.release(v); this.gpu.delete(k); }
    const s = this.sections[i];
    return { g, t: Math.min(Math.max(tt, s.g0), s.g0 + s.nframes - 1e-4) };
  }
  evalCheck(g: GpuSource, t: number, got: Float32Array[], ids: number[]) {
    const ps = (g as GpuSection).ps;
    let maxErr = 0, n = 0;
    ids.forEach((i, k) => {
      const ref = evalSplat(ps, i, t);
      if (!ref.alive) return;
      n++;
      const flat = ref.vals.flat();
      const gv = [got[k][0], got[k][1], got[k][2], got[k][4], got[k][5], got[k][6], got[k][7], got[k][8], got[k][9], got[k][10], got[k][11], got[k][12], got[k][13], got[k][14]];
      flat.forEach((v, j) => { maxErr = Math.max(maxErr, Math.abs(v - gv[j]) / Math.max(1, Math.abs(v))); });
    });
    return { splats: n, max_rel_err: maxErr };
  }
  destroy() { this.worker.terminate(); for (const g of this.gpu.values()) this.renderer.release(g); super.destroy(); }
}

class SplqPane extends SplatPane {
  urls: string[]; bufs: (ArrayBuffer | null)[]; slot: GpuQFrame | null = null; slotFrame = -1; cur: SplqFrame | null = null;
  loaded = 0;
  constructor(spec: PaneSpec, grid: HTMLElement, device: GPUDevice, format: GPUTextureFormat, ts: boolean, base: URL) {
    super(spec, grid, device, format, ts);
    this.urls = spec.frames!.map((u) => new URL(u, base).href);
    this.bufs = this.urls.map(() => null);
    this.bytesTotal = spec.bytes ?? 0;
    // frames in timeline order through the shared queue; all compact frames stay in memory, only ONE frame lives on the GPU
    this.urls.forEach((u, i) => {
      this.fq.fetch(i, u, (n) => { this.bytesLoaded += n; }, () => this.dead).then((b) => {
        this.bufs[i] = b;
        if (++this.loaded === this.urls.length) { this.loadedMs = performance.now(); marks[`loaded_${spec.id}_ms`] = this.loadedMs; }
      }).catch((e) => { if (!this.dead) { this.status = String(e); P9.error = `${spec.id}: ${e}`; } });
    });
  }
  frames() { return this.urls.length; }
  loadedAll() { return this.loaded === this.urls.length; }
  idx(f: number) { return Math.min(Math.max(Math.floor(f), 0), this.urls.length - 1); }
  ready(f: number) { return !!this.bufs[this.idx(f)]; }
  splats() { return this.cur ? this.cur.n : 0; }
  source(f: number) {
    const i = this.idx(f);
    if (!this.bufs[i]) return null;
    if (this.slotFrame !== i) {
      this.cur = parseSplq(this.bufs[i]!);
      this.slot = this.renderer.uploadQ(this.cur, this.slot);
      this.slotFrame = i;
    }
    return { g: this.slot!, t: i };
  }
  evalCheck(_g: GpuSource, _t: number, got: Float32Array[], ids: number[]) {
    let maxErr = 0;
    ids.forEach((i, k) => {
      const flat = evalSplq(this.cur!, i).flat();
      const gv = [got[k][0], got[k][1], got[k][2], got[k][4], got[k][5], got[k][6], got[k][7], got[k][8], got[k][9], got[k][10], got[k][11], got[k][12], got[k][13], got[k][14]];
      flat.forEach((v, j) => { maxErr = Math.max(maxErr, Math.abs(v - gv[j]) / Math.max(1, Math.abs(v))); });
    });
    return { splats: ids.length, max_rel_err: maxErr };
  }
  destroy() { if (this.slot) this.renderer.release(this.slot); super.destroy(); }
}

class GtPane extends Pane {
  imgs: HTMLImageElement[] = []; cam: TestCam; g2: CanvasRenderingContext2D; loaded = 0;
  device: GPUDevice; tex: GPUTexture | null = null; texFrame = -1;
  constructor(scene: SceneSpec, cam: TestCam, grid: HTMLElement, base: URL, device: GPUDevice) {
    super({ id: 'gt', title: scene.gt?.title || 'GT', rate_label: `${tr('Testkamera', 'test camera')} ${cam.name}` }, grid);
    grid.prepend(this.el);
    this.cam = cam; this.device = device;
    this.g2 = this.canvas.getContext('2d')!;
    const gt = scene.gt!;
    const n = Math.max(scene.frames, 1);
    this.bytesTotal = n;
    for (let f = 0; f < n; f++) {
      const im = new Image();
      im.onload = () => { this.loaded++; this.bytesLoaded++; if (this.loaded === n) this.loadedMs = performance.now(); };
      im.onerror = () => { this.status = tr(`GT f${f} fehlt`, `GT f${f} missing`); };
      im.src = new URL(`${gt.dir}/c${String(cam.idx).padStart(2, '0')}/f${String(f).padStart(3, '0')}.${gt.ext}`, base).href;
      this.imgs.push(im);
    }
  }
  progress() { return this.loaded / Math.max(this.imgs.length, 1); }
  updateLabel(extra = '') {
    super.updateLabel(extra);
    if (this.progress() < 1) this.label.querySelector('.st')!.textContent = tr(` · lädt ${this.loaded}/${this.imgs.length} Bilder`, ` · loading ${this.loaded}/${this.imgs.length} images`);
  }
  loadedAll() { return this.loaded === this.imgs.length; }
  ready(f: number) { const im = this.imgs[Math.min(Math.floor(f), this.imgs.length - 1)]; return !!im && im.complete && im.naturalWidth > 0; }
  /** the GT still of frame f as a GPU texture (one slot, re-uploaded when the frame changes) + its fit rect in a w×h pane */
  diffRef(f: number, w: number, h: number, gain: number): DiffRef | null {
    if (!this.ready(f)) return null;
    const fi = Math.min(Math.floor(f), this.imgs.length - 1), im = this.imgs[fi];
    if (!this.tex || this.tex.width !== im.naturalWidth || this.tex.height !== im.naturalHeight) {
      this.tex?.destroy();
      this.tex = this.device.createTexture({ size: [im.naturalWidth, im.naturalHeight], format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
      this.texFrame = -1;
    }
    if (this.texFrame !== fi) {
      this.device.queue.copyExternalImageToTexture({ source: im }, { texture: this.tex }, [im.naturalWidth, im.naturalHeight]);
      this.texFrame = fi;
    }
    const r = fitRect(this.cam, w, h);
    return { tex: this.tex, ox: r.ox, oy: r.oy, dw: r.dw, dh: r.dh, gain };
  }
  destroy() { this.tex?.destroy(); this.tex = null; super.destroy(); }
  render(f: number, w: number, h: number) {
    if (!this.ready(f)) return false;
    const r = fitRect(this.cam, w, h);
    this.g2.fillStyle = '#000'; this.g2.fillRect(0, 0, w, h);
    this.g2.drawImage(this.imgs[Math.min(Math.floor(f), this.imgs.length - 1)], r.ox, r.oy, r.dw, r.dh);
    this.last = { f: Math.floor(f), t: Math.floor(f), cam: `gt:${this.cam.name}` };
    this.renders++;
    if (!this.firstFrameMs) this.firstFrameMs = performance.now();
    return true;
  }
}

// ---------------------------------------------------------------- app
async function main() {
  if (!navigator.gpu) fail(tr('kein WebGPU (navigator.gpu fehlt) – Safari ≥ 26 / Chrome ≥ 113 nötig',
    'no WebGPU (navigator.gpu missing) – needs Safari ≥ 26 / Chrome ≥ 113'));
  // Some laptops (hybrid graphics, Brave with shields) return null for 'high-performance' but hand out a default
  // adapter, so try every preference before giving up.
  const prefs: GPURequestAdapterOptions[] = [{ powerPreference: 'high-performance' }, {}, { powerPreference: 'low-power' },
    { forceFallbackAdapter: true } as GPURequestAdapterOptions];
  let adapter: GPUAdapter | null = null;
  let usedPref = '';
  for (let i = 0; i < 10 && !adapter; i++) {
    for (const opt of prefs) {
      adapter = await navigator.gpu.requestAdapter(opt);
      if (adapter) { usedPref = opt.powerPreference ?? 'default'; break; }
    }
    if (!adapter) await new Promise((r) => setTimeout(r, 200));
  }
  if (!adapter) {
    fail(tr('kein WebGPU-Adapter: requestAdapter() gab für high-performance, default, low-power und den Software-Fallback null zurück. '
      + 'In Brave/Chromium auf Linux meist: Vulkan nicht aktiv (brave://flags/#enable-vulkan auf Enabled, Neustart) oder brave://gpu zeigt „Vulkan: Disabled“. '
      + 'Details stehen in der Konsole (F12).',
      'no WebGPU adapter: requestAdapter() returned null for high-performance, default, low-power and the software fallback. '
      + 'On Brave/Chromium under Linux this is usually Vulkan not enabled (brave://flags/#enable-vulkan → Enabled, restart) or brave://gpu showing "Vulkan: Disabled". '
      + 'Details are in the console (F12).'));
    console.error('WebGPU: no adapter for any powerPreference, forceFallbackAdapter or after 10 retries');
    return;
  }
  console.info(`WebGPU adapter: ${usedPref}`);
  const hasTs = adapter.features.has('timestamp-query') && Q.get('ts') !== '0';
  const device = await adapter.requestDevice({
    requiredFeatures: hasTs ? ['timestamp-query'] : [],
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
  });
  device.lost.then((i) => { P9.error = `device lost (${i.reason}): ${i.message}`; $('status').textContent = P9.error; });
  device.addEventListener('uncapturederror', (e) => { const m = (e as GPUUncapturedErrorEvent).error.message; console.error('WebGPU: ' + m); P9.error ??= m; });
  const format = navigator.gpu.getPreferredCanvasFormat();
  const info = adapter.info ?? ({} as GPUAdapterInfo);
  marks.device_ms = performance.now();

  const manUrl = new URL(Q.get('manifest') ?? 'manifest.json', location.href);
  const man: { scenes: SceneSpec[] } = await (await fetch(manUrl, { cache: 'no-cache' })).json();
  marks.manifest_ms = performance.now();
  const mobile = matchMedia('(max-width: 760px), (pointer: coarse)').matches;
  const dprCap = Number(Q.get('dpr') ?? (mobile ? 2 : 2));
  const grid = $('grid');

  // ---------------- state
  let scene: SceneSpec = man.scenes.find((s) => s.id === Q.get('scene')) ?? man.scenes[0];
  let panes: SplatPane[] = [];
  let gtPane: GtPane | null = null;
  let snap = -1;                                              // index into scene.testcams, -1 = free orbit
  const view: View = { az: 0, elev: 0, zoom: 1, panX: 0, panY: 0 };
  let frame = 0, playing = false, speed = 1, interp = false, solo: string | null = null;
  let diffGain = Number(Q.get('diff') ?? 0);                  // 0 = render, >0 = |ours − Original| × gain (snap mode with GT only)
  let dirty = true, stalls = 0, buffering = false, autoplay = !Q.has('pause');
  // ?sync=1: old lockstep (every pane must have the frame, else all wait; benches). Default: ONE clock that starts as
  // soon as the first pane can play through; every pane that has the frame runs along in sync, a pane that runs out of
  // data stops on its last frame and rejoins when the loop restarts.
  const SYNC = Q.get('sync') === '1';
  // demo=orbit: wait until every pane is fully loaded, play loop 1 at the start view (test camera with snap=…), then from
  // the first repetition on sweep the camera around (orbit_amp degrees, ≥180 = full turn; orbit_period seconds).
  // layout=rows stacks the panes vertically, gt=0 hides the original pane, ui=0 hides the control bar.
  const DEMO = Q.get('demo') === 'orbit';
  const ROWS = Q.get('layout') === 'rows';
  const NO_GT = Q.get('gt') === '0';
  const ORBIT_AMP = Number(Q.get('orbit_amp') ?? 60), ORBIT_PERIOD = Math.max(2, Number(Q.get('orbit_period') ?? 14));
  if (Q.get('ui') === '0') ($('bar') as HTMLElement).style.display = 'none';
  let loops = 0, orbitT = 0, orbitOn = false, orbitElev0 = 0;
  let clockOn = false;
  if (!SYNC) { playing = autoplay; autoplay = false; }
  const seek = (f: number) => {
    frame = f;
    for (const p of allPanes()) { p.pf = f; p.buffering = false; p.started = clockOn && p.ready(f); }
    dirty = true;
  };
  const forceRedraw = Q.get('redraw') === 'always';
  let enabled = new Set<string>();

  const sceneSel = $<HTMLSelectElement>('scene'), camSel = $<HTMLSelectElement>('cam'), slider = $<HTMLInputElement>('slider');
  const diffSel = $<HTMLSelectElement>('diff');
  diffSel.value = String(diffGain); if (diffSel.value !== String(diffGain)) { diffGain = 0; diffSel.value = '0'; }
  man.scenes.forEach((s) => sceneSel.add(new Option(s.title, s.id)));

  const resetView = () => { view.az = 0; view.elev = (scene.orbit.elev_deg * Math.PI) / 180; view.zoom = 1; view.panX = 0; view.panY = 0; dirty = true; };

  // keep existing panes (their data stays loaded), create missing ones, drop disabled ones; GT pane follows the snap camera
  function buildPanes() {
    const base = manUrl;
    for (const p of [...panes]) if (!enabled.has(p.spec.id)) { p.destroy(); panes = panes.filter((x) => x !== p); }
    const next: SplatPane[] = [];
    for (const ps of scene.panes) {
      if (!enabled.has(ps.id)) continue;
      next.push(panes.find((p) => p.spec.id === ps.id) ??
        (faltKind(ps.kind) ? new FaltPane(ps, grid, device, format, hasTs, base, () => interp)
          : ps.kind === 'splq' ? new SplqPane(ps, grid, device, format, hasTs, base)
          : (() => { throw new Error(`pane ${ps.id}: unknown kind ${JSON.stringify(ps.kind)} (falt, csgs or splq)`); })()));
    }
    for (const p of next) if (!panes.includes(p)) p.pf = frame;     // a new pane joins at the next loop start
    panes = next;
    for (const p of panes) grid.appendChild(p.el);                  // manifest order
    const want = !NO_GT && snap >= 0 && scene.gt && scene.gt.cams.includes(scene.testcams[snap].idx) ? scene.testcams[snap] : null;
    if (gtPane && gtPane.cam !== want) { gtPane.destroy(); gtPane = null; }
    if (want && !gtPane) { gtPane = new GtPane(scene, want, grid, base, device); gtPane.pf = frame; }
    if (gtPane) grid.prepend(gtPane.el);
    diffSel.disabled = !gtPane;
    if (!gtPane && diffGain) { diffGain = 0; diffSel.value = '0'; }
    if (solo && ![...panes, ...(gtPane ? [gtPane] : [])].some((p) => p.spec.id === solo)) solo = null;
    layout();
    dirty = true;
  }

  function setScene(id: string, paneIds?: string[]) {
    scene = man.scenes.find((s) => s.id === id) ?? man.scenes[0];
    sceneSel.value = scene.id;
    renderCfg.near = scene.near ?? 0.01;
    for (const p of panes) p.destroy();
    panes = []; gtPane?.destroy(); gtPane = null; solo = null;
    const def = scene.panes.filter((p) => !(p.default_off || (mobile && scene.panes.length > 2 && p !== scene.panes[0] && p !== scene.panes[1]))).map((p) => p.id);
    enabled = new Set(paneIds ?? def);
    const box = $('panesel'); box.innerHTML = '';
    for (const p of scene.panes) {
      const l = document.createElement('label');
      const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = enabled.has(p.id);
      cb.onchange = () => { if (cb.checked) enabled.add(p.id); else enabled.delete(p.id); buildPanes(); };
      l.append(cb, ' ' + p.title); box.appendChild(l);
    }
    camSel.innerHTML = '';
    camSel.add(new Option(tr('Kamera: frei (Orbit)', 'Camera: free (orbit)'), '-1'));
    scene.testcams.forEach((c, i) => camSel.add(new Option(`Snap ${c.name}${scene.gt?.cams.includes(c.idx) ? ' + GT' : ''}`, String(i))));
    snap = -1; camSel.value = '-1';
    slider.max = String(scene.frames - 1);
    frame = 0; resetView();
    $('note').textContent = scene.note;
    buildPanes();
  }

  function setSnap(i: number) {
    snap = i; camSel.value = String(i);
    buildPanes();
  }

  // grid: pick the column count that shows portrait content (aspect of the test cameras) largest
  function layout() {
    const all: Pane[] = [...(gtPane ? [gtPane] : []), ...panes];
    const shown = solo ? all.filter((p) => p.spec.id === solo) : all;
    for (const p of all) { p.visible = shown.includes(p); p.el.style.display = p.visible ? '' : 'none'; }
    const W = grid.clientWidth, H = grid.clientHeight, n = Math.max(shown.length, 1);
    const tc = scene.testcams[0], asp = tc ? tc.w / tc.h : 1;
    let best = 1, bestS = 0;
    for (let c = 1; c <= (ROWS ? 1 : n); c++) {
      const r = Math.ceil(n / c), w = W / c, h = H / r;
      const s = Math.min(w / asp, h);
      if (s > bestS + 1e-6) { bestS = s; best = c; }
    }
    grid.style.gridTemplateColumns = `repeat(${best}, 1fr)`;
    grid.style.gridTemplateRows = `repeat(${Math.ceil(n / best)}, 1fr)`;
    requestAnimationFrame(() => {
      const dpr = Math.min(devicePixelRatio || 1, dprCap);
      for (const p of all) {
        const w = Math.max(1, Math.round(p.canvas.clientWidth * dpr)), h = Math.max(1, Math.round(p.canvas.clientHeight * dpr));
        if (p.canvas.width !== w || p.canvas.height !== h) { p.canvas.width = w; p.canvas.height = h; }
      }
      dirty = true;
    });
  }
  addEventListener('resize', layout);

  const camFor = (w: number, h: number): Camera => (snap >= 0 ? snapCam(scene.testcams[snap], w, h) : orbitCam(scene.orbit, view, w, h, scene.testcams[0] ? scene.testcams[0].w / scene.testcams[0].h : 0));

  // ---------------- render loop (one rAF for all panes: same camera, same frame)
  let lastNow = 0, inflight = 0, rafCount = 0;
  const fpsWin: number[] = [];
  let rafOn = true;
  const allPanes = (): Pane[] => [...(gtPane ? [gtPane] : []), ...panes];
  const tick = (now: number) => {
    if (!rafOn) return;
    requestAnimationFrame(tick);
    const dt = lastNow ? Math.min((now - lastNow) / 1000, 0.25) : 0;
    lastNow = now; rafCount++;
    const ps = allPanes();
    if (SYNC) {
      if (autoplay && ps.length && ps.every((p) => p.ready(0))) { autoplay = false; playing = true; marks.all_first_ready_ms = performance.now(); }
      if (playing) {
        let nf = frame + dt * scene.fps * speed;
        if (nf >= scene.frames) nf -= scene.frames;
        // shared timeline: advance only when every pane has the frame (else hold = stall, panes stay in sync)
        if (ps.every((p) => p.ready(nf))) { if (Math.floor(nf) !== Math.floor(frame) || interp) dirty = true; frame = nf; buffering = false; } else if (!buffering) { buffering = true; stalls++; }
      }
      for (const p of ps) { p.pf = frame; if (playing) p.started = true; }
    } else {
      const fps = scene.fps * speed;
      if (!clockOn && (DEMO ? ps.length > 0 && ps.every((p) => p.loadedAll()) : ps.some((p) => p.canPlayThrough(fps, scene.frames)))) {
        clockOn = true; marks.clock_start_ms = performance.now();
        for (const p of ps) { p.started = p.ready(frame); p.buffering = false; p.pf = frame; }
        dirty = true;
      }
      if (playing && clockOn) {
        let nf = frame + dt * fps;
        const wrap = nf >= scene.frames;
        if (wrap) {                                   // loop restart: every pane that has frame 0 is back in
          nf -= scene.frames;
          loops++;
          if (DEMO && loops === 1 && !orbitOn) {            // first repetition: leave the test camera, start the sweep
            if (snap >= 0) setSnap(-1);
            resetView(); orbitElev0 = view.elev; orbitT = 0; orbitOn = true;
          }
          for (const p of ps) { p.buffering = false; p.started = p.ready(nf); if (p.started) p.pf = nf; }
          dirty = true;
        }
        const live = ps.filter((p) => p.started && !p.buffering);
        if (!wrap && live.length && !live.some((p) => p.ready(nf))) {
          if (!buffering) { buffering = true; stalls++; }   // nobody can show the next frame: the clock waits
        } else {
          buffering = false;
          if (Math.floor(nf) !== Math.floor(frame) || interp) dirty = true;
          frame = nf;
          for (const p of live) {
            if (p.ready(nf)) p.pf = nf;
            else { p.buffering = true; p.stalls++; dirty = true; }   // out of data: hold the last frame until the loop restarts
          }
        }
      }
    }
    if (orbitOn && playing) {
      orbitT += dt;
      const ph = (2 * Math.PI * orbitT) / ORBIT_PERIOD;
      view.az = ORBIT_AMP >= 180 ? ph : ((ORBIT_AMP * Math.PI) / 180) * Math.sin(ph);
      view.elev = orbitElev0 + ((8 * Math.PI) / 180) * Math.sin(ph / 2);
      dirty = true;
    }
    if ((dirty || forceRedraw) && inflight < 2 && ps.length) {
      let any = false;
      for (const p of panes) {
        if (!p.visible) continue;
        const f = Math.floor(p.pf), tt = interp ? p.pf : f;
        const diff = diffGain > 0 && gtPane ? gtPane.diffRef(f, p.canvas.width, p.canvas.height, diffGain) : null;
        any = p.render(f, tt, camFor(p.canvas.width, p.canvas.height), { diff }) || any;
      }
      if (gtPane?.visible) gtPane.render(Math.floor(gtPane.pf), gtPane.canvas.width, gtPane.canvas.height);
      if (any) { inflight++; device.queue.onSubmittedWorkDone().then(() => { inflight--; }); }
      if (ps.every((p) => !p.visible || p.last?.f === Math.floor(p.pf))) dirty = false;
      if (diffGain > 0 && gtPane && !gtPane.ready(Math.floor(gtPane.pf))) dirty = true;   // difference view waits for the GT still
    }
    fpsWin.push(now);
    while (fpsWin.length && fpsWin[0] < now - 1000) fpsWin.shift();
    if (rafCount % 10 === 0) {
      slider.value = String(Math.floor(frame));
      $('frame').textContent = `${Math.floor(frame) + 1}/${scene.frames}`;
      $('play').textContent = playing ? '❚❚' : '▶';
      const clk = (p: Pane) => SYNC ? '' : !clockOn ? tr(' · wartet auf Puffer', ' · waiting for buffer')
        : !p.started ? tr(' · steigt beim nächsten Loop ein', ' · joins at next loop')
        : p.buffering ? tr(` · angehalten bei f${Math.floor(p.pf) + 1} bis zum Loop`, ` · paused at f${Math.floor(p.pf) + 1} until loop`) : '';
      for (const p of panes) {
        const tm = p.renderer.lastTimings;
        p.updateLabel(` · ${(p.splats() / 1000).toFixed(0)}K Splats${tm ? ` · GPU ${tm.gpu_total.toFixed(1).replace('.', ',')} ms` : ''}` +
          (diffGain > 0 && gtPane ? ` · |ours − ${gtPane.spec.title}| ×${diffGain}` : '') + clk(p));
      }
      gtPane?.updateLabel(clk(gtPane));
      $('status').textContent = (DEMO ? (!clockOn ? tr('Demo: lädt alle Ansichten … · ', 'Demo: loading all views … · ') : !orbitOn ? tr('Durchlauf 1: Testkamera · ', 'Loop 1: test camera · ') : tr('ab Wiederholung 1: Rundumblick · ', 'from repeat 1: orbit · ')) : '') + `${fpsWin.length} fps${buffering ? tr(' · puffert …', ' · buffering …') : stalls ? ` · ${stalls}× ${tr('gepuffert', 'buffered')}` : ''}`;
    }
  };
  requestAnimationFrame(tick);

  // ---------------- controls
  const step = (d: number) => { playing = false; seek((Math.floor(frame) + d + scene.frames) % scene.frames); };
  $('play').onclick = () => { playing = !playing; autoplay = false; };
  $('prev').onclick = () => step(-1);
  $('next').onclick = () => step(1);
  slider.oninput = () => { playing = false; autoplay = false; seek(Number(slider.value)); };
  $<HTMLSelectElement>('speed').onchange = (e) => { speed = Number((e.target as HTMLSelectElement).value); };
  $<HTMLInputElement>('interp').onchange = (e) => { interp = (e.target as HTMLInputElement).checked; dirty = true; };
  camSel.onchange = () => setSnap(Number(camSel.value));
  diffSel.onchange = () => { diffGain = Number(diffSel.value); dirty = true; };
  sceneSel.onchange = () => setScene(sceneSel.value);
  $('reset').onclick = () => { if (snap >= 0) setSnap(-1); resetView(); };
  addEventListener('keydown', (ev) => {
    const e = ev as KeyboardEvent;
    if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'SELECT') return;
    if (e.key === ' ') { playing = !playing; e.preventDefault(); } else if (e.key === 'ArrowLeft') step(-1); else if (e.key === 'ArrowRight') step(1);
  });
  // pointer: 1 finger / mouse = orbit, 2 fingers = pinch zoom + pan, double tap = reset, tap on a title = solo pane
  const pts = new Map<number, { x: number; y: number }>();
  let pinch: { d: number; mx: number; my: number } | null = null, lastTap = 0;
  let labelTap: { id: string; x: number; y: number; t: number } | null = null;   // tap on a pane title = solo toggle
  const leaveSnap = () => { if (snap >= 0) { setSnap(-1); } };
  grid.addEventListener('pointerdown', (e) => {
    const lab = (e.target as HTMLElement).closest('.label');
    labelTap = lab ? { id: (lab.parentElement as HTMLElement).dataset.id!, x: e.clientX, y: e.clientY, t: now() } : null;
    grid.setPointerCapture(e.pointerId);
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pts.size === 2) {
      const [a, b] = [...pts.values()];
      pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 };
    }
    if (pts.size === 1) {
      if (now() - lastTap < 300) { leaveSnap(); resetView(); }
      lastTap = now();
    }
  });
  const now = () => performance.now();
  grid.addEventListener('pointermove', (e) => {
    const p = pts.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y;
    p.x = e.clientX; p.y = e.clientY;
    if (pts.size === 1 && (dx || dy)) {
      leaveSnap();
      view.az -= dx * 0.008; view.elev = Math.max(-1.4, Math.min(1.4, view.elev + dy * 0.008)); dirty = true;
    } else if (pts.size === 2 && pinch) {
      const [a, b] = [...pts.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y), mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
      leaveSnap();
      view.zoom = Math.max(0.15, Math.min(6, view.zoom * pinch.d / Math.max(d, 1)));
      const ph = (panes[0]?.canvas.clientHeight || 400);
      view.panX -= (mx - pinch.mx) / ph * 0.8 * view.zoom; view.panY -= (my - pinch.my) / ph * 0.8 * view.zoom;
      pinch = { d, mx, my }; dirty = true;
    }
  });
  const up = (e: PointerEvent) => {
    pts.delete(e.pointerId); if (pts.size < 2) pinch = null;
    if (labelTap && e.type === 'pointerup' && Math.hypot(e.clientX - labelTap.x, e.clientY - labelTap.y) < 8 && now() - labelTap.t < 400) {
      solo = solo === labelTap.id ? null : labelTap.id; layout();
    }
    labelTap = null;
  };
  grid.addEventListener('pointerup', up); grid.addEventListener('pointercancel', up);
  grid.addEventListener('wheel', (e) => { leaveSnap(); view.zoom = Math.max(0.15, Math.min(6, view.zoom * Math.exp(e.deltaY * 0.001))); dirty = true; e.preventDefault(); }, { passive: false });

  setScene(scene.id, Q.get('panes')?.split(','));
  // snap on load: URL `snap=` wins, else the scene's default (e.g. next to the Original at a test camera)
  const snap0 = Q.get('snap') ?? scene.default_snap;
  if (snap0) { const i = scene.testcams.findIndex((c) => c.name === snap0); if (i >= 0) setSnap(i); }
  if (Q.has('frame')) seek(Number(Q.get('frame')));
  // view state from a shared link (🔗): orbit angles in degrees, zoom, pan, speed, in-between frames, solo pane
  const num = (k: string) => { const v = Q.get(k); return v !== null && v !== '' && Number.isFinite(Number(v)) ? Number(v) : null; };
  if (num('az') !== null) view.az = (num('az')! * Math.PI) / 180;
  if (num('elev') !== null) view.elev = (num('elev')! * Math.PI) / 180;
  if (num('zoom') !== null) view.zoom = Math.max(0.15, Math.min(6, num('zoom')!));
  if (num('px') !== null) view.panX = num('px')!;
  if (num('py') !== null) view.panY = num('py')!;
  if (num('speed') !== null) { speed = num('speed')!; $<HTMLSelectElement>('speed').value = String(speed); }
  if (Q.get('interp') === '1') { interp = true; $<HTMLInputElement>('interp').checked = true; }
  if (Q.has('solo')) { solo = Q.get('solo'); layout(); }
  dirty = true;

  // 🔗: link to exactly what is on screen now; clipboard, else share sheet (iOS), else a prompt to copy by hand
  function shareUrl() {
    const u = new URL(location.origin + location.pathname);
    if (Q.has('manifest')) u.searchParams.set('manifest', Q.get('manifest')!);
    u.searchParams.set('scene', scene.id);
    u.searchParams.set('panes', scene.panes.filter((p) => enabled.has(p.id)).map((p) => p.id).join(','));
    u.searchParams.set('snap', snap >= 0 ? scene.testcams[snap].name : 'orbit');
    u.searchParams.set('frame', String(Math.floor(frame)));
    if (snap < 0) {
      const r = (x: number, d: number) => String(Math.round(x * 10 ** d) / 10 ** d);
      u.searchParams.set('az', r((view.az * 180) / Math.PI, 1));
      u.searchParams.set('elev', r((view.elev * 180) / Math.PI, 1));
      u.searchParams.set('zoom', r(view.zoom, 3));
      if (view.panX || view.panY) { u.searchParams.set('px', r(view.panX, 4)); u.searchParams.set('py', r(view.panY, 4)); }
    }
    if (diffGain > 0) u.searchParams.set('diff', String(diffGain));
    if (solo) u.searchParams.set('solo', solo);
    if (speed !== 1) u.searchParams.set('speed', String(speed));
    if (interp) u.searchParams.set('interp', '1');
    if (!playing) u.searchParams.set('pause', '');
    if (SYNC) u.searchParams.set('sync', '1');
    return u.href.replace(/%2C/g, ',').replace('pause=&', 'pause&').replace(/pause=$/, 'pause');
  }
  const linkBtn = $<HTMLButtonElement>('link');
  linkBtn.onclick = async () => {
    const url = shareUrl();
    const done = (t: string) => { linkBtn.textContent = t; setTimeout(() => { linkBtn.textContent = '🔗'; }, 1500); };
    try { await navigator.clipboard.writeText(url); done(tr('✓ kopiert', '✓ copied')); return; } catch { /* no clipboard */ }
    try { if (navigator.share) { await navigator.share({ url }); done('✓'); return; } } catch { /* cancelled */ }
    prompt(tr('Link kopieren:', 'Copy link:'), url);
  };

  // ---------------- bench hooks (Playwright)
  const splatPane = (id: string) => { const p = panes.find((x) => x.spec.id === id); if (!p) throw new Error(`no pane ${id}`); return p; };
  const summarize = (log: Record<string, number>[]) => {
    if (!log.length) return null;
    const out: Record<string, { median: number; p95: number }> = {};
    for (const k of Object.keys(log[0])) {
      const v = log.map((r) => r[k]).filter(Number.isFinite).sort((a, b) => a - b);
      if (v.length) out[k] = { median: v[Math.floor(v.length / 2)], p95: v[Math.floor(v.length * 0.95)] };
    }
    return { samples: log.length, ...out };
  };
  Object.assign(P9, {
    ready: true,
    info: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, timestamps: hasTs, format, mobile, dprCap },
    shareUrl: () => shareUrl(),
    scene: () => ({ id: scene.id, frames: scene.frames, fps: scene.fps, panes: allPanes().map((p) => p.spec.id), snap: snap >= 0 ? scene.testcams[snap].name : null, diff: diffGain }),
    setScene: (id: string, ids?: string[]) => setScene(id, ids),
    setView: (o: { az?: number; elev?: number; zoom?: number; frame?: number; playing?: boolean; snap?: string | null; interp?: boolean; solo?: string | null; diff?: number }) => {
      if (o.az !== undefined) view.az = (o.az * Math.PI) / 180;
      if (o.elev !== undefined) view.elev = (o.elev * Math.PI) / 180;
      if (o.zoom !== undefined) view.zoom = o.zoom;
      if (o.frame !== undefined) seek(o.frame);
      if (o.playing !== undefined) { playing = o.playing; autoplay = false; }
      if (o.interp !== undefined) interp = o.interp;
      if (o.snap !== undefined) setSnap(o.snap === null ? -1 : scene.testcams.findIndex((c) => c.name === o.snap));
      if (o.solo !== undefined) { solo = o.solo; layout(); }
      if (o.diff !== undefined) { diffGain = o.diff; diffSel.value = String(o.diff); }
      dirty = true;
    },
    allLoaded: () => allPanes().every((p) => p.loadedAll()),
    state: () => ({
      frame, playing, stalls, rafCount, sync_mode: SYNC, clock_on: clockOn, demo: DEMO, loops, orbit_on: orbitOn, fps_window: fpsWin.length, view: { ...view }, snap, solo, diff: diffGain,
      panes: allPanes().map((p) => ({ id: p.spec.id, last: p.last, renders: p.renders, canvas: [p.canvas.width, p.canvas.height],
        css: [p.canvas.clientWidth, p.canvas.clientHeight], visible: p.visible, bytes_total: p.bytesTotal, bytes_loaded: p.bytesLoaded,
        first_frame_ms: p.firstFrameMs, loaded_ms: p.loadedMs, rate_label: p.spec.rate_label,
        pf: p.pf, started: p.started, frozen: p.buffering, pane_stalls: p.stalls,
        splats: p instanceof SplatPane ? p.splats() : 0, gpu_bytes: p instanceof SplatPane ? p.renderer.gpuBytes() : 0,
        visible_splats: p instanceof SplatPane ? p.renderer.lastVisible : 0,
        section_timings: p instanceof FaltPane ? p.timings : undefined })),
    }),
    /** rAF-paced playback with every pane redrawn each display frame; per-pane GPU timestamps. */
    async rafFps(seconds: number) {
      for (const p of panes) p.renderer.timingsLog.length = 0;
      const r0 = Object.fromEntries(allPanes().map((p) => [p.spec.id, p.renders])), s0 = stalls, c0 = rafCount;
      playing = true; autoplay = false;
      const t0 = performance.now();
      const keep = setInterval(() => { dirty = true; }, 4);
      await new Promise((r) => setTimeout(r, seconds * 1000));
      clearInterval(keep);
      const el = (performance.now() - t0) / 1000;
      return { seconds: el, raf_fps: (rafCount - c0) / el, stalls: stalls - s0,
        panes: allPanes().map((p) => ({ id: p.spec.id, fps: (p.renders - r0[p.spec.id]) / el,
          gpu: p instanceof SplatPane ? summarize(p.renderer.timingsLog) : null })) };
    },
    /** one pane alone, frames back-to-back into an offscreen texture (max 3 in flight). */
    async uncapped(id: string, nFrames: number) {
      const p = splatPane(id);
      rafOn = false;
      await device.queue.onSubmittedWorkDone();
      const off = device.createTexture({ size: [p.canvas.width, p.canvas.height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
      p.renderer.timingsLog.length = 0;
      const fl: Promise<void>[] = [];
      const t0 = performance.now();
      let fr = frame;
      for (let i = 0; i < nFrames; i++) {
        const f = Math.floor(fr) % scene.frames;
        const s = p.source(f, f);
        if (!s) throw new Error(`${id}: frame ${f} not ready`);
        p.renderer.frame(s.g, s.t, camFor(p.canvas.width, p.canvas.height), off, { timing: true });
        fl.push(device.queue.onSubmittedWorkDone());
        if (fl.length >= 3) await fl.shift();
        fr += scene.fps / 60;
      }
      await Promise.all(fl);
      const el = (performance.now() - t0) / 1000;
      off.destroy();
      rafOn = true; lastNow = 0; dirty = true; requestAnimationFrame(tick);
      return { id, fps: nFrames / el, frames: nFrames, seconds: el, canvas: [p.canvas.width, p.canvas.height], gpu: summarize(p.renderer.timingsLog) };
    },
    /** offline video: render pane `id` at frame f with an orbit view (degrees) or a test camera, w x h -> RGB base64.
     *  Same renderer and camera model as the live viewer (orbitCam with the test-camera aspect). */
    async renderView(id: string, f: number, v: { az?: number; elev?: number; zoom?: number; snap?: string }, w: number, h: number) {
      const p = splatPane(id) as FaltPane | SplqPane;
      rafOn = false;
      await device.queue.onSubmittedWorkDone();
      const s = p.source(f, f);
      if (!s) throw new Error(`${id}: frame ${f} not ready`);
      const tc = scene.testcams[0];
      const cam = v.snap ? snapCam(scene.testcams.find((x) => x.name === v.snap)!, w, h)
        : orbitCam(scene.orbit, { az: ((v.az ?? 0) * Math.PI) / 180, elev: ((v.elev ?? scene.orbit.elev_deg) * Math.PI) / 180,
          zoom: v.zoom ?? 1, panX: 0, panY: 0 }, w, h, tc ? tc.w / tc.h : 0);
      p.renderer.frame(s.g, s.t, cam, null);
      const img = await p.renderer.readTarget();
      let bin = '';
      for (let i = 0; i < img.rgb.length; i += 0x8000) bin += String.fromCharCode(...img.rgb.subarray(i, i + 0x8000));
      return { w: img.w, h: img.h, rgb_b64: btoa(bin) };
    },
    resume() { rafOn = true; lastNow = 0; dirty = true; requestAnimationFrame(tick); },
    /** render pane `id` at test camera `name`, native resolution, frame f -> 8-bit RGB (base64) + GPU/CPU eval check. */
    async renderTest(id: string, name: string, f: number) {
      const p = splatPane(id) as FaltPane | SplqPane;
      rafOn = false;
      await device.queue.onSubmittedWorkDone();
      const c = scene.testcams.find((x) => x.name === name)!;
      const s = p.source(f, f);
      if (!s) throw new Error(`${id}: frame ${f} not ready`);
      p.renderer.frame(s.g, s.t, snapCam(c, c.w, c.h), null);
      const img = await p.renderer.readTarget();
      const n = s.g && (s.g as GpuQFrame).q ? (s.g as GpuQFrame).n : (s.g as GpuSection).ps.n;
      const ids = Array.from({ length: Math.min(256, n) }, (_, k) => Math.floor((k * n) / 256));
      const got = await p.renderer.readWorld(ids);
      const check = p.evalCheck(s.g, s.t, got, ids);
      rafOn = true; lastNow = 0; dirty = true; requestAnimationFrame(tick);
      let bin = '';
      for (let i = 0; i < img.rgb.length; i += 0x8000) bin += String.fromCharCode(...img.rgb.subarray(i, i + 0x8000));
      return { w: img.w, h: img.h, rgb_b64: btoa(bin), eval_check: check, n, t: s.t };
    },
    /** headless screenshots do not capture WebGPU canvases reliably: draw each pane's render target into a 2D canvas on top
     *  (difference view: |render − GT still| computed on the CPU with the same fit rect, like the GPU diff blit). */
    async snapshot() {
      rafOn = false;
      await device.queue.onSubmittedWorkDone();
      const f = Math.floor(frame);
      for (const p of panes) {
        if (!p.visible) continue;
        const img = await p.renderer.readTarget();
        const c2 = document.createElement('canvas'); c2.className = 'snapshot';
        c2.width = img.w; c2.height = img.h;
        const id = new ImageData(img.w, img.h);
        for (let i = 0; i < img.w * img.h; i++) { id.data.set(img.rgb.subarray(i * 3, i * 3 + 3), i * 4); id.data[i * 4 + 3] = 255; }
        const pf = p.last?.f ?? f;
        if (diffGain > 0 && gtPane && gtPane.ready(pf)) {
          const r = fitRect(gtPane.cam, img.w, img.h), g2 = c2.getContext('2d')!;
          g2.fillStyle = '#000'; g2.fillRect(0, 0, img.w, img.h);
          g2.drawImage(gtPane.imgs[Math.min(pf, gtPane.imgs.length - 1)], r.ox, r.oy, r.dw, r.dh);
          const gt = g2.getImageData(0, 0, img.w, img.h).data;
          for (let i = 0; i < img.w * img.h * 4; i++) id.data[i] = i % 4 === 3 ? 255 : Math.min(255, Math.abs(id.data[i] - gt[i]) * diffGain);
        }
        c2.getContext('2d')!.putImageData(id, 0, 0);
        p.el.insertBefore(c2, p.label);
      }
      return true;
    },
    unsnapshot() { document.querySelectorAll('canvas.snapshot').forEach((c) => c.remove()); rafOn = true; lastNow = 0; dirty = true; requestAnimationFrame(tick); },
  });
}

main().catch((e) => { if (P9.error) return; fail(String((e?.message ? e.message + " | " : "") + (e?.stack ?? e))); });
