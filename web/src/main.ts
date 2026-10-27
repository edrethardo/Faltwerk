// CurveStream-GS web player: section streaming (2 sections look-ahead), any display rate, orbit camera, immersive-ar, bench hooks.
import { Renderer, type Camera, type GpuSection } from './renderer.ts';
import { evalSplat, type PackedSection } from './pack.ts';
import { ImmersiveAR, place, type Placement } from './xr.ts';
import { captureRoom, describeGrade, estimateGrade, imageStats, type ImageStats } from './recolor.ts';
import { localizeDom, tr } from './i18n.ts';

interface SceneSection { url: string; g0: number; nframes: number }
interface TestCam { name: string; R: number[][]; t: number[]; fx: number; fy: number; cx: number; cy: number; w: number; h: number }
interface Scene {
  fps: number; sections: SceneSection[];
  orbit: { center: number[]; up: number[]; front: number[]; dist: number; elev_deg: number; fovy_deg: number };
  testcams: TestCam[];
}

const Q = new URLSearchParams(location.search);
const T_NAV = performance.now();             // ms since navigation start at script start
const LOOKAHEAD = 2;
const UPLOAD_SLICE = 8 * 2 ** 20;            // bytes per display frame while the next section streams to the GPU
const marks: Record<string, number> = { script_start_ms: T_NAV };
const overlay = document.getElementById('stats')!;
const canvas = document.getElementById('c') as HTMLCanvasElement;
localizeDom();

const setting = (k: string): string | null => { try { return localStorage.getItem('p5.' + k); } catch { return null; } };
const saveSetting = (k: string, v: string) => { try { localStorage.setItem('p5.' + k, v); } catch { /* private mode */ } };

/** scene*.json files on the server, newest first; empty when the server has no list */
async function sceneList(): Promise<string[]> {
  try {
    const names = await (await fetch('scenes.json', { cache: 'no-store' })).json();
    return Array.isArray(names) ? names : [];
  } catch {
    return [];
  }
}

/** Scene picker; picking one reloads into it. */
function fillScenes(names: string[], current: string): void {
  const sel = document.getElementById('scene') as HTMLSelectElement;
  if (!names.length) return;
  if (!names.includes(current)) names.unshift(current);
  for (const n of names) sel.add(new Option(n.replace(/^scene_?/, '').replace(/\.json$/, '') || 'scene', n, false, n === current));
  sel.hidden = false;
  sel.addEventListener('change', () => {
    saveSetting('scene', sel.value);
    const q = new URLSearchParams(location.search);
    q.delete('scene');
    location.search = q.toString();
  });
}

function fail(msg: string): never {
  overlay.textContent = tr('Fehler: ', 'Error: ') + msg;
  (window as any).__p5 = { error: msg };
  throw new Error(msg);
}

// ---------------------------------------------------------------- cameras (OpenCV: x right, y down, z forward)
function lookAt(pos: number[], target: number[], up: number[]): number[][] {
  const sub = (a: number[], b: number[]) => a.map((v, i) => v - b[i]);
  const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a: number[]) => { const l = Math.hypot(...a); return a.map((v) => v / l); };
  const z = norm(sub(target, pos));
  const x = norm(cross(z, up));
  const y = cross(z, x);
  return [x, y, z];
}

function camFromRt(R: number[][], t: number[], fx: number, fy: number, cx: number, cy: number, w: number, h: number): Camera {
  const view = new Float32Array(16);
  for (let r = 0; r < 3; r++) { for (let c = 0; c < 3; c++) view[c * 4 + r] = R[r][c]; view[12 + r] = t[r]; }
  view[15] = 1;
  const campos = [0, 1, 2].map((c) => -(R[0][c] * t[0] + R[1][c] * t[1] + R[2][c] * t[2])) as [number, number, number];
  return { view, campos, fx, fy, cx, cy, width: w, height: h };
}

function orbitCam(sc: Scene, az: number, elev: number, distScale: number, w: number, h: number): Camera {
  const o = sc.orbit, up = o.up, f = o.front;
  const s = [up[1] * f[2] - up[2] * f[1], up[2] * f[0] - up[0] * f[2], up[0] * f[1] - up[1] * f[0]];
  const ce = Math.cos(elev), se = Math.sin(elev), ca = Math.cos(az), sa = Math.sin(az);
  const d = o.dist * distScale;
  const pos = [0, 1, 2].map((i) => o.center[i] + d * (ce * (ca * f[i] + sa * s[i]) + se * up[i]));
  const R = lookAt(pos, o.center, up);
  const t = R.map((row) => -(row[0] * pos[0] + row[1] * pos[1] + row[2] * pos[2]));
  const fy = (0.5 * h) / Math.tan((o.fovy_deg * Math.PI) / 360);
  return camFromRt(R, t, fy, fy, w / 2, h / 2, w, h);
}

// ---------------------------------------------------------------- setup
async function main() {
  if (!navigator.gpu) fail(tr('kein WebGPU (navigator.gpu fehlt)', 'no WebGPU (navigator.gpu missing)'));
  const xrProblem = await ImmersiveAR.diagnose();
  if (xrProblem) console.info(`MR: ${xrProblem}`);
  const xrReady = xrProblem === null;
  const ask = (xr: boolean) => {
    const p = navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
      ...(xr ? { xrCompatible: true } : {}),
    } as GPURequestAdapterOptions);
    return xr ? p.catch(() => null) : p;
  };
  let adapter: GPUAdapter | null = null;
  for (let i = 0; i < 10 && !adapter; i++) {       // the GPU process may still be starting (seen headless on the GPU host)
    if (xrReady) adapter = await ask(true);
    if (!adapter) adapter = await ask(false);
    if (!adapter) await new Promise((r) => setTimeout(r, 200));
  }
  if (!adapter) fail(tr('kein WebGPU-Adapter', 'no WebGPU adapter'));
  const hasTs = adapter.features.has('timestamp-query') && Q.get('ts') !== '0';
  const device = await adapter.requestDevice({
    requiredFeatures: hasTs ? ['timestamp-query'] : [],
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
  });
  let gpuError: string | null = null;
  device.lost.then((i) => { gpuError = `device lost (${i.reason}): ${i.message}`; console.error(gpuError); });
  device.addEventListener('uncapturederror', (e) => {
    const msg = (e as GPUUncapturedErrorEvent).error.message;
    console.error('WebGPU: ' + msg);
    gpuError ??= msg;
  });
  marks.device_ms = performance.now();
  console.info(`p5: device ${marks.device_ms.toFixed(0)} ms`);
  const info = adapter.info ?? ({} as GPUAdapterInfo);
  const ctx = canvas.getContext('webgpu') as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  const offscreen = Q.get('offscreen') === '1';      // no canvas swap chain (SwiftShader headless on hosts without GPU)
  if (!offscreen) ctx.configure({ device, format, alphaMode: 'opaque' });
  let offTex: GPUTexture | null = null;
  const outTexture = (): GPUTexture => {
    if (!offscreen) return ctx.getCurrentTexture();
    if (!offTex || offTex.width !== canvas.width || offTex.height !== canvas.height) {
      offTex?.destroy();
      offTex = device.createTexture({ size: [canvas.width, canvas.height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
    return offTex;
  };
  const dpr = Number(Q.get('dpr') ?? Math.min(devicePixelRatio || 1, 2));
  const resize = () => {
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  };
  resize();
  addEventListener('resize', resize);
  const renderer = new Renderer(device, format, hasTs);
  const bgInput = document.getElementById('bg') as HTMLInputElement;
  const setBg = (hex: string) => {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex);
    if (!m) return;
    const [r, g, b] = [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16) / 255);
    renderer.setBackground(r, g, b);
    bgInput.value = '#' + m[1].toLowerCase();
  };
  setBg(Q.get('bg') ?? setting('bg') ?? '#000000');
  bgInput.addEventListener('input', () => { setBg(bgInput.value); saveSetting('bg', bgInput.value); });

  const scenes = await sceneList();
  const saved = setting('scene');
  const sceneName = Q.get('scene') ?? (saved && scenes.includes(saved) ? saved : scenes[0] ?? 'scene.json');
  const sceneUrl = new URL(sceneName, location.href);
  fillScenes(scenes, sceneName);
  const scene: Scene = await (await fetch(sceneUrl)).json();
  marks.scene_ms = performance.now();
  const nsec = scene.sections.length;
  const total = scene.sections.reduce((a, s) => a + s.nframes, 0);
  const targetN = Number(Q.get('n') ?? 0);

  // ------------------------------------------------------------ streaming
  const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  const decoded = new Map<number, { ps: PackedSection; timing: any }>();
  const gpu = new Map<number, GpuSection>();
  const requested = new Set<number>();
  const timings: any[] = [];
  const waiters = new Map<number, (() => void)[]>();
  worker.onmessage = (e) => {
    const { seq, ps, timing, error } = e.data;
    if (error) fail(`${tr('Abschnitt', 'Section')} ${seq}: ${error}`);
    timing.seq = seq; timing.index = seq % nsec; timing.recv_ms = performance.now();
    timings.push(timing);
    decoded.set(seq, { ps, timing });
    maxDecoded = Math.max(maxDecoded, seq);
    (waiters.get(seq) ?? []).forEach((f) => f());
    waiters.delete(seq);
  };
  // Short clips loop out of GPU memory: once every section is uploaded and fits the budget, nothing is decoded again.
  const residentMiB = Number(Q.get('resident') ?? 1024);
  let resident: Map<number, GpuSection> | null = residentMiB > 0 ? new Map() : null;
  let residentBytes = 0;
  const residentFull = () => !!resident && resident.size === nsec;
  const isResident = (seq: number, g: GpuSection) => !!resident && resident.get(seq % nsec) === g;
  const keep = (seq: number, g: GpuSection) => {
    if (!resident || resident.has(seq % nsec)) return;
    resident.set(seq % nsec, g);
    residentBytes += g.bytes;
    if (residentBytes <= residentMiB * 2 ** 20) return;
    const live = new Set(gpu.values());
    for (const r of resident.values()) if (!live.has(r)) renderer.release(r);
    resident = null;
    console.info(`p5: clip exceeds ${residentMiB} MiB, streaming`);
  };
  let maxRequested = -1;
  const request = (seq: number) => {
    if (requested.has(seq) || residentFull()) return;
    for (let s = maxRequested + 1; s < seq; s++) request(s);      // the decoder is sequential: never skip a section
    maxRequested = Math.max(maxRequested, seq);
    requested.add(seq);
    worker.postMessage({ seq, url: new URL(scene.sections[seq % nsec].url, sceneUrl).href, targetN });
  };
  let maxDecoded = -1, curSeq = 0;
  const whenDecoded = (seq: number) => new Promise<void>((res) => {
    request(seq);
    if (seq <= maxDecoded || residentFull()) res(); else waiters.set(seq, [...(waiters.get(seq) ?? []), res]);
  });
  /** `now`: the section is drawn this frame, so finish its upload. Otherwise it streams in a slice per frame. */
  const ensureGpu = (seq: number, now = true): GpuSection | null => {
    let g = gpu.get(seq) ?? resident?.get(seq % nsec) ?? null;
    if (!g) {
      const d = decoded.get(seq);
      if (!d) return null;
      console.info(`p5: upload seq ${seq} n=${d.ps.n} at ${performance.now().toFixed(0)} ms`);
      g = renderer.upload(d.ps);
    }
    gpu.set(seq, g);
    if (g.pending) {
      const t0 = performance.now();
      renderer.pump(g, now ? Infinity : UPLOAD_SLICE);
      const d = decoded.get(seq);
      if (d) d.timing.upload_ms = (d.timing.upload_ms ?? 0) + performance.now() - t0;
      if (g.pending) return null;
      keep(seq, g);
    }
    return g;
  };
  // playback position -> (seq, frame time within the stream)
  const locate = (playFrames: number) => {
    const loop = Math.floor(playFrames / total);
    let f = playFrames - loop * total;
    for (let i = 0; i < nsec; i++) {
      const s = scene.sections[i];
      if (f < s.nframes || i === nsec - 1) return { seq: loop * nsec + i, t: s.g0 + Math.min(f, s.nframes - 1e-4) };
      f -= s.nframes;
    }
    throw new Error('unreachable');
  };
  const housekeeping = (seq: number) => {
    curSeq = seq;
    for (let s = seq; s <= seq + LOOKAHEAD; s++) request(s);       // current + 2 sections look-ahead (decode)
    ensureGpu(seq + 1, false);                                      // GPU: current + next
    for (const [k, g] of gpu) if (k < seq) { if (!isResident(k, g)) renderer.release(g); gpu.delete(k); }
    for (const k of decoded.keys()) if (k < seq) decoded.delete(k);
  };

  // ------------------------------------------------------------ render loop
  const fixedT = Q.has('t') ? Number(Q.get('t')) : null;
  let playing = !Q.has('pause') && fixedT === null;
  let playFrames = fixedT ?? 0;
  let lastNow = 0, stalls = 0, frames = 0, firstFrameDone = false, inflight = 0, gpuBound = 0;
  let az = Number(Q.get('az') ?? 0) * Math.PI / 180, elev = (scene.orbit.elev_deg * Math.PI) / 180, distScale = Number(Q.get('zoom') ?? 1);
  const fpsWin: number[] = [];
  const testIdx = Q.has('testcam') ? Number(Q.get('testcam')) : -1;
  const currentCam = (): Camera => {
    if (testIdx >= 0) {
      const c = scene.testcams[testIdx];
      return camFromRt(c.R, c.t, c.fx, c.fy, c.cx, c.cy, c.w, c.h);
    }
    return orbitCam(scene, az, elev, distScale, canvas.width, canvas.height);
  };

  let rafOn = true;
  let hud = () => {};
  const markFps = (now: number) => {
    fpsWin.push(now);
    while (fpsWin.length && fpsWin[0] < now - 1000) fpsWin.shift();
  };
  const statusText = (seq: number, t: number, n: number) => {
    const tm = renderer.lastTimings;
    return `${fpsWin.length} fps · ${tr('Abschnitt', 'section')} ${seq % nsec} t=${t.toFixed(2)} · ${n} splats (${renderer.lastVisible} ${tr('sichtbar', 'visible')})` +
      (tm ? ` · GPU ${tm.gpu_total.toFixed(2)} ms` : '') + ` · ${(renderer.gpuBytes() / 2 ** 20).toFixed(0)} MiB`;
  };
  const tick = (now: number) => {
    if (gpuError) { overlay.textContent = tr('GPU-Fehler: ', 'GPU error: ') + gpuError; (window as any).__p5.error = gpuError; return; }
    if (!rafOn) return;
    const dt = lastNow ? (now - lastNow) / 1000 : 0;
    lastNow = now;
    let { seq, t } = locate(playFrames);
    housekeeping(seq);
    const g = ensureGpu(seq);
    if (g && inflight >= 2) {
      gpuBound++;                                   // GPU still busy with 2 frames: skip this display frame
      if (playing) playFrames += dt * scene.fps;
    } else if (g) {
      renderer.frame(g, t, currentCam(), outTexture(), { timing: true, readVisible: frames % 10 === 0 });
      inflight++;
      device.queue.onSubmittedWorkDone().then(() => { inflight--; });
      frames++;
      if (!firstFrameDone) {
        firstFrameDone = true;
        console.info(`p5: first frame submitted ${performance.now().toFixed(0)} ms`);
        device.queue.onSubmittedWorkDone().then(() => { marks.first_frame_ms = performance.now(); console.info(`p5: first frame done ${marks.first_frame_ms.toFixed(0)} ms`); });
      }
      if (playing) playFrames += dt * scene.fps;
    } else if (playing) {
      stalls++;
    }
    markFps(now);
    hud = () => { overlay.textContent = statusText(seq, t, g ? g.ps.n : 0); };
    if (frames % 15 === 0) hud();
    void t; void seq;
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);

  // orbit controls (pointer + wheel; works with touch)
  let drag: { x: number; y: number } | null = null;
  canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointerup', () => { drag = null; });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    az -= (e.clientX - drag.x) * 0.01; elev = Math.max(-1.4, Math.min(1.4, elev + (e.clientY - drag.y) * 0.01));
    drag = { x: e.clientX, y: e.clientY };
  });
  canvas.addEventListener('wheel', (e) => { distScale *= Math.exp(e.deltaY * 0.001); e.preventDefault(); }, { passive: false });
  document.getElementById('play')!.addEventListener('click', () => { playing = !playing; });

  const xrScaleQ = Number(Q.get('xrscale') ?? 1);
  const xrLayerScale = Number.isFinite(xrScaleQ) && xrScaleQ > 0 ? xrScaleQ : 1;
  const xrDistQ = Number(Q.get('xrdist') ?? 1.5);
  const xrDist = Number.isFinite(xrDistQ) && xrDistQ > 0 ? xrDistQ : 1.5;
  const xrButton = document.getElementById('xr-ar') as HTMLButtonElement;
  const recolorBox = document.getElementById('recolor') as HTMLInputElement;
  const recolorInfo = document.getElementById('recolor-info')!;
  const recolorQ = Number(Q.get('recolor') ?? 0.6);
  const recolorStrength = Number.isFinite(recolorQ) && recolorQ >= 0 ? recolorQ : 0.6;
  /** Runs inside a live session only: the passthrough camera is not exposed before that. */
  const applyRecolor = async (stage: ImmersiveAR) => {
    renderer.setGrade(null);
    recolorInfo.textContent = tr('Kamera …', 'Camera …');
    let room: ImageStats;
    try {
      room = await captureRoom();
    } catch (e) {
      console.warn('MR: recolor', e);
      recolorInfo.textContent = tr('Kamera: ', 'Camera: ') + (e instanceof Error ? e.message : e);
      return;
    }
    if (xr !== stage || !recolorBox.checked) { recolorInfo.textContent = ''; return; }
    const px = await renderer.readTargetRGBA(4);
    const grade = estimateGrade(room, px ? imageStats(px, 1, true) : null, recolorStrength);
    renderer.setGrade(grade);
    recolorInfo.textContent = describeGrade(grade);
    console.info(`MR: recolor ${describeGrade(grade)}`);
  };
  recolorBox.addEventListener('change', () => {
    if (recolorBox.checked) { if (xr) void applyRecolor(xr); return; }
    renderer.setGrade(null);
    recolorInfo.textContent = '';
  });
  let xr: ImmersiveAR | null = null;
  let placed: Placement | null = null;
  let xrFrames = 0;
  let xrDrawError = false;
  const onXR = (time: number, frame: XRFrame) => {
    const stage = xr;
    if (!stage) return;
    stage.session.requestAnimationFrame(onXR);
    if (gpuError) { overlay.textContent = tr('GPU-Fehler: ', 'GPU error: ') + gpuError; (window as any).__p5.error = gpuError; return; }
    const dt = lastNow ? (time - lastNow) / 1000 : 0;
    lastNow = time;
    const { seq, t } = locate(playFrames);
    housekeeping(seq);
    const g = ensureGpu(seq);
    const pose = frame.getViewerPose(stage.refSpace);
    xrFrames++;
    if (pose && !placed?.locked) {
      const entry = currentCam();
      const dist = scene.orbit.dist * distScale;
      const fwd: [number, number, number] = [entry.view[2], entry.view[6], entry.view[10]];
      const center: [number, number, number] = [entry.campos[0] + fwd[0] * dist, entry.campos[1] + fwd[1] * dist, entry.campos[2] + fwd[2] * dist];
      const hm = pose.transform.matrix;
      const up = scene.orbit.up;
      const next = place(entry.campos, center, [up[0], up[1], up[2]], [hm[12], hm[13], hm[14]], [-hm[8], -hm[9], -hm[10]], xrDist);
      if (next) {
        next.locked = !pose.emulatedPosition || xrFrames > 30;
        placed = next;
      }
    }
    if (placed?.locked) stage.pinch(frame, placed, time);
    if (g && inflight >= 2) {
      gpuBound++;
      if (playing) playFrames += dt * scene.fps;
    } else if (g && pose && placed) {
      try {
        const eyes = stage.eyes(pose, placed);
        if (eyes.some((e) => e.occ)) xrLog.occ++;
        if (eyes.length) {
          renderer.frameEyes(g, t, eyes);
          inflight++;
          device.queue.onSubmittedWorkDone().then(() => { inflight--; });
          frames++;
          if (!firstFrameDone) {
            firstFrameDone = true;
            console.info(`p5: first frame submitted ${performance.now().toFixed(0)} ms`);
            device.queue.onSubmittedWorkDone().then(() => { marks.first_frame_ms = performance.now(); console.info(`p5: first frame done ${marks.first_frame_ms.toFixed(0)} ms`); });
          }
        }
      } catch (e) {
        if (!xrDrawError) { xrDrawError = true; console.error('MR', e); }
      }
      if (playing) playFrames += dt * scene.fps;
    } else if (playing) {
      stalls++;
    }
    markFps(time);
    hud = () => { overlay.textContent = statusText(seq, t, g ? g.ps.n : 0); };
    if (frames % 15 === 0) hud();
    if (time - xrLog.at >= 5000) {
      const last = timings[timings.length - 1];
      console.info(`MR: ${fpsWin.length} fps, ${gpuBound - xrLog.busy} frames skipped (GPU busy), ${stalls - xrLog.stalls} stalls (section not ready)` +
        (last ? `, last decode ${last.decode_ms.toFixed(0)} ms seq ${last.seq}` : '') + `, ${resident ? `resident ${resident.size}/${nsec}` : 'streaming'}` +
        `, depth occlusion ${stage.depthSensing ? (stage.occlude ? `${xrLog.occ} frames` : 'off') : 'unavailable'}`);
      xrLog.at = time; xrLog.busy = gpuBound; xrLog.stalls = stalls; xrLog.occ = 0;
    }
  };
  const xrLog = { at: 0, busy: 0, stalls: 0, occ: 0 };
  const occBox = document.getElementById('occ') as HTMLInputElement;
  const occDebugBox = document.getElementById('occ-debug') as HTMLInputElement;
  occBox.checked = setting('occlude') !== '0';
  occDebugBox.checked = setting('occdebug') === '1';
  occBox.addEventListener('change', () => { saveSetting('occlude', occBox.checked ? '1' : '0'); if (xr) xr.occlude = occBox.checked; });
  occDebugBox.addEventListener('change', () => { saveSetting('occdebug', occDebugBox.checked ? '1' : '0'); if (xr) xr.occDebug = occDebugBox.checked; });
  if (xrReady) {
    xrButton.hidden = false;
    document.getElementById('occ-box')!.hidden = false;
    document.getElementById('occ-debug-box')!.hidden = false;
    document.getElementById('recolor-box')!.hidden = !navigator.mediaDevices?.getUserMedia;
    xrButton.addEventListener('click', () => {
      if (xr) { void xr.session.end(); return; }
      xrButton.disabled = true;
      xrButton.textContent = '…';
      void (async () => {
        try {
          const stage = await ImmersiveAR.start(device, xrLayerScale);
          stage.occlude = occBox.checked;
          stage.occDebug = occDebugBox.checked;
          stage.session.addEventListener('end', () => {
            xr = null;
            placed = null;
            xrFrames = 0;
            rafOn = true;
            lastNow = 0;
            renderer.setGrade(null);
            xrButton.textContent = 'MR';
            requestAnimationFrame(tick);
          });
          placed = null;
          xrFrames = 0;
          xrDrawError = false;
          rafOn = false;
          lastNow = 0;
          xr = stage;
          xrButton.textContent = 'Exit';
          stage.session.requestAnimationFrame(onXR);
          if (recolorBox.checked) void applyRecolor(stage);
        } catch (e) {
          console.error('MR', e);
          overlay.textContent = 'MR: ' + (e instanceof Error ? e.message : e);
          xrButton.textContent = 'MR';
        } finally {
          xrButton.disabled = false;
        }
      })();
    });
  }

  // ------------------------------------------------------------ bench hooks (Playwright)
  const allLoaded = async () => {
    const c = curSeq;
    for (let s = c; s <= c + LOOKAHEAD; s++) await whenDecoded(s);
  };
  const p5 = {
    ready: true,
    info: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, timestamps: hasTs, format },
    marks, timings, scene: { nsec, total, fps: scene.fps, targetN },
    state: () => ({ frames, stalls, playFrames, visible: renderer.lastVisible, gpu_bytes: renderer.gpuBytes(), gpu_bytes_peak: renderer.peakBytes,
      gpu_bytes_by: Object.fromEntries(renderer.bytes), sort_dispatches: renderer.sortDispatches(), canvas: [canvas.width, canvas.height] }),
    setView: (o: { az?: number; elev?: number; zoom?: number; t?: number; playing?: boolean }) => {
      if (o.az !== undefined) az = (o.az * Math.PI) / 180;
      if (o.elev !== undefined) elev = (o.elev * Math.PI) / 180;
      if (o.zoom !== undefined) distScale = o.zoom;
      if (o.t !== undefined) {                      // seek forward only (the stream decoder is sequential)
        let pf = Math.floor(playFrames / total) * total + (o.t % total);
        if (pf < playFrames) pf += total;
        playFrames = pf;
      }
      if (o.playing !== undefined) playing = o.playing;
    },
    /** rAF-paced playback for `seconds`; returns the display-paced frame rate. */
    async rafFps(seconds: number) {
      await allLoaded();
      playing = true;
      const f0 = frames, t0 = performance.now(), s0 = stalls, b0 = gpuBound;
      renderer.timingsLog.length = 0;
      await new Promise((r) => setTimeout(r, seconds * 1000));
      const el = (performance.now() - t0) / 1000;
      return { fps: (frames - f0) / el, frames: frames - f0, seconds: el, stalls: stalls - s0, skipped_gpu_busy: gpuBound - b0, gpu_timings: summarize(renderer.timingsLog) };
    },
    /** frames back-to-back without vsync (max 2 in flight); measures what the GPU sustains. */
    async uncapped(nFrames: number, timing: boolean, maxInflight = 3) {
      await allLoaded();
      rafOn = false;
      await device.queue.onSubmittedWorkDone();
      const off = device.createTexture({ size: [canvas.width, canvas.height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
      renderer.timingsLog.length = 0;
      const inflight: Promise<void>[] = [];
      const t0 = performance.now();
      let pf = playFrames;
      for (let i = 0; i < nFrames; i++) {
        const { seq, t } = locate(pf);
        housekeeping(seq);
        let g = ensureGpu(seq);
        if (!g) { await whenDecoded(seq); g = ensureGpu(seq); }
        if (!g) throw new Error(`section seq ${seq} not available`);
        renderer.frame(g, t, currentCam(), off, { timing });
        inflight.push(device.queue.onSubmittedWorkDone());
        if (inflight.length >= maxInflight) await inflight.shift();
        pf += scene.fps / 60;                     // content advances as at 60 Hz display
      }
      await Promise.all(inflight);
      const el = (performance.now() - t0) / 1000;
      await new Promise((r) => setTimeout(r, 100));
      off.destroy();
      playFrames = pf;
      rafOn = true; lastNow = 0;
      requestAnimationFrame(tick);
      return { fps: nFrames / el, frames: nFrames, seconds: el, max_inflight: maxInflight, gpu_timings: summarize(renderer.timingsLog) };
    },
    /** offscreen mode only: copy the current float target into a 2D canvas over the WebGPU canvas (screenshots). */
    async snapshot() {
      rafOn = false;
      await device.queue.onSubmittedWorkDone();
      hud();
      const img = await renderer.readTarget();
      const c2 = document.createElement('canvas');
      c2.width = img.w; c2.height = img.h;
      c2.style.cssText = 'position:fixed;left:0;top:0;width:100vw;height:100vh';
      const id = new ImageData(img.w, img.h);
      for (let i = 0; i < img.w * img.h; i++) { id.data.set(img.rgb.subarray(i * 3, i * 3 + 3), i * 4); id.data[i * 4 + 3] = 255; }
      c2.getContext('2d')!.putImageData(id, 0, 0);
      document.body.insertBefore(c2, document.getElementById('hud'));
      return { w: img.w, h: img.h };
    },
    /** render a test camera at frame t into the float target and return 8-bit RGB (base64). */
    async renderTest(idx: number, t: number) {
      await allLoaded();
      rafOn = false;
      await device.queue.onSubmittedWorkDone();
      const c = scene.testcams[idx];
      t = ((t % total) + total) % total;
      const si = scene.sections.findIndex((s) => t >= s.g0 && t < s.g0 + s.nframes);
      let seq = Math.floor(curSeq / nsec) * nsec + si;
      if (seq < curSeq) seq += nsec;
      const tt = t;
      await whenDecoded(seq);
      const g = ensureGpu(seq);
      if (!g) throw new Error(`section seq ${seq} not available (cur ${curSeq}, maxDecoded ${maxDecoded}, decoded [${[...decoded.keys()]}], gpu [${[...gpu.keys()]}])`);
      renderer.frame(g, tt, camFromRt(c.R, c.t, c.fx, c.fy, c.cx, c.cy, c.w, c.h), null);
      const img = await renderer.readTarget();
      // GPU eval vs CPU mirror on 256 splats
      const ids = Array.from({ length: Math.min(256, g.ps.n) }, (_, k) => Math.floor((k * g.ps.n) / 256));
      const got = await renderer.readWorld(ids);
      let maxErr = 0, checked = 0;
      ids.forEach((i, k) => {
        const ref = evalSplat(g.ps, i, tt);
        if (!ref.alive) return;
        checked++;
        const flat = [...ref.vals[0], ...ref.vals[1], ...ref.vals[2], ...ref.vals[3], ...ref.vals[4]];
        const gpuv = [got[k][0], got[k][1], got[k][2], got[k][4], got[k][5], got[k][6], got[k][7], got[k][8], got[k][9], got[k][10], got[k][11], got[k][12], got[k][13], got[k][14]];
        flat.forEach((v, j) => { maxErr = Math.max(maxErr, Math.abs(v - gpuv[j]) / Math.max(1, Math.abs(v))); });
      });
      rafOn = true; lastNow = 0; requestAnimationFrame(tick);
      let bin = '';
      for (let i = 0; i < img.rgb.length; i += 0x8000) bin += String.fromCharCode(...img.rgb.subarray(i, i + 0x8000));
      return { w: img.w, h: img.h, rgb_b64: btoa(bin), eval_check: { splats: checked, max_rel_err: maxErr }, n: g.ps.n, seq, t: tt };
    },
  };
  (window as any).__p5 = p5;
}

function summarize(log: Record<string, number>[]) {
  if (!log.length) return null;
  const out: Record<string, { median: number; p95: number; mean: number }> = {};
  for (const k of Object.keys(log[0])) {
    const v = log.map((r) => r[k]).filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
    if (!v.length) continue;
    out[k] = { median: v[Math.floor(v.length / 2)], p95: v[Math.floor(v.length * 0.95)], mean: v.reduce((a, b) => a + b, 0) / v.length };
  }
  return { samples: log.length, ...out };
}

main().catch((e) => fail(String(e?.stack ?? e)));
