// Automatic grade for MR: match the splats to the room as the passthrough camera sees it.
// Parameters and gain model follow cheap-project client/grade.ts (ContentGrade / gradeRecolor).

import { tr } from './i18n.ts';

type V3 = [number, number, number];

export interface Grade {
  exposure: number;
  temperature: number;
  tint: number;
  saturation: number;
  contrast: number;
}

export const NEUTRAL: Grade = { exposure: 0, temperature: 0, tint: 0, saturation: 1, contrast: 1 };

export interface ImageStats {
  /** shades-of-gray illuminant (linear, luma 1) */
  illum: V3;
  /** mean ln linear luma */
  logY: number;
  /** mean |c − luma| in display space, what saturation scales */
  chroma: number;
  /** display luma standard deviation, what contrast scales */
  spread: number;
}

const LUMA: V3 = [0.2126, 0.7152, 0.0722];
const MINKOWSKI = 4;
const CLIPPED = 0.98;

/** cheap-project gradeRecolor: per-channel response to temperature t and tint g */
const WB_T: V3 = [0.35, 0, -0.4];
const WB_G: V3 = [-0.08, 0.22, -0.04];

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const toLinear = (v: number) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);

/**
 * RGBA pixels, display-encoded. `scale` maps raw values to 0..1; premultiplied input is divided by
 * alpha and weighted by it, with thin coverage (alpha < 0.5) skipped. null when nothing is left.
 */
export function imageStats(rgba: ArrayLike<number>, scale = 1, premultiplied = false, step = 1): ImageStats | null {
  let w = 0, wIllum = 0, logY = 0, chroma = 0, y1 = 0, y2 = 0;
  const p: V3 = [0, 0, 0];
  for (let i = 0; i + 3 < rgba.length; i += 4 * step) {
    const a = premultiplied ? rgba[i + 3] * scale : 1;
    if (a < 0.5) continue;
    const k = scale / a;
    const c: V3 = [clamp(rgba[i] * k, 0, 1), clamp(rgba[i + 1] * k, 0, 1), clamp(rgba[i + 2] * k, 0, 1)];
    const yd = c[0] * LUMA[0] + c[1] * LUMA[1] + c[2] * LUMA[2];
    chroma += a * (Math.abs(c[0] - yd) + Math.abs(c[1] - yd) + Math.abs(c[2] - yd)) / 3;
    y1 += a * yd;
    y2 += a * yd * yd;
    const l: V3 = [toLinear(c[0]), toLinear(c[1]), toLinear(c[2])];
    logY += a * Math.log(l[0] * LUMA[0] + l[1] * LUMA[1] + l[2] * LUMA[2] + 1e-3);
    w += a;
    if (Math.max(c[0], c[1], c[2]) >= CLIPPED) continue;
    for (let j = 0; j < 3; j++) p[j] += a * l[j] ** MINKOWSKI;
    wIllum += a;
  }
  if (w <= 0) return null;
  const e: V3 = wIllum > 0 ? (p.map((v) => (v / wIllum) ** (1 / MINKOWSKI)) as V3) : [1, 1, 1];
  const ey = e[0] * LUMA[0] + e[1] * LUMA[1] + e[2] * LUMA[2];
  const mean = y1 / w;
  return {
    illum: ey > 1e-6 ? (e.map((v) => v / ey) as V3) : [1, 1, 1],
    logY: logY / w,
    chroma: chroma / w,
    spread: Math.sqrt(Math.max(0, y2 / w - mean * mean)),
  };
}

/**
 * Grade that moves `content` toward `room`, blended by `strength` (0 neutral, 1 full match).
 * White balance solves the gradeRecolor model in log space up to a common gain; without content
 * stats the content is taken as neutral and only the white balance moves.
 */
export function estimateGrade(room: ImageStats, content: ImageStats | null, strength: number): Grade {
  const ref = content ?? { illum: [1, 1, 1] as V3, logY: room.logY, chroma: room.chroma, spread: room.spread };
  const lr = room.illum.map((v, i) => Math.log(Math.max(v, 1e-4) / Math.max(ref.illum[i], 1e-4)));
  const a = lr[0] - lr[1], b = lr[2] - lr[1];
  const tr = WB_T[0] - WB_T[1], gr = WB_G[0] - WB_G[1];
  const tb = WB_T[2] - WB_T[1], gb = WB_G[2] - WB_G[1];
  const det = tr * gb - gr * tb;
  const s = clamp(strength, 0, 1);
  const ratio = (num: number, den: number, lo: number, hi: number) => 1 + (clamp(den > 1e-4 ? num / den : 1, lo, hi) - 1) * s;
  return {
    exposure: s * clamp((room.logY - ref.logY) / Math.LN2, -2, 2),
    temperature: s * clamp((a * gb - gr * b) / det, -1, 1),
    tint: s * clamp((tr * b - a * tb) / det, -1, 1),
    saturation: ratio(room.chroma, ref.chroma, 0.5, 1.5),
    contrast: ratio(room.spread, ref.spread, 0.7, 1.3),
  };
}

/** Linear per-channel gain (exposure + white balance), as cheap-project gradeRecolor. */
export function gradeGain(g: Grade): V3 {
  const e = 2 ** g.exposure;
  const t = clamp(g.temperature, -1, 1);
  const n = clamp(g.tint, -1, 1);
  return [
    e * (1 + t * WB_T[0]) * (1 + n * WB_G[0]),
    e * (1 + n * WB_G[1]),
    e * (1 + t * WB_T[2]) * (1 + n * WB_G[2]),
  ];
}

export function describeGrade(g: Grade): string {
  const sg = (v: number) => (v >= 0 ? '+' : '') + v.toFixed(2);
  return `${tr('Bel', 'Exp')} ${sg(g.exposure)} · Temp ${sg(g.temperature)} · Tint ${sg(g.tint)} · ` +
    `${tr('Sätt', 'Sat')} ${g.saturation.toFixed(2)} · ${tr('Kontr', 'Contr')} ${g.contrast.toFixed(2)}`;
}

/** auto exposure / white balance of a freshly opened camera */
const SETTLE_MS = 700;
const SAMPLE_W = 160;
/** a flat frame (placeholder, black, or blown out) says nothing about the room */
const MIN_SPREAD = 0.02;

function withTimeout<T>(p: Promise<T>, ms: number, what: string, onLate?: (v: T) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    let done = false;
    const timer = setTimeout(() => { done = true; reject(new Error(`${what} timed out after ${ms}ms`)); }, ms);
    p.then(
      (v) => { if (done) { onLate?.(v); return; } done = true; clearTimeout(timer); resolve(v); },
      (e) => { if (done) return; done = true; clearTimeout(timer); reject(e instanceof Error ? e : new Error(String(e))); },
 );
  });
}

const stopStream = (s: MediaStream) => s.getTracks().forEach((t) => t.stop());

/** One frame of the headset's passthrough camera (Quest Browser exposes it through getUserMedia). */
export async function captureRoom(timeoutMs = 5000): Promise<ImageStats> {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('no camera access (getUserMedia)');
  const stream = await withTimeout(
    navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 640 } }, audio: false }),
    timeoutMs, 'getUserMedia', stopStream,
 );
  try {
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.srcObject = stream;
    await withTimeout(video.play(), timeoutMs, 'camera play');
    await new Promise((r) => setTimeout(r, SETTLE_MS));
    const vw = video.videoWidth || 4, vh = video.videoHeight || 3;
    const w = SAMPLE_W, h = Math.max(1, Math.round((SAMPLE_W * vh) / vw));
    const ctx = new OffscreenCanvas(w, h).getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('no 2d context');
    ctx.drawImage(video, 0, 0, w, h);
    const stats = imageStats(ctx.getImageData(0, 0, w, h).data, 1 / 255);
    if (!stats || stats.spread < MIN_SPREAD) throw new Error('camera frame is blank');
    return stats;
  } finally {
    stopStream(stream);
  }
}
