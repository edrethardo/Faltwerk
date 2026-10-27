// Cameras (OpenCV: x right, y down, z forward) shared by the comparison viewer.
import type { Camera } from './renderer.ts';

export interface TestCam { idx: number; name: string; R: number[][]; t: number[]; fx: number; fy: number; cx: number; cy: number; w: number; h: number }
export interface Orbit { center: number[]; up: number[]; front: number[]; dist: number; elev_deg: number; fovy_deg: number }
export interface View { az: number; elev: number; zoom: number; panX: number; panY: number }

const sub = (a: number[], b: number[]) => a.map((v, i) => v - b[i]);
const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: number[]) => { const l = Math.hypot(...a); return a.map((v) => v / l); };

export function lookAt(pos: number[], target: number[], up: number[]): number[][] {
  const z = norm(sub(target, pos));
  const x = norm(cross(z, up));
  const y = cross(z, x);
  return [x, y, z];
}

export function camFromRt(R: number[][], t: number[], fx: number, fy: number, cx: number, cy: number, w: number, h: number): Camera {
  const view = new Float32Array(16);
  for (let r = 0; r < 3; r++) { for (let c = 0; c < 3; c++) view[c * 4 + r] = R[r][c]; view[12 + r] = t[r]; }
  view[15] = 1;
  const campos = [0, 1, 2].map((c) => -(R[0][c] * t[0] + R[1][c] * t[1] + R[2][c] * t[2])) as [number, number, number];
  return { view, campos, fx, fy, cx, cy, width: w, height: h };
}

/** orbit around o.center; panX/panY shift the target in the camera plane (units of the orbit distance).
 *  The focal length fits a test-camera-shaped frame (aspect = its w/h, vertical fov o.fovy_deg) into the pane (contain),
 *  so narrow panes (phone, 2 side by side) do not crop the subject. */
export function orbitCam(o: Orbit, v: View, w: number, h: number, aspect = 0): Camera {
  const up = o.up, f = o.front;
  const s = cross(up, f);
  const ce = Math.cos(v.elev), se = Math.sin(v.elev), ca = Math.cos(v.az), sa = Math.sin(v.az);
  const d = o.dist * v.zoom;
  const dir = [0, 1, 2].map((i) => ce * (ca * f[i] + sa * s[i]) + se * up[i]);
  let R = lookAt(dir.map((x, i) => o.center[i] + d * x), o.center, up);
  const target = o.center.map((c, i) => c + o.dist * (v.panX * R[0][i] + v.panY * R[1][i]));
  const pos = target.map((c, i) => c + d * dir[i]);
  R = lookAt(pos, target, up);
  const t = R.map((row) => -(row[0] * pos[0] + row[1] * pos[1] + row[2] * pos[2]));
  const ty = Math.tan((o.fovy_deg * Math.PI) / 360);
  const fy = aspect > 0 ? Math.min((0.5 * h) / ty, (0.5 * w) / (ty * aspect)) : (0.5 * h) / ty;
  return camFromRt(R, t, fy, fy, w / 2, h / 2, w, h);
}

/** test camera image (c.w x c.h) fitted into a w x h pane (contain, centred): same mapping as the GT pane. */
export function fitRect(c: TestCam, w: number, h: number) {
  const k = Math.min(w / c.w, h / c.h);
  return { k, ox: (w - c.w * k) / 2, oy: (h - c.h * k) / 2, dw: c.w * k, dh: c.h * k };
}

export function snapCam(c: TestCam, w: number, h: number): Camera {
  const r = fitRect(c, w, h);
  return camFromRt(c.R, c.t, c.fx * r.k, c.fy * r.k, c.cx * r.k + r.ox, c.cy * r.k + r.oy, w, h);
}

export function camKey(c: Camera): string {
  return Array.from(c.view).map((x) => x.toFixed(5)).join(',') + `|${(c.fx / c.width).toFixed(5)}`;
}
