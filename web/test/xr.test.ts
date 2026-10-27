// Headset pose -> OpenCV camera. Scene up stays world up; metres scale the eye, not the view rotation.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { applyGrab, eyeCamera, grabState, intrinsics, occlusionData, pinchClosed, place, stepGrab, type Placement } from '../src/xr.ts';

function near(a: number, b: number, eps = 1e-5) {
  assert.ok(Math.abs(a - b) <= eps, `${a} vs ${b}`);
}

function pose(x: number, y: number, z: number) {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  m[12] = x; m[13] = y; m[14] = z;
  return m;
}

function proj(m0: number, m5: number, m8 = 0, m9 = 0) {
  const p = new Float32Array(16);
  p[0] = m0; p[5] = m5; p[8] = m8; p[9] = m9; p[11] = -1;
  return p;
}

function apply(view: Float32Array, p: [number, number, number]) {
  return [
    view[0] * p[0] + view[4] * p[1] + view[8] * p[2] + view[12],
    view[1] * p[0] + view[5] * p[1] + view[9] * p[2] + view[13],
    view[2] * p[0] + view[6] * p[1] + view[10] * p[2] + view[14],
  ];
}

const cam: [number, number, number] = [0, 0, 0];
const center: [number, number, number] = [0, 0, 1];
const up: [number, number, number] = [0, 1, 0];
const head: [number, number, number] = [0, 1.6, 0];

test('intrinsics from a WebXR projection matrix', () => {
  const k = intrinsics(proj(1, 2), 200, 100);
  near(k.fx, 100); near(k.fy, 100); near(k.cx, 100); near(k.cy, 50);
  const off = intrinsics(proj(1, 1, 0.5, -0.25), 200, 100);
  near(off.fx, 100); near(off.fy, 50); near(off.cx, 50); near(off.cy, 37.5);
});

test('anchor: upright scene, centre in front, view is a rotation', () => {
  const placement = place(cam, center, up, head, [0, 0, -1], 1.5);
  assert.ok(placement);
  near(placement.scale, 1.5);
  assert.equal(placement.locked, false);
  const eye = eyeCamera(placement, proj(1, 1), pose(0, 1.6, 0), 200, 100);
  near(eye.campos[0], 0); near(eye.campos[1], 0); near(eye.campos[2], 0);
  near(eye.fx, 100); near(eye.fy, 50); near(eye.cx, 100); near(eye.cy, 50);
  const at = apply(eye.view, center);
  near(at[0], 0); near(at[1], 0); near(at[2], 1);
  const lifted = apply(eye.view, [0, 1, 0]);
  near(lifted[1], -1);
  const right = apply(eye.view, [-1, 0, 1]);
  near(right[0], 1); near(right[1], 0); near(right[2], 1);
  const r0 = [eye.view[0], eye.view[4], eye.view[8]];
  const r1 = [eye.view[1], eye.view[5], eye.view[9]];
  const r2 = [eye.view[2], eye.view[6], eye.view[10]];
  const len = (v: number[]) => Math.hypot(...v);
  near(len(r0), 1); near(len(r1), 1); near(len(r2), 1);
  near(r0[0] * r1[0] + r0[1] * r1[1] + r0[2] * r1[2], 0);
  const det = r0[0] * (r1[1] * r2[2] - r1[2] * r2[1]) - r0[1] * (r1[0] * r2[2] - r1[2] * r2[0]) + r0[2] * (r1[0] * r2[1] - r1[1] * r2[0]);
  near(det, 1);
});

test('depth occlusion: a splat fragment lands on the depth texel the sensor sees it in', () => {
  const W = 200, H = 100, zn = 0.1, zf = 20;
  const p = proj(1, 2, 0.2, -0.1);
  p[10] = zf / (zn - zf); p[14] = (zn * zf) / (zn - zf);
  const k = intrinsics(p, W, H);
  const eyeCam = { view: new Float32Array(16), campos: [0, 0, 0] as [number, number, number], ...k, width: W, height: H };
  const eyePose = pose(0.3, 1.6, -0.2);
  const ident = pose(0, 0, 0);
  const fold = (depthInv: Float32Array, u: number, v: number, z: number, scale: number) => {
    const d = occlusionData(eyeCam, scale, eyePose, depthInv, p, ident, 0.001, 'uint', 64, 32);
    const w = [u * z, v * z, z, 1];
    const h = [0, 1, 3].map((r) => d[r] * w[0] + d[4 + r] * w[1] + d[8 + r] * w[2] + d[12 + r] * w[3]);
    return { d, uv: [h[0] / h[2], h[1] / h[2]], z: d[16] * w[0] + d[17] * w[1] + d[18] * w[2] + d[19] * w[3] };
  };
  const same = pose(-0.3, -1.6, 0.2);
  const a = fold(same, 50, 80, 4, 0.5);
  near(a.uv[0], 50 / W); near(a.uv[1], 80 / H); near(a.z, -2);
  near(a.d[20], 0.001); near(a.d[21], 2); near(a.d[24], zn);

  const right = pose(-0.35, -1.6, 0.2);
  const b = fold(right, 120, 30, 3, 0.5);
  const zm = 1.5;
  const pd = [(120 - k.cx) / k.fx * zm - 0.05, -(30 - k.cy) / k.fy * zm, -zm];
  const cw = -pd[2];
  const ndc = [(p[0] * pd[0] + p[8] * pd[2]) / cw, (p[5] * pd[1] + p[9] * pd[2]) / cw];
  near(b.uv[0], ndc[0] * 0.5 + 0.5); near(b.uv[1], 0.5 - ndc[1] * 0.5); near(b.z, -zm);
});

function room(p: Placement, scene: [number, number, number]): [number, number, number] {
  const d: [number, number, number] = [scene[0] - p.cam[0], scene[1] - p.cam[1], scene[2] - p.cam[2]];
  return [
    p.anchor[0] + p.scale * (p.R[0] * d[0] + p.R[3] * d[1] + p.R[6] * d[2]),
    p.anchor[1] + p.scale * (p.R[1] * d[0] + p.R[4] * d[1] + p.R[7] * d[2]),
    p.anchor[2] + p.scale * (p.R[2] * d[0] + p.R[5] * d[1] + p.R[8] * d[2]),
  ];
}

function viewDet(view: Float32Array) {
  const r0 = [view[0], view[4], view[8]];
  const r1 = [view[1], view[5], view[9]];
  const r2 = [view[2], view[6], view[10]];
  return r0[0] * (r1[1] * r2[2] - r1[2] * r2[1]) - r0[1] * (r1[0] * r2[2] - r1[2] * r2[0]) + r0[2] * (r1[0] * r2[1] - r1[1] * r2[0]);
}

test('pinch: midpoint translates, hands scale and yaw about the grab, view stays a rotation', () => {
  const placement = place(cam, center, up, head, [0, 0, -1], 1.5);
  assert.ok(placement);
  const centre = room(placement, center);
  near(centre[0], 0); near(centre[1], 1.6); near(centre[2], -1.5);

  const shift: [number, number, number] = [0.03, -0.01, 0.02];
  const left: [number, number, number] = [-0.1, 1.4, -1.5];
  const right: [number, number, number] = [0.1, 1.4, -1.5];
  assert.equal(applyGrab(placement, [left, right], [
    [left[0] + shift[0], left[1] + shift[1], left[2] + shift[2]],
    [right[0] + shift[0], right[1] + shift[1], right[2] + shift[2]],
  ]), true);
  const moved = room(placement, center);
  near(moved[0], centre[0] + shift[0]); near(moved[1], centre[1] + shift[1]); near(moved[2], centre[2] + shift[2]);
  near(placement.scale, 1.5);

  const mid: [number, number, number] = [moved[0], moved[1], moved[2]];
  const grown = applyGrab(placement, [
    [mid[0] - 0.1, mid[1], mid[2]],
    [mid[0] + 0.1, mid[1], mid[2]],
  ], [
    [mid[0] - 0.104, mid[1], mid[2]],
    [mid[0] + 0.104, mid[1], mid[2]],
  ]);
  assert.equal(grown, true);
  near(placement.scale, 1.5 * 1.04);
  const scaledCentre = room(placement, center);
  near(scaledCentre[0], mid[0]); near(scaledCentre[1], mid[1]); near(scaledCentre[2], mid[2]);
  const scaledCam = room(placement, cam);
  near(scaledCam[0], mid[0]); near(scaledCam[1], mid[1]); near(scaledCam[2] - mid[2], 1.5 * 1.04);

  const yaw = 0.05;
  const half = 0.1;
  const sy = Math.sin(yaw);
  const cy = Math.cos(yaw);
  assert.equal(applyGrab(placement, [
    [mid[0], mid[1], mid[2] - half],
    [mid[0], mid[1], mid[2] + half],
  ], [
    [mid[0] - sy * half, mid[1], mid[2] - cy * half],
    [mid[0] + sy * half, mid[1], mid[2] + cy * half],
  ]), true);
  const spunCentre = room(placement, center);
  near(spunCentre[0], mid[0]); near(spunCentre[1], mid[1]); near(spunCentre[2], mid[2]);
  const spunCam = room(placement, cam);
  const reach = 1.5 * 1.04;
  near(spunCam[0], mid[0] + sy * reach); near(spunCam[1], mid[1]); near(spunCam[2], mid[2] + cy * reach);
  near(placement.R[3], 0); near(placement.R[4], 1); near(placement.R[5], 0);
  const eye = eyeCamera(placement, proj(1, 1), pose(spunCentre[0], spunCentre[1], spunCentre[2]), 64, 64);
  near(eye.campos[0], center[0]); near(eye.campos[1], center[1]); near(eye.campos[2], center[2]);
  near(viewDet(eye.view), 1);
});

test('pinch drops a hand teleport and a scale or yaw jump, and clamps scale', () => {
  const placement = place(cam, center, up, head, [0, 0, -1], 1.5);
  assert.ok(placement);
  const before = placement.anchor.slice();
  const left: [number, number, number] = [0, 1, 0];
  const right: [number, number, number] = [0.2, 1, 0];
  assert.equal(applyGrab(placement, [left, right], [[0.06, 1, 0], right]), false);
  near(placement.anchor[0], before[0]); near(placement.scale, 1.5);

  assert.equal(applyGrab(placement, [left, right], [[0.01, 1, 0], [0.25, 1, 0]]), true);
  near(placement.scale, 1.5);
  near(placement.anchor[0], before[0] + 0.03);

  const yawed = place(cam, center, up, head, [0, 0, -1], 1.5);
  assert.ok(yawed);
  const r0 = yawed.R.slice();
  const s = Math.sin(0.25);
  const c = Math.cos(0.25);
  assert.equal(applyGrab(yawed, [[0, 1, 0], [0, 1, 0.12]], [[0, 1, 0], [0.12 * s, 1, 0.12 * c]]), true);
  for (let i = 0; i < 9; i++) near(yawed.R[i], r0[i]);

  const limited = place(cam, center, up, head, [0, 0, -1], 1.5);
  assert.ok(limited);
  limited.scale = limited.scale0 * 19.5;
  const pivot: [number, number, number] = [0, 1, 0];
  assert.equal(applyGrab(limited, [[pivot[0] - 0.1, 1, 0], [pivot[0] + 0.1, 1, 0]], [[pivot[0] - 0.104, 1, 0], [pivot[0] + 0.104, 1, 0]]), true);
  near(limited.scale, limited.scale0 * 20);
});

test('pinch arms for 400 ms, and a dip during the arm cancels below 7.8 cm', () => {
  const placement = place(cam, center, up, head, [0, 0, -1], 1.5);
  assert.ok(placement);
  const gate = grabState();
  const a: [number, number, number] = [0, 1, 0];
  const b: [number, number, number] = [0.2, 1, 0];
  stepGrab(gate, a, b, 1000, placement);
  assert.equal(gate.placing, false);
  assert.equal(gate.armAt, 1000);
  stepGrab(gate, a, [0.08, 1, 0], 1200, placement);
  assert.equal(gate.armAt, 1000);
  stepGrab(gate, a, [0.05, 1, 0], 1300, placement);
  assert.equal(gate.armAt, null);

  const again = grabState();
  stepGrab(again, a, b, 1000, placement);
  stepGrab(again, a, b, 1399, placement);
  assert.equal(again.placing, false);
  stepGrab(again, a, b, 1400, placement);
  assert.equal(again.placing, true);
  near(placement.anchor[0], head[0]);
  stepGrab(again, [0.03, 1, 0], [0.23, 1, 0], 1420, placement);
  near(placement.anchor[0], head[0] + 0.03);
});

test('pinch hysteresis', () => {
  assert.equal(pinchClosed(false, 0.01, 0.01), true);
  assert.equal(pinchClosed(false, 0.03, 0.01), false);
  assert.equal(pinchClosed(true, 0.03, 0.01), true);
  assert.equal(pinchClosed(true, 0.05, 0.01), false);
  assert.equal(pinchClosed(false, 0.02, 0), true);
  assert.equal(pinchClosed(false, 0.04, 0), false);
  assert.equal(pinchClosed(true, 0.05, 0), true);
  assert.equal(pinchClosed(true, 0.06, 0), false);
});

test('walking the framed distance reaches the centre; a pitched head does not roll the scene', () => {
  const level = place(cam, center, up, head, [0, 0, -1], 1.5);
  const pitched = place(cam, center, up, head, [0, -0.5, -1], 1.5);
  assert.ok(level && pitched);
  for (let i = 0; i < 9; i++) near(level.R[i], pitched.R[i]);
  const eye = eyeCamera(level, proj(1, 1), pose(0, 1.6, -1.5), 64, 64);
  near(eye.campos[0], 0); near(eye.campos[1], 0); near(eye.campos[2], 1);
  const at = apply(eye.view, center);
  near(at[0], 0); near(at[1], 0); near(at[2], 0);
});
