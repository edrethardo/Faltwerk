// Room vs content statistics -> grade. Identical images stay neutral; a warm, dim, flat room warms, darkens and flattens.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { estimateGrade, gradeGain, imageStats, NEUTRAL } from '../src/recolor.ts';

function image(fn: (i: number) => [number, number, number], n = 256): Uint8Array {
  const px = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = fn(i);
    px.set([r, g, b, 255], i * 4);
  }
  return px;
}

const ramp = (gain: [number, number, number], lo: number, hi: number) =>
  image((i) => {
    const v = lo + ((hi - lo) * (i % 16)) / 15;
    const hue = [1 + 0.3 * Math.sin(i), 1 + 0.3 * Math.cos(i), 1 - 0.3 * Math.sin(i)];
    return gain.map((k, c) => Math.round(Math.min(250, v * k * hue[c]))) as [number, number, number];
  });

test('same image gives the neutral grade', () => {
  const s = imageStats(ramp([1, 1, 1], 30, 200), 1 / 255)!;
  const g = estimateGrade(s, s, 1);
  for (const k of Object.keys(NEUTRAL) as (keyof typeof NEUTRAL)[]) assert.ok(Math.abs(g[k] - NEUTRAL[k]) < 1e-9, k);
});

test('warm dim low-contrast room', () => {
  const content = imageStats(ramp([1, 1, 1], 30, 220), 1 / 255)!;
  const room = imageStats(ramp([1.1, 0.95, 0.7], 50, 120), 1 / 255)!;
  const g = estimateGrade(room, content, 1);
  assert.ok(g.temperature > 0.2, `temperature ${g.temperature}`);
  assert.ok(g.exposure < 0, `exposure ${g.exposure}`);
  assert.ok(g.contrast < 1, `contrast ${g.contrast}`);
  const half = estimateGrade(room, content, 0.5);
  assert.ok(Math.abs(half.temperature - g.temperature / 2) < 1e-9);
  const [r, , b] = gradeGain(g);
  assert.ok(r > b);
});

test('premultiplied input ignores thin coverage and divides alpha out', () => {
  const opaque = imageStats(new Float32Array([0.5, 0.25, 0.1, 1]), 1, true)!;
  const mixed = imageStats(new Float32Array([0.25, 0.125, 0.05, 0.5, 0.9, 0.1, 0.1, 0.2]), 1, true)!;
  assert.deepEqual(mixed, opaque);
  assert.equal(imageStats(new Float32Array([1, 1, 1, 0.1]), 1, true), null);
});
