// Decoder worker: fetch (prefetched, parallel) -> rANS/.csgs decode -> pack (strictly in sequence order).
import { Decoder } from './csgs.ts';
import { packSection } from './pack.ts';

const dec = new Decoder();
let chain: Promise<void> = Promise.resolve();

interface Job { seq: number; url: string; targetN: number; buf?: ArrayBuffer }   // buf: already fetched by the page

self.onmessage = (e: MessageEvent<Job>) => {
  const job = e.data;
  const tq = performance.now();
  const fetched = job.buf ? Promise.resolve({ b: new Uint8Array(job.buf), t: tq }) : fetch(job.url).then(async (r) => {
    if (!r.ok) throw new Error(`${job.url}: HTTP ${r.status}`);
    const b = new Uint8Array(await r.arrayBuffer());
    return { b, t: performance.now() };
  });
  chain = chain.then(async () => {
    try {
      const { b, t: tf } = await fetched;
      const t0 = performance.now();
      const sec = dec.decode(b);
      const t1 = performance.now();
      const ps = packSection(sec, dec.params!, job.targetN);
      const t2 = performance.now();
      const timing = {
        bytes: b.length, key: sec.key, n_coded: sec.n, n_cont: sec.nCont, nframes: sec.nframes,
        fetch_ms: tf - tq, wait_ms: t0 - tf, decode_ms: t1 - t0, pack_ms: t2 - t1,
      };
      (self as unknown as Worker).postMessage({ seq: job.seq, ps, timing }, [ps.hdr.buffer, ps.knots.buffer, ps.sn.buffer]);
    } catch (err) {
      (self as unknown as Worker).postMessage({ seq: job.seq, error: String(err) });
    }
  });
};
